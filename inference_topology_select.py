"""Topology selector via ValueNet critic scoring (Option A).

For a given spec, runs all 6 topologies through the trained GNN and
scores each with V_psi(spec, z_topo). The ranking replaces an explicit
topology-choice policy — the value network IS the selector.

Usage
-----
# Score a spec from the specset by index:
python inference_topology_select.py --run runs_diffusion/run_20260521_185745 --spec-idx 42

# Score a custom spec passed as JSON:
python inference_topology_select.py --run runs_diffusion/run_20260521_185745 \
    --spec '{"fc_ghz":28,"bw_pct":20,"phase_coverage_deg":360,"phase_bits":5,
             "rms_phase_err_deg":5,"rms_gain_err_db":1,"max_il_db":5,
             "min_rl_db":10,"vdd":1.8,"pmax_mw":15,"tech":0,"app":2}'

# Use the most recent run automatically (omit --run):
python inference_topology_select.py --spec-idx 0

# Batch mode: score all specs in specset and print win-count table:
python inference_topology_select.py --run runs_diffusion/run_20260521_185745 --batch
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from env.graph_utils import get_topology_graph, TOPOLOGY_PARAMS
from models.gnn_encoder import TopologyEncoder
from models.diffusion_policy import ValueNet
from specset.generate_specset import SPEC_BOUNDS

TOPOLOGY_NAMES = list(TOPOLOGY_PARAMS.keys())
SPEC_KEYS = list(SPEC_BOUNDS.keys())


# ---------------------------------------------------------------------------
# Spec normalization (mirrors PhaseShifterEnv._normalize exactly)
# ---------------------------------------------------------------------------

def normalize_spec(spec: dict) -> torch.Tensor:
    vec = []
    for k in SPEC_KEYS:
        bnd = SPEC_BOUNDS[k]
        val = spec[k]
        if isinstance(bnd, tuple):
            mn, mx = bnd
            if k in ("fc_ghz", "pmax_mw"):
                mn, mx = np.log10(mn), np.log10(mx)
                val = np.log10(max(val, 1e-12))
            v = (val - mn) / (mx - mn)
            vec.append(float(np.clip(v, 0.0, 1.0)))
        elif isinstance(bnd, list):
            max_val = max(bnd) if max(bnd) > 0 else 1
            vec.append(float(val) / float(max_val))
    return torch.tensor(vec, dtype=torch.float).unsqueeze(0)  # [1, 12]


# ---------------------------------------------------------------------------
# Core: score all 6 topologies for one spec
# ---------------------------------------------------------------------------

def score_topologies(spec: dict, gnn: TopologyEncoder, value_net: ValueNet,
                     device: torch.device) -> list[tuple[str, float]]:
    """Return list of (topology_name, V_score) sorted best-first."""
    spec_tensor = normalize_spec(spec).to(device)  # [1, 12]
    results = []
    for name in TOPOLOGY_NAMES:
        g = get_topology_graph(name)
        x = g.x.to(device)
        edge_index = g.edge_index.to(device)
        with torch.no_grad():
            z_topo = gnn(x, edge_index)          # [1, 64]  (batch=None → global_mean_pool handles it)
            v = value_net(spec_tensor, z_topo)   # [1]
        results.append((name, v.item()))
    results.sort(key=lambda t: t[1], reverse=True)
    return results


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def find_latest_run(runs_dir: str) -> str:
    runs_dir = os.path.join(REPO_ROOT, runs_dir)
    candidates = sorted(
        [d for d in os.listdir(runs_dir) if os.path.isdir(os.path.join(runs_dir, d))],
        reverse=True,
    )
    for d in candidates:
        run_path = os.path.join(runs_dir, d)
        if os.path.exists(os.path.join(run_path, "gnn_encoder.pt")) and \
           os.path.exists(os.path.join(run_path, "value_net.pt")):
            return run_path
    raise FileNotFoundError(f"No complete run found in {runs_dir}")


def load_models(run_dir: str, device: torch.device):
    gnn = TopologyEncoder().to(device)
    value_net = ValueNet().to(device)
    gnn.load_state_dict(torch.load(os.path.join(run_dir, "gnn_encoder.pt"), map_location=device))
    value_net.load_state_dict(torch.load(os.path.join(run_dir, "value_net.pt"), map_location=device))
    gnn.eval()
    value_net.eval()
    return gnn, value_net


def print_ranking(spec: dict, ranked: list[tuple[str, float]]):
    print("\nSpec:")
    for k, v in spec.items():
        print(f"  {k}: {v}")
    print("\nTopology ranking (V_psi score, higher = better):")
    print(f"  {'Rank':<5} {'Topology':<22} {'V score':>10}")
    print(f"  {'-'*5} {'-'*22} {'-'*10}")
    for rank, (name, score) in enumerate(ranked, 1):
        marker = "  <-- BEST" if rank == 1 else ""
        print(f"  {rank:<5} {name:<22} {score:>10.4f}{marker}")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Select topology via ValueNet critic scoring.")
    parser.add_argument("--run", type=str, default=None,
                        help="Path to diffusion run dir (default: auto-detect latest).")
    parser.add_argument("--spec-idx", type=int, default=None,
                        help="Index into specset_phaseshifter.json. Mutually exclusive with --spec.")
    parser.add_argument("--spec", type=str, default=None,
                        help="JSON string with all 12 spec fields. Mutually exclusive with --spec-idx.")
    parser.add_argument("--batch", action="store_true",
                        help="Score all 600 specs and print topology win-count table.")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Torch device (default: cpu).")
    args = parser.parse_args()

    device = torch.device(args.device)

    # Resolve run directory
    run_dir = args.run
    if run_dir is None:
        run_dir = find_latest_run("runs_diffusion")
        print(f"[Auto] Using latest run: {run_dir}")
    elif not os.path.isabs(run_dir):
        run_dir = os.path.join(REPO_ROOT, run_dir)

    gnn, value_net = load_models(run_dir, device)
    print(f"[Loaded] gnn_encoder + value_net from {run_dir}")

    # Load specset for --batch or --spec-idx
    specset_path = os.path.join(REPO_ROOT, "specset/specset_phaseshifter.json")
    specset = []
    if os.path.exists(specset_path):
        with open(specset_path) as f:
            specset = json.load(f)

    # -----------------------------------------------------------------------
    # Batch mode: score all specs, report win counts
    # -----------------------------------------------------------------------
    if args.batch:
        if not specset:
            print("[Error] specset not found.")
            sys.exit(1)
        wins = {name: 0 for name in TOPOLOGY_NAMES}
        scores_by_topo = {name: [] for name in TOPOLOGY_NAMES}
        for entry in specset:
            ranked = score_topologies(entry["spec"], gnn, value_net, device)
            winner = ranked[0][0]
            wins[winner] += 1
            for name, score in ranked:
                scores_by_topo[name].append(score)

        total = len(specset)
        print(f"\nBatch topology selection over {total} specs:")
        print(f"  {'Topology':<22} {'Wins':>6} {'Win%':>7} {'Mean V':>10} {'Std V':>9}")
        print(f"  {'-'*22} {'-'*6} {'-'*7} {'-'*10} {'-'*9}")
        sorted_topos = sorted(TOPOLOGY_NAMES, key=lambda n: wins[n], reverse=True)
        for name in sorted_topos:
            sc = scores_by_topo[name]
            print(f"  {name:<22} {wins[name]:>6} {100*wins[name]/total:>6.1f}% "
                  f"{np.mean(sc):>10.4f} {np.std(sc):>9.4f}")
        print()
        return

    # -----------------------------------------------------------------------
    # Single spec mode
    # -----------------------------------------------------------------------
    if args.spec is not None and args.spec_idx is not None:
        print("[Error] --spec and --spec-idx are mutually exclusive.")
        sys.exit(1)

    if args.spec is not None:
        spec = json.loads(args.spec)
    elif args.spec_idx is not None:
        if not specset:
            print("[Error] specset not found.")
            sys.exit(1)
        entry = specset[args.spec_idx]
        spec = entry["spec"]
        print(f"[Spec] index {args.spec_idx} | heuristic label: {entry.get('topology', 'N/A')}")
    else:
        # Default: first spec from dataset
        if specset:
            spec = specset[0]["spec"]
            print("[Spec] Using specset index 0 (no --spec or --spec-idx given).")
        else:
            spec = {
                "fc_ghz": 28.0, "bw_pct": 30.0, "phase_coverage_deg": 360.0,
                "phase_bits": 5, "rms_phase_err_deg": 5.0, "rms_gain_err_db": 1.0,
                "max_il_db": 5.0, "min_rl_db": 10.0, "vdd": 1.8, "pmax_mw": 15.0,
                "tech": 0, "app": 2,
            }
            print("[Spec] Using hardcoded 28 GHz / 5-bit default (no specset found).")

    ranked = score_topologies(spec, gnn, value_net, device)
    print_ranking(spec, ranked)


if __name__ == "__main__":
    main()
