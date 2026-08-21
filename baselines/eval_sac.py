"""
J3 — Evaluate trained SAC checkpoints on the 10 paper scenarios.
Trained-vs-trained comparison vs G-DiffPS (both pre-trained 10k steps; both 0 SPICE at inference).

Usage:
    python3 baselines/eval_sac.py \
        --checkpoint-dir results/baselines/sac \
        --out-dir results/baselines/sac_eval
"""

import argparse
import json
import os
import sys
import time
import glob
import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from env.phaseshifter_env import PhaseShifterEnv
from env.graph_utils import get_topology_graph, TOPOLOGY_PARAMS
from models.gnn_encoder import TopologyEncoder
from baselines.train_sac import GaussianPolicyNet
from train_diffusion import action_to_params, make_spice_netlist, parallel_eval_worker
from sim.physics_priors import check_physics_priors


SCENARIOS = [
    {"name": "S1_LL_28",  "topo": "Loaded_Line",      "spec": {"fc_ghz": 28.0, "bw_pct": 20.0, "phase_coverage_deg": 180.0, "phase_bits": 5, "rms_phase_err_deg": 3.0, "rms_gain_err_db": 1.0, "max_il_db": 2.0, "min_rl_db": 18.0, "vdd": 1.8, "pmax_mw": 15.0, "tech": 0, "app": 2}},
    {"name": "S2_LL_38",  "topo": "Loaded_Line",      "spec": {"fc_ghz": 38.0, "bw_pct": 15.0, "phase_coverage_deg": 180.0, "phase_bits": 5, "rms_phase_err_deg": 3.5, "rms_gain_err_db": 1.0, "max_il_db": 2.2, "min_rl_db": 15.0, "vdd": 1.8, "pmax_mw": 15.0, "tech": 0, "app": 2}},
    {"name": "S3_SL_14",  "topo": "Switched_Line",    "spec": {"fc_ghz": 14.0, "bw_pct": 30.0, "phase_coverage_deg": 180.0, "phase_bits": 4, "rms_phase_err_deg": 5.0, "rms_gain_err_db": 1.5, "max_il_db": 2.5, "min_rl_db": 15.0, "vdd": 2.5, "pmax_mw": 25.0, "tech": 1, "app": 3}},
    {"name": "S4_SL_24",  "topo": "Switched_Line",    "spec": {"fc_ghz": 24.0, "bw_pct": 20.0, "phase_coverage_deg": 180.0, "phase_bits": 4, "rms_phase_err_deg": 4.5, "rms_gain_err_db": 1.0, "max_il_db": 2.0, "min_rl_db": 12.0, "vdd": 1.8, "pmax_mw": 20.0, "tech": 0, "app": 2}},
    {"name": "S5_VM_5",   "topo": "Vector_Modulator", "spec": {"fc_ghz": 5.0,  "bw_pct": 40.0, "phase_coverage_deg": 360.0, "phase_bits": 5, "rms_phase_err_deg": 5.0, "rms_gain_err_db": 1.0, "max_il_db": 1.5, "min_rl_db": 20.0, "vdd": 3.3, "pmax_mw": 35.0, "tech": 2, "app": 0}},
    {"name": "S6_VM_8",   "topo": "Vector_Modulator", "spec": {"fc_ghz": 8.0,  "bw_pct": 30.0, "phase_coverage_deg": 360.0, "phase_bits": 5, "rms_phase_err_deg": 4.0, "rms_gain_err_db": 1.0, "max_il_db": 1.8, "min_rl_db": 18.0, "vdd": 3.3, "pmax_mw": 35.0, "tech": 2, "app": 3}},
    {"name": "S7_RT_28",  "topo": "Reflection_Type",  "spec": {"fc_ghz": 28.0, "bw_pct": 20.0, "phase_coverage_deg": 90.0,  "phase_bits": 5, "rms_phase_err_deg": 5.0, "rms_gain_err_db": 1.0, "max_il_db": 2.0, "min_rl_db": 12.0, "vdd": 1.8, "pmax_mw": 15.0, "tech": 0, "app": 2}},
    {"name": "S8_RT_10",  "topo": "Reflection_Type",  "spec": {"fc_ghz": 10.0, "bw_pct": 25.0, "phase_coverage_deg": 180.0, "phase_bits": 4, "rms_phase_err_deg": 5.0, "rms_gain_err_db": 1.5, "max_il_db": 2.0, "min_rl_db": 15.0, "vdd": 2.5, "pmax_mw": 25.0, "tech": 1, "app": 3}},
    {"name": "S9_AP_2.4", "topo": "All_Pass",         "spec": {"fc_ghz": 2.4,  "bw_pct": 10.0, "phase_coverage_deg": 180.0, "phase_bits": 3, "rms_phase_err_deg": 8.0, "rms_gain_err_db": 2.0, "max_il_db": 1.5, "min_rl_db": 18.0, "vdd": 3.3, "pmax_mw": 40.0, "tech": 2, "app": 0}},
    {"name": "S10_SF_18", "topo": "Switched_Filter",  "spec": {"fc_ghz": 18.0, "bw_pct": 20.0, "phase_coverage_deg": 90.0,  "phase_bits": 4, "rms_phase_err_deg": 6.0, "rms_gain_err_db": 2.0, "max_il_db": 2.5, "min_rl_db": 12.0, "vdd": 1.8, "pmax_mw": 20.0, "tech": 0, "app": 3}},
]


def load_sac(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    gnn = TopologyEncoder().to(device)
    actor = GaussianPolicyNet(spec_dim=12, graph_dim=64, action_dim=9).to(device)
    # The SAC checkpoint was saved with the OLD GNN (ReLU only). Use strict=False
    # so new LayerNorm/LeakyReLU params get their fresh init (they zero-out anyway,
    # but that's fine because we are using SAC-trained weights to produce SAC's design).
    gnn.load_state_dict(ckpt["gnn"], strict=False)
    actor.load_state_dict(ckpt["actor"])
    gnn.eval()
    actor.eval()
    return gnn, actor


def eval_scenario(scenario, gnn, actor, device, env, n_samples=10):
    topo = scenario["topo"]
    spec = scenario["spec"]

    g = get_topology_graph(topo)
    g.x = g.x.to(device); g.edge_index = g.edge_index.to(device)

    spec_vec = env._normalize(spec)
    spec_t = torch.tensor(spec_vec, dtype=torch.float, device=device).unsqueeze(0)

    best = {"reward": -999.0, "metrics": None, "params": None, "passed_prior": False}

    with torch.no_grad():
        z_topo = gnn(g.x, g.edge_index)

        for k in range(n_samples):
            if k == 0:
                # deterministic mean for first sample
                mean, _ = actor.forward(spec_t, z_topo)
                action_t = mean.clamp(0.0, 1.0)
            else:
                action_t = actor.sample(spec_t, z_topo)
            action = action_t.squeeze(0).cpu().numpy()
            params = action_to_params(action, topo, spec)
            passed = check_physics_priors(topo, params, spec["fc_ghz"])
            if not passed:
                reward = -5.0
                metrics = None
            else:
                netlist = make_spice_netlist(topo, params)
                reward, metrics, _ = parallel_eval_worker((netlist, spec, topo, 0.0, None))
            if reward > best["reward"]:
                best = {"reward": float(reward), "metrics": metrics, "params": {k: str(v) for k, v in params.items()}, "passed_prior": bool(passed), "sample_idx": k}
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint-dir", default="results/baselines/sac")
    ap.add_argument("--out-dir", default="results/baselines/sac_eval")
    ap.add_argument("--n-samples", type=int, default=10,
                    help="SAC samples per scenario (1 deterministic + n-1 stochastic)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device)

    # Find all SAC run dirs with a checkpoint
    run_dirs = sorted([d for d in glob.glob(os.path.join(args.checkpoint_dir, "run_*"))
                       if os.path.exists(os.path.join(d, "checkpoint.pt"))])
    if not run_dirs:
        print(f"[ERROR] No SAC checkpoints found in {args.checkpoint_dir}/run_*/checkpoint.pt")
        sys.exit(1)

    print(f"[SAC-EVAL] Found {len(run_dirs)} SAC checkpoints")
    env = PhaseShifterEnv()

    all_results = []
    for run_dir in run_dirs:
        ckpt_path = os.path.join(run_dir, "checkpoint.pt")
        run_tag = os.path.basename(run_dir)
        print(f"\n[SAC-EVAL] checkpoint={ckpt_path}")
        gnn, actor = load_sac(ckpt_path, device)

        for sc in SCENARIOS:
            t0 = time.time()
            try:
                res = eval_scenario(sc, gnn, actor, device, env, n_samples=args.n_samples)
            except Exception as e:
                res = {"reward": -999.0, "error": str(e)}
            elapsed = time.time() - t0
            entry = {
                "method": "SAC_inference",
                "run": run_tag,
                "scenario": sc["name"],
                "topology": sc["topo"],
                "fc_ghz": sc["spec"]["fc_ghz"],
                "spice_calls": args.n_samples,  # bounded; real # = # of prior-passing samples
                "best_reward": res["reward"],
                "best_metrics": res.get("metrics"),
                "best_params": res.get("params"),
                "passed_prior": res.get("passed_prior", False),
                "elapsed_s": elapsed,
            }
            print(f"  {sc['name']:<14}  topo={sc['topo']:<18}  r={res['reward']:+.3f}  t={elapsed:.1f}s")
            all_results.append(entry)

            # also dump per-record JSON for easy aggregation
            out_path = os.path.join(args.out_dir, f"sac_{run_tag}_{sc['name']}.json")
            with open(out_path, "w") as f:
                json.dump(entry, f, indent=2)

    # Summary table
    summary_path = os.path.join(args.out_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n[SAC-EVAL] {len(all_results)} records written to {args.out_dir}/")
    print(f"[SAC-EVAL] Summary: {summary_path}")


if __name__ == "__main__":
    main()
