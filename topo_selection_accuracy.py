"""
Topology-selection accuracy: does the ValueNet head recommend the topology that
actually achieves the best SPICE reward?  (App C ground-truth validation.)

For each spec:
  1. ValueNet scores all 6 topologies -> ranked recommendation.
  2. CFM actor sizes each topology (k samples); SPICE-eval; take best reward/topo.
  3. Empirical-best topology = argmax_topo (best SPICE reward), among feasible.
  4. Record top-1 / top-2 agreement of ValueNet ranking with the empirical best,
     and agreement with the specset heuristic label.

Usage:
  python topo_selection_accuracy.py --run runs_diffusion/run_20260530_031117 \
      --n-specs 120 --k 8 --seed 42 --out results/topo_select/acc_seed42.json
"""
import argparse, json, os, sys, random
import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from env.phaseshifter_env import PhaseShifterEnv
from env.graph_utils import get_topology_graph
from models.gnn_encoder import TopologyEncoder
from models.diffusion_policy import FlowMatchingPolicy, ValueNet
from sim.physics_priors import check_physics_priors
from train_diffusion import action_to_params, make_spice_netlist, parallel_eval_worker

TOPOS = ["Loaded_Line", "Switched_Line", "Reflection_Type",
         "Switched_Filter", "Vector_Modulator", "All_Pass"]
FEASIBLE_REWARD = 0.5   # a topology counts as "achievable" for this spec above this


def load_models(run_dir, device):
    gnn = TopologyEncoder(in_channels=5, hidden_channels=64, out_channels=64).to(device)
    actor = FlowMatchingPolicy(action_dim=9, spec_dim=12, graph_dim=64).to(device)
    value_net = ValueNet(spec_dim=12, graph_dim=64).to(device)
    gnn.load_state_dict(torch.load(os.path.join(run_dir, "gnn_encoder.pt"), map_location=device))
    actor.load_state_dict(torch.load(os.path.join(run_dir, "actor.pt"), map_location=device))
    value_net.load_state_dict(torch.load(os.path.join(run_dir, "value_net.pt"), map_location=device))
    gnn.eval(); actor.eval(); value_net.eval()
    return gnn, actor, value_net


def best_spice_reward(actor, z, spec_norm, topo, spec, env, k):
    """Best SPICE reward over k CFM samples for one topology."""
    best = -999.0
    for _ in range(k):
        with torch.no_grad():
            a = actor.sample(spec_norm, z).squeeze(0).cpu().numpy()
        params = action_to_params(a, topo, spec)
        if not check_physics_priors(topo, params, spec["fc_ghz"]):
            continue
        eb = env.compute_expert_bonus(topo, spec)
        nl = make_spice_netlist(topo, params)
        r, _, _ = parallel_eval_worker((nl, spec, topo, eb, None))
        best = max(best, float(r))
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--n-specs", type=int, default=120)
    ap.add_argument("--k", type=int, default=8, help="CFM samples per topology")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="results/topo_select/acc.json")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)

    gnn, actor, value_net = load_models(args.run, device)
    env = PhaseShifterEnv()

    specset = json.load(open(os.path.join(REPO_ROOT, "specset/specset_phaseshifter.json")))
    idxs = list(range(len(specset)))
    random.shuffle(idxs)
    idxs = idxs[:args.n_specs]

    # Pre-encode the six topology graphs once.
    z_cache = {}
    for t in TOPOS:
        g = get_topology_graph(t)
        with torch.no_grad():
            z_cache[t] = gnn(g.x.to(device), g.edge_index.to(device))

    top1 = top2 = label_match = feasible_specs = 0
    records = []
    for n, i in enumerate(idxs):
        entry = specset[i]
        spec = entry["spec"]
        label = entry["topology"]
        spec_norm = torch.tensor(env._normalize(spec), dtype=torch.float, device=device).unsqueeze(0)

        # ValueNet ranking (no SPICE)
        vscores = {}
        for t in TOPOS:
            with torch.no_grad():
                vscores[t] = value_net(spec_norm, z_cache[t]).item()
        ranked = sorted(TOPOS, key=lambda t: vscores[t], reverse=True)

        # SPICE ground truth: best reward per topology
        rewards = {t: best_spice_reward(actor, z_cache[t], spec_norm, t, spec, env, args.k)
                   for t in TOPOS}
        emp_best = max(TOPOS, key=lambda t: rewards[t])
        emp_best_r = rewards[emp_best]

        rec = {"spec_id": entry["id"], "label": label, "vn_top1": ranked[0],
               "vn_top2": ranked[:2], "emp_best": emp_best,
               "emp_best_reward": emp_best_r, "feasible": emp_best_r >= FEASIBLE_REWARD}
        records.append(rec)

        if emp_best_r >= FEASIBLE_REWARD:
            feasible_specs += 1
            if ranked[0] == emp_best: top1 += 1
            if emp_best in ranked[:2]: top2 += 1
            if ranked[0] == label: label_match += 1

        if (n + 1) % 20 == 0:
            d = max(1, feasible_specs)
            print(f"  [{n+1}/{len(idxs)}] feasible={feasible_specs}  "
                  f"top1={top1/d*100:.1f}%  top2={top2/d*100:.1f}%  "
                  f"label_match={label_match/d*100:.1f}%", flush=True)

    d = max(1, feasible_specs)
    summary = {
        "run": args.run, "seed": args.seed, "n_specs": len(idxs), "k": args.k,
        "feasible_specs": feasible_specs,
        "top1_accuracy": top1 / d,
        "top2_accuracy": top2 / d,
        "valuenet_vs_heuristic_label": label_match / d,
        "random_baseline_top1": 1.0 / len(TOPOS),
    }
    print("\n=== TOPOLOGY SELECTION ACCURACY ===")
    for k_, v_ in summary.items():
        print(f"  {k_}: {v_}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump({"summary": summary, "records": records}, open(args.out, "w"), indent=2)
    print(f"\nSaved -> {args.out}")


if __name__ == "__main__":
    main()
