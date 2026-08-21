"""CFM inference eval on the 10 paper scenarios -> best S-parameter metrics.
Mirrors baselines/eval_sac.py but uses the G-DiffPS CFM actor (J12 checkpoint).
Fills the S-param columns of Table 2 (convergence) and Table 3 (ablation).

Usage:
  python cfm_scenario_eval.py --run runs_diffusion/run_20260530_031117 \
      --n-samples 16 --seed 42 --out results/cfm_eval/cfm_scenarios_seed42.json
"""
import argparse, json, os, sys
import numpy as np, torch

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from env.phaseshifter_env import PhaseShifterEnv
from env.graph_utils import get_topology_graph
from models.gnn_encoder import TopologyEncoder
from models.diffusion_policy import FlowMatchingPolicy, ValueNet
from sim.physics_priors import check_physics_priors
from train_diffusion import action_to_params, make_spice_netlist, parallel_eval_worker
from baselines.eval_sac import SCENARIOS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--n-samples", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="results/cfm_eval/cfm_scenarios.json")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    gnn = TopologyEncoder(in_channels=5, hidden_channels=64, out_channels=64).to(device)
    actor = FlowMatchingPolicy(action_dim=9, spec_dim=12, graph_dim=64).to(device)
    gnn.load_state_dict(torch.load(os.path.join(args.run, "gnn_encoder.pt"), map_location=device))
    actor.load_state_dict(torch.load(os.path.join(args.run, "actor.pt"), map_location=device))
    gnn.eval(); actor.eval()
    env = PhaseShifterEnv()

    results = []
    for sc in SCENARIOS:
        topo, spec = sc["topo"], sc["spec"]
        g = get_topology_graph(topo)
        with torch.no_grad():
            z = gnn(g.x.to(device), g.edge_index.to(device))
        spec_norm = torch.tensor(env._normalize(spec), dtype=torch.float, device=device).unsqueeze(0)

        best_r, best_m = -999.0, None
        for _ in range(args.n_samples):
            with torch.no_grad():
                a = actor.sample(spec_norm, z).squeeze(0).cpu().numpy()
            params = action_to_params(a, topo, spec)
            if not check_physics_priors(topo, params, spec["fc_ghz"]):
                continue
            eb = env.compute_expert_bonus(topo, spec)
            nl = make_spice_netlist(topo, params)
            r, m, _ = parallel_eval_worker((nl, spec, topo, eb, None))
            if r > best_r:
                best_r, best_m = float(r), m
        rec = {"scenario": sc["name"], "topology": topo, "fc_ghz": spec["fc_ghz"],
               "spice_calls": 0, "best_reward": best_r, "best_metrics": best_m}
        results.append(rec)
        pm = best_m or {}
        print(f"  {sc['name']:<11} {topo:<17} r={best_r:>6.3f}  "
              f"phase_err={pm.get('rms_phase_err_deg', float('nan')):.2f}  "
              f"IL={pm.get('il_db', float('nan')):.2f}  RL={pm.get('rl_db', float('nan')):.2f}", flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(results, open(args.out, "w"), indent=2)
    print(f"\nSaved -> {args.out}")


if __name__ == "__main__":
    main()
