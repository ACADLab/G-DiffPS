"""
Baseline: Simulated Annealing (SA) via scipy.optimize.dual_annealing.
Dual annealing combines classical SA with fast simulated annealing (FSA)
and local search. Competitive global optimizer for noisy landscapes.

Budget: 2000 SPICE evaluations (same as random search, much less than DE's 20k).
Useful for showing sample efficiency comparison.

Usage:
    python3 baselines/simulated_annealing.py --topo Loaded_Line --fc 28.0 --seed 42
    python3 baselines/simulated_annealing.py --all-topos --seed 42
"""

import argparse
import json
import os
import sys
import time
import numpy as np
from scipy.optimize import dual_annealing

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from env.phaseshifter_env import PhaseShifterEnv
from train_diffusion import action_to_params, make_spice_netlist, parallel_eval_worker
from sim.physics_priors import check_physics_priors

TOPOLOGY_NAMES = [
    "Loaded_Line", "Switched_Line", "Reflection_Type",
    "Switched_Filter", "Vector_Modulator", "All_Pass"
]

DEFAULT_SPECS = {
    "Loaded_Line":      {"fc_ghz": 28.0, "bw_pct": 20.0, "phase_coverage_deg": 180.0, "phase_bits": 5, "rms_phase_err_deg": 3.0, "rms_gain_err_db": 1.0, "max_il_db": 2.0, "min_rl_db": 18.0, "vdd": 1.8, "pmax_mw": 15.0, "tech": 0, "app": 2},
    "Switched_Line":    {"fc_ghz": 24.0, "bw_pct": 20.0, "phase_coverage_deg": 180.0, "phase_bits": 4, "rms_phase_err_deg": 4.5, "rms_gain_err_db": 1.0, "max_il_db": 2.0, "min_rl_db": 12.0, "vdd": 1.8, "pmax_mw": 20.0, "tech": 0, "app": 2},
    "Reflection_Type":  {"fc_ghz": 10.0, "bw_pct": 25.0, "phase_coverage_deg": 180.0, "phase_bits": 4, "rms_phase_err_deg": 5.0, "rms_gain_err_db": 1.5, "max_il_db": 2.0, "min_rl_db": 15.0, "vdd": 2.5, "pmax_mw": 25.0, "tech": 1, "app": 3},
    "Switched_Filter":  {"fc_ghz": 18.0, "bw_pct": 20.0, "phase_coverage_deg": 90.0,  "phase_bits": 4, "rms_phase_err_deg": 6.0, "rms_gain_err_db": 2.0, "max_il_db": 2.5, "min_rl_db": 12.0, "vdd": 1.8, "pmax_mw": 20.0, "tech": 0, "app": 3},
    "Vector_Modulator": {"fc_ghz": 8.0,  "bw_pct": 30.0, "phase_coverage_deg": 360.0, "phase_bits": 5, "rms_phase_err_deg": 4.0, "rms_gain_err_db": 1.0, "max_il_db": 1.8, "min_rl_db": 18.0, "vdd": 3.3, "pmax_mw": 35.0, "tech": 2, "app": 3},
    "All_Pass":         {"fc_ghz": 2.4,  "bw_pct": 10.0, "phase_coverage_deg": 180.0, "phase_bits": 3, "rms_phase_err_deg": 8.0, "rms_gain_err_db": 2.0, "max_il_db": 1.5, "min_rl_db": 18.0, "vdd": 3.3, "pmax_mw": 40.0, "tech": 2, "app": 0},
}


def run_simulated_annealing(topo: str, spec: dict, maxfun: int, seed: int, out_dir: str):
    env = PhaseShifterEnv()
    t0 = time.time()
    spice_calls = [0]
    best_reward = [-999.0]
    best_metrics = [None]
    trace = []

    print(f"[SA] {topo} | fc={spec['fc_ghz']} GHz | maxfun={maxfun} | seed={seed}")

    def objective(x):
        action = np.clip(x, 0.0, 1.0).astype(np.float32)
        params = action_to_params(action, topo, spec)
        fc = spec["fc_ghz"]

        passed_prior = check_physics_priors(topo, params, fc)
        if not passed_prior:
            trace.append({"spice_calls": spice_calls[0], "reward": -5.0, "passed_prior": False})
            return 5.0

        expert_bonus = env.compute_expert_bonus(topo, spec)
        netlist_path = make_spice_netlist(topo, params)
        reward, agg_metrics, _ = parallel_eval_worker(
            (netlist_path, spec, topo, expert_bonus, None)
        )
        spice_calls[0] += 1

        if reward > best_reward[0]:
            best_reward[0] = reward
            best_metrics[0] = agg_metrics
            elapsed = time.time() - t0
            print(f"  [SA] New best: reward={reward:.3f}  spice={spice_calls[0]}  {elapsed:.1f}s")

        trace.append({"spice_calls": spice_calls[0], "reward": float(reward), "passed_prior": True})
        return -float(reward)

    bounds = [(0.0, 1.0)] * 9
    result = dual_annealing(
        objective,
        bounds,
        maxfun=maxfun,
        seed=seed,
        initial_temp=5230,   # default; controls early exploration width
        restart_temp_ratio=2e-5,
        visit=2.62,          # visiting distribution q_v parameter
        accept=-5.0,         # acceptance parameter
        no_local_search=False,  # enable local L-BFGS-B polishing
    )

    elapsed = time.time() - t0
    summary = {
        "method": "simulated_annealing_dual",
        "topology": topo,
        "fc_ghz": spec["fc_ghz"],
        "seed": seed,
        "maxfun": maxfun,
        "spice_calls": spice_calls[0],
        "best_reward": float(best_reward[0]),
        "best_metrics": best_metrics[0],
        "sa_success": bool(result.success),
        "sa_message": result.message,
        "elapsed_s": elapsed,
    }

    print(f"[SA] Done: best={best_reward[0]:.3f}  spice={spice_calls[0]}  {elapsed:.1f}s")

    os.makedirs(out_dir, exist_ok=True)
    tag = f"{topo}_fc{spec['fc_ghz']}_seed{seed}"
    with open(os.path.join(out_dir, f"sa_{tag}.json"), "w") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(out_dir, f"sa_{tag}_trace.jsonl"), "w") as f:
        for r in trace:
            f.write(json.dumps(r) + "\n")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topo", default=None, choices=TOPOLOGY_NAMES)
    ap.add_argument("--all-topos", action="store_true")
    ap.add_argument("--fc", type=float, default=None)
    ap.add_argument("--maxfun", type=int, default=2000,
                    help="Max SPICE evaluations for dual annealing")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default="results/baselines/simulated_annealing")
    args = ap.parse_args()

    topos = TOPOLOGY_NAMES if args.all_topos else [args.topo]
    if not topos[0]:
        ap.error("Specify --topo or --all-topos")

    all_results = []
    for t in topos:
        spec = dict(DEFAULT_SPECS[t])
        if args.fc is not None:
            spec["fc_ghz"] = args.fc
        r = run_simulated_annealing(t, spec, args.maxfun, args.seed, args.out_dir)
        all_results.append(r)

    print("\n=== SIMULATED ANNEALING SUMMARY ===")
    print(f"{'Topology':<22}  {'Best Reward':>11}  {'SPICE Calls':>11}  {'Converged':>9}")
    for r in all_results:
        print(f"{r['topology']:<22}  {r['best_reward']:>11.3f}  {r['spice_calls']:>11d}  "
              f"{str(r['sa_success']):>9}")


if __name__ == "__main__":
    main()
