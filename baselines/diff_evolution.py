"""
Baseline: Differential Evolution (DE) via scipy.optimize.differential_evolution.
Industry-standard global optimizer. Operates over log-warped [0,1]^9 action space
with the same physics prior and SPICE reward as G-DiffPS.

Population size 50, maxiter 400 → up to 20,000 SPICE calls (same budget as G-DiffPS 20k steps).

Usage:
    python3 baselines/diff_evolution.py --topo Loaded_Line --fc 28.0 --seed 42
    python3 baselines/diff_evolution.py --all-topos --seed 42
"""

import argparse
import json
import os
import sys
import time
import numpy as np
from scipy.optimize import differential_evolution

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


def run_de(topo: str, spec: dict, popsize: int, maxiter: int, seed: int, out_dir: str):
    env = PhaseShifterEnv()
    t0 = time.time()
    spice_calls = [0]
    best_reward = [-999.0]
    best_metrics = [None]
    trace = []

    print(f"[DE] {topo} | fc={spec['fc_ghz']} GHz | pop={popsize} | maxiter={maxiter} | seed={seed}")
    print(f"     Max SPICE budget: ~{popsize * maxiter}")

    def objective(x):
        action = np.clip(x, 0.0, 1.0).astype(np.float32)
        params = action_to_params(action, topo, spec)
        fc = spec["fc_ghz"]

        passed_prior = check_physics_priors(topo, params, fc)
        if not passed_prior:
            return 5.0  # high loss = reject

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
            print(f"  [DE] New best: reward={reward:.3f}  spice={spice_calls[0]}  {elapsed:.1f}s")

        trace.append({"spice_calls": spice_calls[0], "reward": float(reward)})
        return -float(reward)  # DE minimizes

    bounds = [(0.0, 1.0)] * 9
    result = differential_evolution(
        objective,
        bounds,
        popsize=popsize,
        maxiter=maxiter,
        seed=seed,
        tol=1e-6,
        workers=1,  # serial to avoid SPICE file collisions
        polish=True,  # final Nelder-Mead refinement step
    )

    elapsed = time.time() - t0
    summary = {
        "method": "differential_evolution",
        "topology": topo,
        "fc_ghz": spec["fc_ghz"],
        "seed": seed,
        "popsize": popsize,
        "maxiter": maxiter,
        "spice_calls": spice_calls[0],
        "best_reward": float(best_reward[0]),
        "best_metrics": best_metrics[0],
        "de_success": bool(result.success),
        "de_message": result.message,
        "elapsed_s": elapsed,
    }

    print(f"[DE] Done: best={best_reward[0]:.3f}  spice={spice_calls[0]}  {elapsed:.1f}s  "
          f"converged={result.success}")

    os.makedirs(out_dir, exist_ok=True)
    tag = f"{topo}_fc{spec['fc_ghz']}_seed{seed}"
    with open(os.path.join(out_dir, f"de_{tag}.json"), "w") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(out_dir, f"de_{tag}_trace.jsonl"), "w") as f:
        for r in trace:
            f.write(json.dumps(r) + "\n")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topo", default=None, choices=TOPOLOGY_NAMES)
    ap.add_argument("--all-topos", action="store_true")
    ap.add_argument("--fc", type=float, default=None)
    ap.add_argument("--popsize", type=int, default=50,
                    help="DE population size (50 × 9-D = 450 initial evals)")
    ap.add_argument("--maxiter", type=int, default=400,
                    help="Max DE generations (50×400=20k budget matches G-DiffPS)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default="results/baselines/diff_evolution")
    args = ap.parse_args()

    topos = TOPOLOGY_NAMES if args.all_topos else [args.topo]
    if not topos[0]:
        ap.error("Specify --topo or --all-topos")

    all_results = []
    for t in topos:
        spec = dict(DEFAULT_SPECS[t])
        if args.fc is not None:
            spec["fc_ghz"] = args.fc
        r = run_de(t, spec, args.popsize, args.maxiter, args.seed, args.out_dir)
        all_results.append(r)

    print("\n=== DIFFERENTIAL EVOLUTION SUMMARY ===")
    print(f"{'Topology':<22}  {'Best Reward':>11}  {'SPICE Calls':>11}  {'Converged':>9}")
    for r in all_results:
        print(f"{r['topology']:<22}  {r['best_reward']:>11.3f}  {r['spice_calls']:>11d}  "
              f"{str(r['de_success']):>9}")


if __name__ == "__main__":
    main()
