"""
Baseline: Bayesian Optimization with Gaussian Process (GP-UCB / Expected Improvement).
Uses scikit-optimize (skopt) gp_minimize over [0,1]^9 action space.

Budget: 200 calls (warm-start 20 random + 180 BO) per topology/seed.
Represents the practical surrogate-model optimization alternative.

Usage:
    python3 baselines/bayesian_opt.py --topo Loaded_Line --fc 28.0 --seed 42
    python3 baselines/bayesian_opt.py --all-topos --seed 42

Requires: scikit-optimize (pip install scikit-optimize)
"""

import argparse
import json
import os
import sys
import time
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

try:
    from skopt import gp_minimize
    from skopt.space import Real
    HAS_SKOPT = True
except ImportError:
    HAS_SKOPT = False
    print("[BO] WARNING: scikit-optimize not installed. Falling back to random search.")

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


def run_bayesian_opt(topo: str, spec: dict, n_calls: int, n_initial: int, seed: int, out_dir: str):
    env = PhaseShifterEnv()
    t0 = time.time()
    spice_calls = [0]
    best_reward = [-999.0]
    best_metrics = [None]
    trace = []

    print(f"[BO] {topo} | fc={spec['fc_ghz']} GHz | n_calls={n_calls} | n_initial={n_initial} | seed={seed}")

    def objective(x):
        action = np.clip(np.array(x, dtype=np.float32), 0.0, 1.0)
        params = action_to_params(action, topo, spec)
        fc = spec["fc_ghz"]

        passed_prior = check_physics_priors(topo, params, fc)
        if not passed_prior:
            trace.append({"spice_calls": spice_calls[0], "reward": -5.0, "passed_prior": False})
            return 5.0  # GP-BO minimizes; return high loss for rejected samples

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
            print(f"  [BO] New best: reward={reward:.3f}  spice={spice_calls[0]}  {elapsed:.1f}s")

        trace.append({"spice_calls": spice_calls[0], "reward": float(reward), "passed_prior": True})
        return -float(reward)

    if HAS_SKOPT:
        space = [Real(0.0, 1.0, name=f"a{i}") for i in range(9)]
        result = gp_minimize(
            objective,
            space,
            n_calls=n_calls,
            n_initial_points=n_initial,
            acq_func="EI",          # Expected Improvement
            random_state=seed,
            noise=1e-5,
            verbose=False,
        )
        best_x = result.x
        gp_success = True
    else:
        # Fallback: pure random search if skopt not available
        rng = np.random.default_rng(seed)
        for _ in range(n_calls):
            x = rng.uniform(0.0, 1.0, size=9).tolist()
            objective(x)
        best_x = None
        gp_success = False

    elapsed = time.time() - t0
    summary = {
        "method": "bayesian_opt_gp_ei" if HAS_SKOPT else "random_fallback",
        "topology": topo,
        "fc_ghz": spec["fc_ghz"],
        "seed": seed,
        "n_calls": n_calls,
        "n_initial": n_initial,
        "spice_calls": spice_calls[0],
        "best_reward": float(best_reward[0]),
        "best_metrics": best_metrics[0],
        "gp_success": gp_success,
        "elapsed_s": elapsed,
    }

    print(f"[BO] Done: best={best_reward[0]:.3f}  spice={spice_calls[0]}  {elapsed:.1f}s")

    os.makedirs(out_dir, exist_ok=True)
    tag = f"{topo}_fc{spec['fc_ghz']}_seed{seed}"
    with open(os.path.join(out_dir, f"bo_{tag}.json"), "w") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(out_dir, f"bo_{tag}_trace.jsonl"), "w") as f:
        for r in trace:
            f.write(json.dumps(r) + "\n")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topo", default=None, choices=TOPOLOGY_NAMES)
    ap.add_argument("--all-topos", action="store_true")
    ap.add_argument("--fc", type=float, default=None)
    ap.add_argument("--n-calls", type=int, default=200,
                    help="Total BO evaluations (20 random warm-start + 180 GP-guided)")
    ap.add_argument("--n-initial", type=int, default=20,
                    help="Random warm-start evaluations before GP kicks in")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default="results/baselines/bayesian_opt")
    args = ap.parse_args()

    topos = TOPOLOGY_NAMES if args.all_topos else [args.topo]
    if not topos[0]:
        ap.error("Specify --topo or --all-topos")

    all_results = []
    for t in topos:
        spec = dict(DEFAULT_SPECS[t])
        if args.fc is not None:
            spec["fc_ghz"] = args.fc
        r = run_bayesian_opt(t, spec, args.n_calls, args.n_initial, args.seed, args.out_dir)
        all_results.append(r)

    print("\n=== BAYESIAN OPT SUMMARY ===")
    print(f"{'Topology':<22}  {'Best Reward':>11}  {'SPICE Calls':>11}  {'Method':>20}")
    for r in all_results:
        print(f"{r['topology']:<22}  {r['best_reward']:>11.3f}  {r['spice_calls']:>11d}  "
              f"{r['method']:>20}")


if __name__ == "__main__":
    main()
