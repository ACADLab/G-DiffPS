"""Midpoint / random / untrained baselines for LOOCV transfer comparison."""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from env.graph_utils import TOPOLOGY_PARAMS
from env.phaseshifter_env import PhaseShifterEnv
from env.reward import WEIGHTS_AREA, classify_attempt, compute_sim_reward
from sim.physics_priors import check_physics_priors
from sim.mna_scorer import topology_admits_spec
from train_diffusion import action_to_params, make_spice_netlist, parallel_eval_worker


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--held-out", required=True)
    ap.add_argument("--mode", choices=["midpoint", "random"], default="midpoint")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--bounds", default="electrical")
    ap.add_argument("--switch-model", required=True, choices=["ideal", "realistic"])
    ap.add_argument("--eval-specset", default=os.path.join(REPO_ROOT, "specset", "specset_eval.json"))
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    env = PhaseShifterEnv(
        restrict_to=[args.held_out],
        eval_specset_path=args.eval_specset,
        pool="eval",
        expert_bonus_scale=0.0,
    )
    keys = TOPOLOGY_PARAMS[args.held_out]
    n_strict = n_prior = n_elig = n_sim = 0
    rewards = []
    for i in range(args.n):
        obs, _ = env.reset()
        spec = dict(env.current_spec)
        if topology_admits_spec(args.held_out, spec):
            n_elig += 1
        if args.mode == "midpoint":
            a = np.full(len(keys), 0.5, dtype=np.float32)
        else:
            a = rng.random(len(keys), dtype=np.float32)
        params = action_to_params(
            a, args.held_out, spec, bounds=args.bounds, switch_model=args.switch_model,
        )
        prior = check_physics_priors(
            args.held_out, params, spec["fc_ghz"],
            pmax_mw=float(spec.get("pmax_mw", 1e9)),
        )
        metrics = None
        phys = -5.0
        if prior:
            n_prior += 1
            nl = make_spice_netlist(args.held_out, params, spec_dict=spec, fc_mode="spec")
            r, agg, _ = parallel_eval_worker(
                (nl, spec, args.held_out, 0.0, [args.held_out], 0.0, params)
            )
            metrics = agg
            if metrics is not None:
                n_sim += 1
                phys = float(compute_sim_reward(metrics, spec, weights=WEIGHTS_AREA))
            else:
                phys = float(r)
        out = classify_attempt(
            topology=args.held_out, spec=spec, prior_pass=prior,
            metrics=metrics, physical_reward=phys,
        )
        if out["strict_compliance"]:
            n_strict += 1
        rewards.append(phys)

    summary = {
        "held_out": args.held_out,
        "mode": args.mode,
        "n": args.n,
        "bounds": args.bounds,
        "switch_model": args.switch_model,
        "eligible_rate": n_elig / args.n,
        "prior_pass_rate": n_prior / args.n,
        "sim_success_rate": n_sim / args.n,
        "strict_compliance": n_strict / args.n,
        "mean_physical_reward": float(np.mean(rewards)),
        "best_physical_reward": float(np.max(rewards)),
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
