#!/usr/bin/env python3
"""T1.5 acceptance: envelope spread with area reward, stratified by area_rank.

Relative to D2 baseline (RF-only envelope). Acceptance: for area_rank <= 3 on
sub-10 GHz specs, per-spec cross-topo spread of r* exceeds 0.3.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from env.graph_utils import TOPOLOGY_PARAMS
from env.reward import WEIGHTS_AREA, compute_sim_reward
from sim.mna_scorer import mna_evaluate
from sim.physics_priors import check_physics_priors
from train_diffusion import action_to_params


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--specset",
        default=os.path.join(REPO_ROOT, "specset", "specset_train.json"),
    )
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--bounds", default="electrical")
    ap.add_argument("--switch-model", default="ideal")
    ap.add_argument(
        "--d2-baseline",
        default=os.path.join(REPO_ROOT, "results", "joint", "envelope_spread_mna.json"),
    )
    ap.add_argument(
        "--out",
        default=os.path.join(REPO_ROOT, "results", "joint", "t15_accept.json"),
    )
    args = ap.parse_args()

    from specset.schema import load_specset
    specs = load_specset(args.specset)

    # Focus: sub-10 GHz with area_rank <= 3 (and report all strata).
    targets = [
        e for e in specs
        if float(e["spec"].get("fc_ghz", 99)) < 10.0
        and int(e.get("area_rank", 99)) <= 3
    ]
    # Cap for wall-clock; prefer all matching.
    rng = np.random.default_rng(args.seed)
    if len(targets) > 80:
        idx = rng.choice(len(targets), size=80, replace=False)
        targets = [targets[i] for i in idx]

    topos = list(TOPOLOGY_PARAMS.keys())
    print(f"T1.5 accept: {len(targets)} sub-10 GHz specs with area_rank<=3, K={args.k}")

    spreads = []
    per_spec = []
    mean_rstar = defaultdict(list)

    for si, entry in enumerate(targets):
        spec = entry["spec"]
        sid = entry.get("id", f"idx_{si}")
        rstar = {}
        for topo in topos:
            d = len(TOPOLOGY_PARAMS[topo])
            best = -5.0
            for _ in range(args.k):
                a = rng.random(d)
                params = action_to_params(
                    a, topo, spec, bounds=args.bounds, switch_model=args.switch_model,
                )
                if not check_physics_priors(
                    topo, params, float(spec["fc_ghz"]),
                    pmax_mw=float(spec.get("pmax_mw", 1e9)),
                ):
                    continue
                _, agg = mna_evaluate(topo, params, spec)
                if agg is None:
                    continue
                r = float(compute_sim_reward(agg, spec, weights=WEIGHTS_AREA))
                best = max(best, r)
            rstar[topo] = float(best)
            mean_rstar[topo].append(best)
        sp = float(max(rstar.values()) - min(rstar.values()))
        spreads.append(sp)
        per_spec.append({
            "id": sid,
            "fc_ghz": float(spec["fc_ghz"]),
            "area_rank": int(entry.get("area_rank", -1)),
            "max_area_mm2": float(spec["max_area_mm2"]),
            "spread": sp,
            "rstar": rstar,
        })
        if (si + 1) % 10 == 0 or si == 0:
            print(f"  [{si+1}/{len(targets)}] median spread={np.median(spreads):.4f}")

    spreads_a = np.asarray(spreads, dtype=float)
    d2 = None
    if os.path.exists(args.d2_baseline):
        with open(args.d2_baseline) as f:
            d2 = json.load(f)

    out = {
        "n_specs": len(targets),
        "k": args.k,
        "switch_model": args.switch_model,
        "spread": {
            "median": float(np.median(spreads_a)),
            "mean": float(np.mean(spreads_a)),
            "frac_above_0_3": float(np.mean(spreads_a > 0.3)),
        },
        "mean_rstar": {t: float(np.mean(v)) for t, v in mean_rstar.items()},
        "d2_baseline_spread_after": None if d2 is None else d2.get("spread_after"),
        "accept": bool(np.mean(spreads_a > 0.3) >= 0.5),  # at least half
        "accept_strict_all": bool(np.all(spreads_a > 0.3)),
        "per_spec": per_spec,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)

    print("\n=== T1.5 acceptance ===")
    print(f"spread: {out['spread']}")
    print(f"D2 baseline: {out['d2_baseline_spread_after']}")
    print(f"ACCEPT (frac>0.3 >= 0.5): {out['accept']}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
