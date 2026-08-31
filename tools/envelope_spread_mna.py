#!/usr/bin/env python3
"""D2: best-of-K MNA envelope spread before (90°) vs after (T1)."""
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
from tools.reward_term_decompose import _legacy_parts


def _stratified_sample(specs, n, seed):
    """Sample n specs stratified by fc_ghz band when possible."""
    rng = np.random.default_rng(seed)
    bands = defaultdict(list)
    for i, entry in enumerate(specs):
        spec = entry.get("spec", entry)
        fc = float(spec.get("fc_ghz", 28.0))
        if fc < 10:
            band = "sub10"
        elif fc < 20:
            band = "mid"
        else:
            band = "mmw"
        bands[band].append(i)
    keys = [k for k in ("sub10", "mid", "mmw") if bands[k]]
    if not keys:
        idx = rng.choice(len(specs), size=min(n, len(specs)), replace=False)
        return [specs[i] for i in idx]
    per = max(1, n // len(keys))
    chosen = []
    for k in keys:
        pool = bands[k]
        take = min(per, len(pool))
        chosen.extend(rng.choice(pool, size=take, replace=False).tolist())
    # fill remainder
    remaining = [i for i in range(len(specs)) if i not in set(chosen)]
    need = n - len(chosen)
    if need > 0 and remaining:
        chosen.extend(rng.choice(remaining, size=min(need, len(remaining)), replace=False).tolist())
    return [specs[i] for i in chosen[:n]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--specset",
        default=os.path.join(REPO_ROOT, "specset", "specset_train.json"),
    )
    ap.add_argument("--n-specs", type=int, default=100)
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--bounds", default="electrical")
    ap.add_argument("--switch-model", default="ideal")
    ap.add_argument("--skip-prior-reject", action="store_true", default=True)
    ap.add_argument(
        "--out",
        default=os.path.join(REPO_ROOT, "results", "joint", "envelope_spread_mna.json"),
    )
    args = ap.parse_args()

    from specset.schema import load_specset
    all_specs = load_specset(args.specset)
    specs = _stratified_sample(all_specs, args.n_specs, args.seed)
    topos = list(TOPOLOGY_PARAMS.keys())
    rng = np.random.default_rng(args.seed + 7)

    print(
        f"D2: {len(specs)} specs × {len(topos)} topos × K={args.k} "
        f"switch={args.switch_model}"
    )

    spreads_b, spreads_a = [], []
    mean_rstar_b = defaultdict(list)
    mean_rstar_a = defaultdict(list)
    per_spec = []

    for si, entry in enumerate(specs):
        spec = entry.get("spec", entry)
        sid = entry.get("id", f"idx_{si}")
        rstar_b, rstar_a = {}, {}
        for topo in topos:
            d = len(TOPOLOGY_PARAMS[topo])
            best_b, best_a = -5.0, -5.0
            for _ in range(args.k):
                a = rng.random(d).astype(np.float64)
                params = action_to_params(
                    a, topo, spec, bounds=args.bounds, switch_model=args.switch_model,
                )
                if args.skip_prior_reject:
                    if not check_physics_priors(topo, params, float(spec.get("fc_ghz", 28.0))):
                        continue
                _, agg = mna_evaluate(topo, params, spec)
                if agg is None:
                    continue
                rb, _ = _legacy_parts(agg, spec)
                ra = float(compute_sim_reward(agg, spec, weights=WEIGHTS_AREA))
                best_b = max(best_b, rb)
                best_a = max(best_a, ra)
            rstar_b[topo] = float(best_b)
            rstar_a[topo] = float(best_a)
            mean_rstar_b[topo].append(best_b)
            mean_rstar_a[topo].append(best_a)
        sb = float(max(rstar_b.values()) - min(rstar_b.values()))
        sa = float(max(rstar_a.values()) - min(rstar_a.values()))
        spreads_b.append(sb)
        spreads_a.append(sa)
        per_spec.append({
            "id": sid,
            "fc_ghz": float(spec.get("fc_ghz", 28.0)),
            "spread_before": sb,
            "spread_after": sa,
            "rstar_before": rstar_b,
            "rstar_after": rstar_a,
        })
        if (si + 1) % 10 == 0 or si == 0:
            print(
                f"  [{si+1}/{len(specs)}] median envelope spread after="
                f"{np.median(spreads_a):.4f}"
            )

    def _summ(spreads):
        a = np.asarray(spreads, dtype=float)
        return {
            "median": float(np.median(a)),
            "mean": float(np.mean(a)),
            "p90": float(np.percentile(a, 90)),
            "frac_above_0_3": float(np.mean(a > 0.3)),
            "frac_below_0_15": float(np.mean(a < 0.15)),
        }

    out = {
        "n_specs": len(specs),
        "k": args.k,
        "seed": args.seed,
        "switch_model": args.switch_model,
        "bounds": args.bounds,
        "topos": topos,
        "spread_before": _summ(spreads_b),
        "spread_after": _summ(spreads_a),
        "mean_rstar_before": {t: float(np.mean(v)) for t, v in mean_rstar_b.items()},
        "mean_rstar_after": {t: float(np.mean(v)) for t, v in mean_rstar_a.items()},
        "per_spec": per_spec,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)

    print("\n=== D2 envelope spread ===")
    print(f"before: {out['spread_before']}")
    print(f"after:  {out['spread_after']}")
    print(f"mean r* after: {out['mean_rstar_after']}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
