#!/usr/bin/env python3
"""Light archive-vs-DE noise floor over a subsample of the 40-topology set."""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
from scipy.optimize import differential_evolution

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from env.graph_utils import TOPOLOGY_PARAMS
from env.reward import WEIGHTS_AREA, compute_sim_reward
from specset.schema import load_specset
from topology.load_pool import register_all, all_topology_names
from tools.compute_envelope import (
    _electrical_metrics, r_star_for_spec, FC_REFS,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", default=os.path.join(
        REPO_ROOT, "results", "open_topo", "envelope_archive_40.json"))
    ap.add_argument("--specset", default=os.path.join(
        REPO_ROOT, "specset", "specset_eval.json"))
    ap.add_argument("--n-specs", type=int, default=8)
    ap.add_argument("--n-topos", type=int, default=8)
    ap.add_argument("--de-maxiter", type=int, default=15)
    ap.add_argument("--out", default=os.path.join(
        REPO_ROOT, "results", "open_topo", "envelope_validate_40.json"))
    args = ap.parse_args()

    register_all()
    names = all_topology_names()
    rng = np.random.default_rng(0)
    # Mix originals + composed
    orig = [n for n in names if not n.startswith("Gen_")]
    gen = [n for n in names if n.startswith("Gen_")]
    pick = list(orig) + list(rng.choice(gen, size=min(args.n_topos - len(orig), len(gen)),
                                        replace=False))
    pick = pick[:args.n_topos]

    with open(args.archive) as fh:
        archives = json.load(fh)["archives"]
    specs = load_specset(args.specset)
    idx = rng.choice(len(specs), size=min(args.n_specs, len(specs)), replace=False)

    gaps = []
    rows = []
    for j, i in enumerate(idx, 1):
        spec = specs[int(i)]["spec"]
        fc, tech = float(spec["fc_ghz"]), int(spec["tech"])
        bound = r_star_for_spec(spec, archives, pick)
        for sm in ("ideal",):
            for topo in pick:
                d = len(TOPOLOGY_PARAMS[topo])
                arch_r = bound[sm].get(topo)
                if arch_r is None:
                    continue

                def neg_r(a, topo=topo, sm=sm):
                    m = _electrical_metrics(
                        topo, np.asarray(a, float), fc, tech, sm)
                    if m is None:
                        return 5.0
                    m["area_mm2"] = m.get("area_ref_mm2")
                    return -compute_sim_reward(m, spec, weights=WEIGHTS_AREA)

                try:
                    res = differential_evolution(
                        neg_r, bounds=[(0.0, 1.0)] * d, maxiter=args.de_maxiter,
                        popsize=8, tol=0.02, seed=0, polish=False, init="sobol",
                    )
                    de_r = -float(res.fun)
                except Exception as exc:
                    rows.append({"topo": topo, "error": repr(exc)})
                    continue
                gap = float(de_r - arch_r)
                gaps.append(gap)
                rows.append({
                    "topo": topo, "sm": sm, "spec_i": int(i),
                    "archive_r": arch_r, "de_r": de_r, "gap": gap,
                })
                print(f"  [{len(rows)}] {topo} gap={gap:+.4f}", flush=True)

    arr = np.asarray(gaps) if gaps else np.zeros(1)
    summary = {
        "n": len(gaps),
        "mean_gap": float(arr.mean()),
        "median_gap": float(np.median(arr)),
        "p95_gap": float(np.percentile(arr, 95)),
        "max_gap": float(arr.max()),
        "frac_within_0.05": float(np.mean(np.abs(arr) < 0.05)),
        "topologies": pick,
        "rows": rows,
    }
    with open(args.out, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps({k: summary[k] for k in
                      ("n", "mean_gap", "median_gap", "p95_gap", "max_gap")},
                     indent=2))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
