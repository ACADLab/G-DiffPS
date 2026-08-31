"""Does the g_il discontinuity break the Pareto-archive factorization?

`r_star = max over the Pareto front` is only provably `max over achievable` if
reward is monotone in every archived metric. It is not: `env/reward.py` scores

    il_db =  2 dB  ->  g_il = 1 - 2/t
    il_db =  0 dB  ->  g_il = 1.0
    il_db = -1 dB  ->  g_il = 0.0     (§9.2 anti-reward-hacking branch)

so reward jumps down as IL improves through zero. A point with *worse* IL can
therefore score higher, and if it is Pareto-dominated it never enters the
archive. Only Vector_Modulator produces negative IL, so that is where this
would bite.

This re-runs each cell's Sobol+DE sweep with the archive's seed, keeps the
samples the Pareto filter discarded, and asks directly: for real specs in that
cell, does any dominated sample outscore the front maximum?

  clean  -> the assumption is empirically safe; record it and move on.
  dirty  -> redefine the archive's IL axis as max(il_db, 0), which is monotone
            under the reward, and rebuild.

Run: .venv/bin/python tools/monotonicity_audit.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from env.graph_utils import TOPOLOGY_PARAMS
from env.reward import WEIGHTS_AREA, compute_sim_reward
from specset.schema import TRAIN_SPECSET_PATH, load_specset
from tools.compute_envelope import (
    C_OFF_CLASSES, FC_REFS, SWITCH_MODELS, build_cell_archive, c_off_class,
    fc_bin,
)


def _score(rows: list[dict], spec: dict) -> tuple[float, int]:
    best, arg = -1e18, -1
    for i, r in enumerate(rows):
        m = dict(r["metrics"])
        if m.get("area_mm2") is None:
            m["area_mm2"] = m.get("area_ref_mm2")
        v = float(compute_sim_reward(m, spec, weights=WEIGHTS_AREA))
        if v > best:
            best, arg = v, i
    return best, arg


def audit_cell(job: tuple) -> dict:
    topo, sm, bi, c, specs = job
    built = build_cell_archive(topo, sm, bi, c, return_all_rows=True)
    front, allr = built["front"], built["all_rows"]
    front_ids = {id(r) for r in front}
    dominated = [r for r in allr if id(r) not in front_ids]

    worst_gap, n_viol = 0.0, 0
    example = None
    for spec in specs:
        fbest, _ = _score(front, spec)
        dbest, darg = _score(dominated, spec) if dominated else (-1e18, -1)
        gap = dbest - fbest
        if gap > 1e-9:
            n_viol += 1
            if gap > worst_gap:
                worst_gap = gap
                example = {
                    "spec_id": spec.get("_id"),
                    "front_max": fbest,
                    "dominated_max": dbest,
                    "gap": gap,
                    "dominated_il_db": dominated[darg]["metrics"].get("il_db"),
                }
    return {
        "cell": f"{topo}|{sm}|{bi}|{c or 'na'}",
        "topology": topo,
        "n_front": len(front),
        "n_dominated": len(dominated),
        "n_specs": len(specs),
        "n_violations": n_viol,
        "worst_gap": worst_gap,
        "example": example,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--specset", default=TRAIN_SPECSET_PATH)
    ap.add_argument("--specs-per-cell", type=int, default=25)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--out", default=os.path.join(
        REPO_ROOT, "results", "joint", "monotonicity_audit.json"))
    args = ap.parse_args()

    specs = load_specset(args.specset)
    rng = np.random.default_rng(0)

    # Bucket specs by the cell they would be scored against.
    buckets: dict[tuple, list] = {}
    for e in specs:
        s = dict(e["spec"])
        s["_id"] = e["id"]
        bi = fc_bin(float(s["fc_ghz"]))
        for sm in SWITCH_MODELS:
            c = None if sm == "ideal" else c_off_class(int(s["tech"]))
            buckets.setdefault((bi, sm, c), []).append(s)

    jobs = []
    for topo in sorted(TOPOLOGY_PARAMS):
        for bi in range(len(FC_REFS)):
            for sm in SWITCH_MODELS:
                cs = [None] if sm == "ideal" else list(C_OFF_CLASSES)
                for c in cs:
                    pool = buckets.get((bi, sm, c), [])
                    if not pool:
                        continue
                    idx = rng.choice(
                        len(pool), size=min(args.specs_per_cell, len(pool)),
                        replace=False)
                    jobs.append((topo, sm, bi, c, [pool[i] for i in idx]))

    print(f"auditing {len(jobs)} cells x <= {args.specs_per_cell} specs")
    results = []
    try:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            for i, r in enumerate(ex.map(audit_cell, jobs), 1):
                results.append(r)
                if r["n_violations"]:
                    print(f"  [{i}/{len(jobs)}] {r['cell']}  "
                          f"VIOLATIONS {r['n_violations']}/{r['n_specs']}  "
                          f"worst gap {r['worst_gap']:.4f}")
    except (PermissionError, OSError):
        print("  (parallel unavailable, running serial)")
        results = [audit_cell(j) for j in jobs]

    tot_v = sum(r["n_violations"] for r in results)
    tot_s = sum(r["n_specs"] for r in results)
    worst = max(results, key=lambda r: r["worst_gap"])
    by_topo: dict[str, dict] = {}
    for r in results:
        b = by_topo.setdefault(r["topology"], {"violations": 0, "specs": 0,
                                               "worst_gap": 0.0})
        b["violations"] += r["n_violations"]
        b["specs"] += r["n_specs"]
        b["worst_gap"] = max(b["worst_gap"], r["worst_gap"])

    print(f"\n{'topology':>18s} {'violations':>12s} {'rate':>8s} {'worst gap':>10s}")
    for t in sorted(by_topo):
        b = by_topo[t]
        print(f"{t:>18s} {b['violations']:5d}/{b['specs']:<6d} "
              f"{b['violations']/max(b['specs'],1):7.2%} {b['worst_gap']:10.4f}")

    clean = tot_v == 0
    print(f"\ntotal: {tot_v}/{tot_s} ({tot_v/max(tot_s,1):.3%}) "
          f"worst gap {worst['worst_gap']:.4f}")
    if clean:
        print("VERDICT: CLEAN — no dominated sample outscores its front max. "
              "The factorization is empirically safe despite the g_il jump.")
    else:
        print("VERDICT: DIRTY — the g_il discontinuity does let dominated "
              "points win. Redefine the archive IL axis as max(il_db, 0) and "
              "rebuild.")
        if worst["example"]:
            print(f"  worst: {worst['cell']}  {worst['example']}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump({"clean": clean, "total_violations": tot_v,
                   "total_checks": tot_s, "by_topology": by_topo,
                   "worst": worst, "cells": results}, fh, indent=2)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
