#!/usr/bin/env python3
"""Distribution check over the 40-topology archive before LOOCV."""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from topology.load_pool import all_topology_names, load_pool, register_all


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", default=os.path.join(
        REPO_ROOT, "results", "open_topo", "envelope_archive_40.json"))
    ap.add_argument("--r-star", default=os.path.join(
        REPO_ROOT, "results", "open_topo", "r_star_40.json"))
    ap.add_argument("--out", default=os.path.join(
        REPO_ROOT, "results", "open_topo", "distribution_check.json"))
    ap.add_argument("--min-dphi", type=float, default=10.0)
    args = ap.parse_args()

    register_all()
    names = all_topology_names()
    with open(args.archive) as fh:
        archives = json.load(fh)["archives"]

    per_topo = {}
    for name in names:
        # Use ideal mid-bin cell
        key = f"{name}|ideal|4|na"
        cell = archives.get(key)
        if not cell or "front" not in cell:
            per_topo[name] = {"functional": False, "reason": "missing_cell"}
            continue
        pes = [pt["m"].get("rms_phase_err_deg") for pt in cell["front"]
               if pt["m"].get("rms_phase_err_deg") is not None]
        ils = [pt["m"].get("il_db") for pt in cell["front"]
               if pt["m"].get("il_db") is not None]
        rls = [pt["m"].get("rl_db") for pt in cell["front"]
               if pt["m"].get("rl_db") is not None]
        best_pe = float(min(pes)) if pes else 99.0
        best_il = float(min(ils)) if ils else 99.0
        best_rl = float(max(rls)) if rls else 0.0
        # Rough delta-phi proxy: if rms_phase_err stays huge, no useful phase shift
        functional = best_pe < 45.0 and best_il < 6.0
        per_topo[name] = {
            "functional": functional,
            "best_rms_phase_err_deg": best_pe,
            "best_il_db": best_il,
            "best_rl_db": best_rl,
            "n_front": cell.get("n_front", 0),
        }

    argmax_counts = {n: 0 for n in names}
    if os.path.exists(args.r_star):
        with open(args.r_star) as fh:
            rdoc = json.load(fh)
        for entry in rdoc.get("specs", []):
            rs = entry.get("r_star", {}).get("ideal", {})
            arg = rs.get("argmax")
            if arg in argmax_counts:
                argmax_counts[arg] += 1

    functional = [n for n, v in per_topo.items() if v.get("functional")]
    out = {
        "n_topologies": len(names),
        "n_functional": len(functional),
        "functional": functional,
        "per_topology": per_topo,
        "argmax_counts_ideal": argmax_counts,
        "note": ("A set where most topologies are non-functional makes held-out "
                 "trivially easy; report LOOCV over all 40 and over functional."),
    }
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"functional: {len(functional)}/{len(names)}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
