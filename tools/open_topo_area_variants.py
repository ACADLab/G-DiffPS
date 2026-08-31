#!/usr/bin/env python3
"""Produce frozen vs 40-re-anchored max_area_mm2 variants for open-topo eval.

Headline uses frozen v5 budgets (calibrated on six). Secondary redraws rank
anchors over the full 40-topology set. The two sit on different specs and are
not directly comparable.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from sim.area_model import reference_areas_mm2
from topology.load_pool import register_all


def reanchor_entry(entry: dict, rng: np.random.Generator) -> dict:
    e = copy.deepcopy(entry)
    spec = e["spec"]
    fc = float(spec["fc_ghz"])
    tech = int(spec["tech"])
    areas = reference_areas_mm2(fc, tech=tech)
    ordered = sorted(areas.values())
    n = len(ordered)
    r = int(rng.integers(1, n + 1))
    eps = float(rng.uniform(0.02, 0.10))
    e["spec"]["max_area_mm2"] = float(ordered[r - 1] * (1.0 + eps))
    e["area_rank"] = r
    e["area_rank_n"] = n
    e["area_budget_source"] = "reanchored_40"
    return e


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--r-star", default=os.path.join(
        REPO_ROOT, "results", "open_topo", "r_star_40.json"))
    ap.add_argument("--out-dir", default=os.path.join(
        REPO_ROOT, "results", "open_topo"))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    register_all()
    with open(args.r_star) as fh:
        doc = json.load(fh)

    frozen_path = os.path.join(args.out_dir, "r_star_40_frozen_area.json")
    with open(frozen_path, "w") as fh:
        out = copy.deepcopy(doc)
        out["area_budget"] = "frozen_v5_six"
        for e in out["specs"]:
            e["area_budget_source"] = "frozen_v5_six"
        json.dump(out, fh)
    print(f"wrote {frozen_path} (headline)")

    rng = np.random.default_rng(args.seed)
    reanch = copy.deepcopy(doc)
    reanch["area_budget"] = "reanchored_40"
    reanch["specs"] = [reanchor_entry(e, rng) for e in reanch["specs"]]
    # Note: r_star rewards were computed under frozen budgets; re-anchoring
    # changes the area term. Re-score would require re-running assign — for the
    # secondary report we only record the new budgets and warn.
    reanch["note"] = (
        "Area budgets redrawn over 40 topologies; r_star values still reflect "
        "frozen-budget rewards. Treat as secondary / not directly comparable."
    )
    re_path = os.path.join(args.out_dir, "r_star_40_reanchored_area.json")
    with open(re_path, "w") as fh:
        json.dump(reanch, fh)
    print(f"wrote {re_path} (secondary)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
