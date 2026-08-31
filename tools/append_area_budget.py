#!/usr/bin/env python3
"""Append (or redraw) max_area_mm2 + area_rank on the existing spec JSON.

Preserves the original RF fields bit-identically; only writes the area budget
field and the entry-level area_rank metadata. Ranks are drawn U{1..N} over the
current topology set, so rerun with --force whenever that set changes.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from sim.area_model import reference_areas_mm2
from specset.schema import TRAIN_SPECSET_PATH, atomic_write_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--specset",
        default=TRAIN_SPECSET_PATH,
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--force", action="store_true",
                    help="redraw even if max_area_mm2 already present")
    args = ap.parse_args()

    with open(args.specset) as f:
        payload = json.load(f)

    wrapped = isinstance(payload, dict) and "specs" in payload
    specs = payload["specs"] if wrapped else payload
    if not isinstance(specs, list):
        raise SystemExit(f"unexpected payload type in {args.specset}: {type(payload)}")

    rng = np.random.default_rng(args.seed)

    n_topo = None
    for entry in specs:
        spec = entry["spec"]
        if not args.force and "max_area_mm2" in spec and "area_rank" in entry:
            continue
        areas = reference_areas_mm2(float(spec["fc_ghz"]), tech=int(spec.get("tech", 0)))
        ordered = sorted(areas.values())
        n_topo = len(ordered)
        r = int(rng.integers(1, n_topo + 1))
        eps = float(rng.uniform(0.02, 0.10))
        spec["max_area_mm2"] = float(ordered[r - 1] * (1.0 + eps))
        entry["area_rank"] = r

    if wrapped:
        payload["specs"] = specs
        atomic_write_json(args.specset, payload)
    else:
        atomic_write_json(args.specset, specs)

    ranks = [e.get("area_rank") for e in specs if e.get("area_rank")]
    print(f"wrote {len(specs)} entries → {args.specset}")
    print(f"n_topo={n_topo}  area_rank range={min(ranks)}..{max(ranks)}")


if __name__ == "__main__":
    main()
