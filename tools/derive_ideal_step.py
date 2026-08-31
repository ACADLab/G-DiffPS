#!/usr/bin/env python3
"""Pass-1 phase probe: derive ideal_step_deg before archive build."""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np
from scipy.stats import qmc

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from env.graph_utils import TOPOLOGY_PARAMS
from env.topology_registry import original_six
from sim import mna_scorer
from sim.mna_scorer import solve_sparams
from topology.load_pool import load_pool, register_composed
from train_diffusion import action_to_params

SNAP_GRID = (22.5, 45.0, 90.0, 180.0)
FC_PROBE = 7.96  # mid fc bin (bin 4)


def _wrap180(d: float) -> float:
    return ((d + 180.0) % 360.0) - 180.0


def _dphi(topo: str, params: dict, fc: float) -> float:
    _, s0 = solve_sparams(topo, params, fc, state=0)
    _, s1 = solve_sparams(topo, params, fc, state=1)
    p0 = math.degrees(math.atan2(s0.imag, s0.real))
    p1 = math.degrees(math.atan2(s1.imag, s1.real))
    return abs(_wrap180(p1 - p0))


def snap_step(raw_deg: float) -> float:
    return -min(SNAP_GRID, key=lambda g: abs(raw_deg - g))


def probe_topology(topo: str, n_sobol: int = 256, fc: float = FC_PROBE) -> dict:
    d = len(TOPOLOGY_PARAMS[topo])
    spec = {"fc_ghz": fc, "tech": 0}
    sob = qmc.Sobol(d, scramble=True, seed=abs(hash(topo)) % (2**31))
    pts = sob.random(n_sobol)
    dphis = []
    for a in pts:
        try:
            params = action_to_params(
                np.asarray(a, float), topo, spec,
                bounds="electrical", switch_model="ideal",
            )
            dp = _dphi(topo, params, fc)
            if math.isfinite(dp) and dp > 0.5:
                dphis.append(dp)
        except Exception:
            continue
    if not dphis:
        return {"raw_p90": 0.0, "snapped": -90.0, "n_finite": 0}
    raw = float(np.percentile(dphis, 90))
    return {"raw_p90": raw, "snapped": snap_step(raw), "n_finite": len(dphis)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default=os.path.join(REPO_ROOT, "results", "open_topo", "pool.json"))
    ap.add_argument("--out", default=os.path.join(REPO_ROOT, "results", "open_topo", "ideal_steps.json"))
    ap.add_argument("--n-sobol", type=int, default=256)
    args = ap.parse_args()

    pool = load_pool(args.pool)
    register_composed(pool)

    results = {}
    # Original six: record existing steps (frozen).
    for name in sorted(original_six()):
        step = mna_scorer._IDEAL_STEP.get(name, -90.0)
        results[name] = {"raw_p90": None, "snapped": step, "frozen": True}

    print(f"probing {len(pool['composed'])} composed topologies @ {FC_PROBE} GHz …")
    for entry in pool["composed"]:
        name = entry["name"]
        r = probe_topology(name, n_sobol=args.n_sobol)
        r["frozen"] = False
        results[name] = r
        mna_scorer._IDEAL_STEP[name] = r["snapped"]
        entry["ideal_step_deg"] = r["snapped"]
        entry["ideal_step_raw_p90"] = r["raw_p90"]
        print(f"  {name}: raw_p90={r['raw_p90']:.1f}° → snapped={r['snapped']}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump({"fc_probe_ghz": FC_PROBE, "steps": results}, fh, indent=2)

    with open(args.pool, "w") as fh:
        json.dump(pool, fh, indent=2)

    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
