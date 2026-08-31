#!/usr/bin/env python3
"""D1 follow-up: is Switched_Filter's g_phase lead earned, or a sign artifact?

The MNA phase error is wrap_180(delta - sid * ideal_step), and every entry in
_IDEAL_STEP is negative. If the solver's phase convention yields a positive
delta, the error is computed against the wrong half-plane:

  * ideal = -180  ->  +180 and -180 wrap to the SAME point; a sign flip is
                      invisible. Switched_Filter is immune.
  * ideal =  -90  ->  a +90 circuit scores 180 deg of error instead of ~0.
  * ideal = -22.5 ->  a +22.5 circuit scores 45 deg of error instead of ~0.

So this probe reports, per topology, the signed delta and the error computed
both as-shipped and with the ideal grid's sign flipped. If flipping collapses
the error everywhere EXCEPT Switched_Filter, the lead is an artifact.
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

from env.graph_utils import TOPOLOGY_PARAMS
from env.netlist_graph import nominal_params
from sim.mna_scorer import (
    _IDEAL_STEP,
    _states_for_topo,
    _wrap_180,
    solve_sparams,
    sparams_to_metrics,
)
from train_diffusion import action_to_params


def deltas_for(topo, params, fc):
    """Signed per-state phase deltas vs state 0, plus their state ids."""
    states = _states_for_topo(topo)
    phases, sids = [], []
    for s in states:
        s11, s21 = solve_sparams(topo, params, fc, state=s)
        m = sparams_to_metrics(s11, s21)
        phases.append(m["phase_deg"])
        sids.append(s)
    ref = phases[0]
    return [(_wrap_180(p - ref), sid) for p, sid in zip(phases, sids)]


def rms(errs):
    return float(np.sqrt(np.mean(np.square(errs)))) if errs else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=64, help="random sizings per topo")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--bounds", default="electrical")
    ap.add_argument("--switch-model", default="ideal")
    ap.add_argument("--fcs", default="2.4,10,28")
    ap.add_argument(
        "--out",
        default=os.path.join(REPO_ROOT, "results", "joint", "phase_metric_probe.json"),
    )
    args = ap.parse_args()

    fcs = [float(x) for x in args.fcs.split(",")]
    rng = np.random.default_rng(args.seed)
    topos = list(TOPOLOGY_PARAMS.keys())
    out = {"switch_model": args.switch_model, "k": args.k, "fcs": fcs, "topologies": {}}

    for topo in topos:
        ideal_step = _IDEAL_STEP[topo]
        d = len(TOPOLOGY_PARAMS[topo])
        rec = {
            "ideal_step_deg": ideal_step,
            "delta_signed": [],
            "rms_asis": [],
            "rms_flip": [],
            "nominal": {},
        }
        for fc in fcs:
            spec = {"fc_ghz": fc, "tech": 0}
            # nominal (mid-action) reference point
            try:
                nom = deltas_for(topo, nominal_params(topo, spec, bounds=args.bounds,
                                                      switch_model=args.switch_model), fc)
                rec["nominal"][str(fc)] = {
                    "delta_deg": [round(dd, 3) for dd, _ in nom],
                    "rms_asis": round(rms([_wrap_180(dd - sid * ideal_step) for dd, sid in nom]), 3),
                    "rms_flip": round(rms([_wrap_180(dd + sid * ideal_step) for dd, sid in nom]), 3),
                }
            except Exception as e:
                rec["nominal"][str(fc)] = {"error": str(e)}

            for _ in range(args.k):
                a = rng.random(d)
                params = action_to_params(
                    a, topo, spec, bounds=args.bounds, switch_model=args.switch_model,
                )
                try:
                    ds = deltas_for(topo, params, fc)
                except Exception:
                    continue
                rec["delta_signed"].extend([dd for dd, sid in ds if sid != 0])
                rec["rms_asis"].append(rms([_wrap_180(dd - sid * ideal_step) for dd, sid in ds]))
                rec["rms_flip"].append(rms([_wrap_180(dd + sid * ideal_step) for dd, sid in ds]))

        ds = np.asarray(rec["delta_signed"], dtype=float)
        rec["summary"] = {
            "delta_median": float(np.median(ds)) if ds.size else None,
            "delta_frac_positive": float(np.mean(ds > 0)) if ds.size else None,
            "delta_std": float(np.std(ds)) if ds.size else None,
            "rms_asis_median": float(np.median(rec["rms_asis"])) if rec["rms_asis"] else None,
            "rms_flip_median": float(np.median(rec["rms_flip"])) if rec["rms_flip"] else None,
        }
        # keep the file small
        rec["delta_signed"] = [round(x, 3) for x in rec["delta_signed"][:200]]
        rec["rms_asis"] = []
        rec["rms_flip"] = []
        out["topologies"][topo] = rec

        s = rec["summary"]
        print(
            f"{topo:20s} ideal={ideal_step:7.1f}  "
            f"delta_med={s['delta_median']:8.2f} (+{100*s['delta_frac_positive']:.0f}% pos, "
            f"sd={s['delta_std']:6.2f})  rms_asis={s['rms_asis_median']:7.2f}  "
            f"rms_flip={s['rms_flip_median']:7.2f}"
        )

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
