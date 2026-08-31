#!/usr/bin/env python3
"""Is All_Pass a hard topology, or is its bounds box not an all-pass network?

A bridged-T all-pass is a constant-resistance network: with series inductors
L (two halves), a bridging capacitor C_br and a shunt capacitor C_c, it is
matched at every frequency only on a locus of the form Z0 = sqrt(L/C). If the
sampling box is not centred on that locus, most of the box is a mismatched
lowpass-ish network rather than an all-pass, and no representation can predict
a phase shift that the circuit does not produce.

Reports, per fc:
  * the midpoint (a=0.5) sizing and its sqrt(L/C) ratios vs Z0=50
  * RL and delta-phi at the midpoint
  * the best |delta-phi| and best RL reachable anywhere in the box
  * where the box sits relative to the frequency it would need to be centred on
"""
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

from env.graph_utils import TOPOLOGY_PARAMS
from sim.mna_scorer import (
    solve_sparams,
    sparams_to_metrics,
    _states_for_topo,
    _wrap_180,
)
from train_diffusion import action_to_params

Z0 = 50.0


def eval_action(a, topo, spec, fc, switch_model="ideal"):
    params = action_to_params(a, topo, spec, bounds="electrical",
                              switch_model=switch_model)
    ph, rl, il = [], [], []
    for s in _states_for_topo(topo):
        s11, s21 = solve_sparams(topo, params, fc, state=s)
        m = sparams_to_metrics(s11, s21)
        ph.append(m["phase_deg"])
        rl.append(m["rl_db"])
        il.append(m["il_db"])
    return params, _wrap_180(ph[1] - ph[0]), float(np.mean(rl)), float(np.mean(il))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--fcs", default="2.4,10,28")
    ap.add_argument("--switch-model", default="ideal")
    ap.add_argument(
        "--out",
        default=os.path.join(REPO_ROOT, "results", "joint", "allpass_box_probe.json"),
    )
    args = ap.parse_args()

    topo = "All_Pass"
    keys = TOPOLOGY_PARAMS[topo]
    d = len(keys)
    rng = np.random.default_rng(args.seed)
    fcs = [float(x) for x in args.fcs.split(",")]
    out = {"z0": Z0, "keys": keys, "k": args.k, "fcs": {}}

    for fc in fcs:
        spec = {"fc_ghz": fc, "tech": 0}
        mid = np.full(d, 0.5)
        pm, dphi_m, rl_m, il_m = eval_action(mid, topo, spec, fc, args.switch_model)
        L = float(pm["L_apA_nh"]) * 1e-9
        Cbr = float(pm["C_brA_pf"]) * 1e-12
        Cc = float(pm["C_cA_pf"]) * 1e-12
        rec = {
            "midpoint_params": {k: float(pm[k]) for k in keys},
            "sqrt_L_over_Cbr": math.sqrt(L / Cbr),
            "sqrt_L_over_Cc": math.sqrt(L / Cc),
            "sqrt_2L_over_Cc": math.sqrt(2 * L / Cc),
            "midpoint_dphi_deg": dphi_m,
            "midpoint_rl_db": rl_m,
            "midpoint_il_db": il_m,
            # the frequency at which the midpoint network would be a 90 deg cell
            "midpoint_corner_ghz": 1.0 / (2 * math.pi * math.sqrt(L * Cbr)) / 1e9,
            "omega_sqrt_LC_at_fc": 2 * math.pi * fc * 1e9 * math.sqrt(L * Cbr),
        }

        best_dphi = (0.0, None)
        best_rl = (-1e9, None)
        dphis, rls = [], []
        for _ in range(args.k):
            a = rng.random(d)
            try:
                p, dphi, rl, il = eval_action(a, topo, spec, fc, args.switch_model)
            except Exception:
                continue
            dphis.append(abs(dphi))
            rls.append(rl)
            if abs(dphi) > best_dphi[0]:
                best_dphi = (abs(dphi), (a.copy(), rl, il))
            if rl > best_rl[0]:
                best_rl = (rl, (a.copy(), dphi, il))

        rec["box_max_abs_dphi_deg"] = best_dphi[0]
        rec["box_max_abs_dphi_rl_db"] = best_dphi[1][1] if best_dphi[1] else None
        rec["box_max_abs_dphi_action"] = (
            [round(float(x), 3) for x in best_dphi[1][0]] if best_dphi[1] else None
        )
        rec["box_best_rl_db"] = best_rl[0]
        rec["box_best_rl_dphi_deg"] = best_rl[1][1] if best_rl[1] else None
        rec["abs_dphi_median"] = float(np.median(dphis)) if dphis else None
        rec["abs_dphi_p90"] = float(np.percentile(dphis, 90)) if dphis else None
        rec["frac_abs_dphi_over_45"] = float(np.mean(np.asarray(dphis) > 45.0)) if dphis else None
        rec["frac_abs_dphi_over_80"] = float(np.mean(np.asarray(dphis) > 80.0)) if dphis else None
        out["fcs"][str(fc)] = rec

        print(f"--- fc = {fc} GHz")
        print(f"    midpoint  L={pm['L_apA_nh']} nH  C_br={pm['C_brA_pf']} pF  C_c={pm['C_cA_pf']} pF")
        print(f"    sqrt(L/C_br) = {rec['sqrt_L_over_Cbr']:6.2f} ohm    "
              f"sqrt(L/C_c) = {rec['sqrt_L_over_Cc']:6.2f} ohm    "
              f"sqrt(2L/C_c) = {rec['sqrt_2L_over_Cc']:6.2f} ohm")
        print(f"    midpoint: dphi={dphi_m:8.3f} deg   RL={rl_m:7.2f} dB   IL={il_m:7.3f} dB")
        print(f"    midpoint corner freq = {rec['midpoint_corner_ghz']:.1f} GHz  "
              f"(w*sqrt(LC) at fc = {rec['omega_sqrt_LC_at_fc']:.4f})")
        print(f"    box: max|dphi| = {rec['box_max_abs_dphi_deg']:7.2f} deg "
              f"(RL there {rec['box_max_abs_dphi_rl_db']:6.2f} dB), "
              f"median|dphi| = {rec['abs_dphi_median']:6.2f}, "
              f"frac>45 deg = {100*rec['frac_abs_dphi_over_45']:.1f}%, "
              f"frac>80 deg = {100*rec['frac_abs_dphi_over_80']:.1f}%")
        print()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
