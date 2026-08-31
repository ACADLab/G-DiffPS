"""Choose the All_Pass section split so that a = 0.5 is a real design.

All_Pass builds differential phase from two bridged-T sections that are switched
between. Under bounds='electrical' both sections fall through to the same generic
`_nh` / `_pf` rules, so at the box midpoint they are identically sized, the two
switch states are the same circuit, and Delta-phi is 0 by symmetry. This is the
same defect T0.4 fixed for Switched_Line's arms.

The fix mirrors T0.4: give the sections offset windows. Parametrized by a single
midpoint ratio r,

    A: log window centred at L0 / sqrt(r)   (and C0 / sqrt(r))
    B: log window centred at L0 * sqrt(r)   (and C0 * sqrt(r))

Scaling L and C together in each section shifts that section's resonance while
leaving sqrt(L/C) -- its characteristic impedance -- at 50 ohm, so the split buys
differential phase without spending return loss. Window *width* is unchanged, so
symmetric designs stay reachable; only the midpoint moves.

This sweeps r and reports Delta-phi / IL / RL at the midpoint so the choice is
measured rather than assumed.

Run: .venv/bin/python tools/allpass_recentre_probe.py
"""
from __future__ import annotations

import json
import math
import os
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from sim.mna_scorer import solve_sparams, sparams_to_metrics

Z0 = 50.0
FCS = (2.4, 10.0, 28.0, 40.0)
RATIOS = (1.0, 1.25, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 12.0)

# All_Pass's ideal differential step (sim.mna_scorer._IDEAL_STEP). The midpoint
# should be a good instance of what the topology is for, not merely non-zero.
IDEAL_STEP_DEG = 90.0
MIN_RL_DB = 18.0
MAX_IL_DB = 1.0


def _resonant(fc_ghz: float) -> tuple[float, float]:
    """(C0 pF, L0 nH) at fc for a 50 ohm reference."""
    omega = 2.0 * math.pi * fc_ghz * 1e9
    c0_pf = (1.0 / (omega * Z0)) * 1e12
    l0_nh = (Z0 / omega) * 1e9
    return c0_pf, l0_nh


def midpoint_params(fc_ghz: float, r: float, coupling_k: float = 2.6) -> dict:
    """Params at a = 0.5 for section-split ratio r."""
    c0, l0 = _resonant(fc_ghz)
    s = math.sqrt(r)
    l_a, l_b = l0 / s, l0 * s
    c_a, c_b = c0 / s, c0 * s
    return {
        "L_apA_nh": f"{l_a:.6f}", "C_brA_pf": f"{c_a:.6f}",
        "C_cA_pf": f"{coupling_k * c_a:.6f}",
        "L_apB_nh": f"{l_b:.6f}", "C_brB_pf": f"{c_b:.6f}",
        "C_cB_pf": f"{coupling_k * c_b:.6f}",
        "R_on": "3.0000e+00", "R_off": "1.0000e+04",
    }


def measure(fc_ghz: float, r: float) -> dict:
    p = midpoint_params(fc_ghz, r)
    out = {"fc_ghz": fc_ghz, "ratio": r}
    phases, ils, rls = [], [], []
    for state in (0, 1):
        s11, s21 = solve_sparams("All_Pass", p, fc_ghz, state=state)
        m = sparams_to_metrics(s11, s21)
        phases.append(m["phase_deg"])
        ils.append(m["il_db"])
        rls.append(abs(m["rl_db"]))
    out["dphi_deg"] = abs(((phases[1] - phases[0] + 180.0) % 360.0) - 180.0)
    out["il_db"] = float(np.mean(ils))
    out["rl_db"] = float(np.mean(rls))
    out["gain_err_db"] = float(abs(ils[1] - ils[0]))
    return out


def main() -> int:
    rows = []
    print(f"{'ratio':>7s}  " + "  ".join(f"{fc:>6.1f}GHz" for fc in FCS)
          + "     (Delta-phi deg)")
    for r in RATIOS:
        cells = [measure(fc, r) for fc in FCS]
        rows.extend(cells)
        print(f"{r:7.2f}  " + "  ".join(f"{c['dphi_deg']:9.2f}" for c in cells))

    print(f"\n{'ratio':>7s}  {'IL dB':>8s}  {'RL dB':>8s}  {'gain err dB':>12s}"
          f"   (at 28 GHz)")
    for r in RATIOS:
        c = measure(28.0, r)
        print(f"{r:7.2f}  {c['il_db']:8.3f}  {c['rl_db']:8.2f}  "
              f"{c['gain_err_db']:12.3f}")

    # Pick the ratio whose midpoint sits closest to the ideal 90 deg step while
    # staying well matched and low loss. Delta-phi is non-monotonic in r (it
    # peaks near r=3 and falls again), so 90 deg is reachable twice; the
    # high-r branch is the matched one.
    feasible = []
    for r in RATIOS:
        cells = [measure(fc, r) for fc in FCS]
        if (all(c["rl_db"] >= MIN_RL_DB for c in cells)
                and all(c["il_db"] <= MAX_IL_DB for c in cells)):
            err = max(abs(c["dphi_deg"] - IDEAL_STEP_DEG) for c in cells)
            feasible.append((err, r, cells))
    feasible.sort()
    best = feasible[0][1] if feasible else None

    print(f"\nfeasible ratios (RL>={MIN_RL_DB} dB, IL<={MAX_IL_DB} dB), "
          f"ranked by |Delta-phi - {IDEAL_STEP_DEG:.0f} deg|:")
    for err, r, cells in feasible[:5]:
        print(f"  r={r:5.2f}  dphi={cells[0]['dphi_deg']:7.2f} deg  "
              f"err={err:6.2f}  RL={cells[0]['rl_db']:5.2f} dB  "
              f"IL={cells[0]['il_db']:5.3f} dB")
    print(f"\nrecommended section split ratio: {best}")
    out = os.path.join(REPO_ROOT, "results", "joint", "allpass_recentre.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as fh:
        json.dump({"rows": rows, "recommended_ratio": best,
                   "criteria": {"ideal_step_deg": IDEAL_STEP_DEG,
                                "min_rl_db": MIN_RL_DB,
                                "max_il_db": MAX_IL_DB}}, fh, indent=2)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
