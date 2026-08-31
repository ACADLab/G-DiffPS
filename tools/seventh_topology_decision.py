"""Decision record for the proposed seventh topology (Switched_Line_SeriesShunt).

The series-shunt variant was proposed as a remedy for T0.1c, which reported that
realistic switches collapse Switched_Line's differential phase (65.7 deg -> 3.4
deg). This script measures whether that collapse still exists.

Four measurements:

  A. reward 2x2  -- SL vs SL+shunt, ideal vs realistic, at mid-action.
  B. dphi table  -- the quantity T0.1c actually claimed collapsed.
  C. arm-ratio sweep -- the mechanism test. Switched_Line's dphi comes from the
     path-length difference between its two arms. Before T0.4, both L_short_mm
     and L_long_mm fell through to the generic `_mm` rule and shared one window,
     so at mid-action the arms were equal, dphi was near zero, and off-arm
     leakage dominated whatever was left. T0.4 restored the partition
     (short 0.3-0.8 lam/4, long 0.8-2.5 lam/4). This sweep re-creates the old
     geometry by forcing the arm ratio and shows where leakage sensitivity lives.
  D. All_Pass midpoint -- All_Pass reads dphi = 0 under *both* switch models, so
     whatever ails it is not an isolation problem. Checked against the box.

Run: .venv/bin/python tools/seventh_topology_decision.py
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

from env.graph_utils import TOPOLOGY_PARAMS
from sim.mna_scorer import mna_evaluate, parse_spice_number, solve_sparams
from tools.shunt_isolation_probe import SHUNT_CONFIGS, _with_shunt_switches
from train_diffusion import action_to_params

FC = 28.0
SPEC = {
    "fc_ghz": FC,
    "tech": 0,
    "phase_bits": 1,
    "phase_coverage_deg": 180.0,
    "rms_phase_err_deg": 5.0,
    "max_il_db": 3.0,
    "min_rl_db": 15.0,
    "max_gain_err_db": 1.0,
    "max_area_mm2": 5.0,
}


def _mid_params(topo: str, switch_model: str) -> dict:
    keys = TOPOLOGY_PARAMS[topo]
    return action_to_params(
        np.full(len(keys), 0.5), topo, {"fc_ghz": FC, "tech": 0},
        bounds="electrical", switch_model=switch_model,
    )


def _dphi(topo: str, params: dict) -> float:
    _, s21_0 = solve_sparams(topo, params, FC, state=0)
    _, s21_1 = solve_sparams(topo, params, FC, state=1)
    ph0 = math.degrees(math.atan2(s21_0.imag, s21_0.real))
    ph1 = math.degrees(math.atan2(s21_1.imag, s21_1.real))
    return abs(((ph1 - ph0 + 180.0) % 360.0) - 180.0)


def part_a_reward_matrix() -> dict:
    """The 2x2 the seventh topology was to be judged on."""
    cells = {}
    for model in ("ideal", "realistic"):
        params = _mid_params("Switched_Line", model)
        score, agg = mna_evaluate("Switched_Line", params, SPEC)
        cells[f"Switched_Line/{model}"] = {
            "score": score,
            "rms_phase_err_deg": agg["rms_phase_err_deg"],
            "il_db": agg["il_db"],
            "rl_db": agg["rl_db"],
            "R_off_ohm": parse_spice_number(params["R_off"]),
        }
        with _with_shunt_switches(
            "Switched_Line", SHUNT_CONFIGS["Switched_Line"]["off_arm_mids"]
        ):
            score_s, agg_s = mna_evaluate("Switched_Line", params, SPEC)
        cells[f"Switched_Line_SeriesShunt/{model}"] = {
            "score": score_s,
            "rms_phase_err_deg": agg_s["rms_phase_err_deg"],
            "il_db": agg_s["il_db"],
            "rl_db": agg_s["rl_db"],
            "R_off_ohm": parse_spice_number(params["R_off"]),
        }
    sl_i = cells["Switched_Line/ideal"]["score"]
    ss_i = cells["Switched_Line_SeriesShunt/ideal"]["score"]
    sl_r = cells["Switched_Line/realistic"]["score"]
    ss_r = cells["Switched_Line_SeriesShunt/realistic"]["score"]
    return {
        "cells": cells,
        "crossover": bool(sl_i > ss_i and ss_r > sl_r),
        "crossover_definition": "SL beats shunt under ideal AND shunt beats SL under realistic",
    }


def part_b_dphi_table() -> dict:
    out = {}
    for topo in ("Switched_Line", "Switched_Filter", "All_Pass"):
        row = {}
        for model in ("ideal", "realistic"):
            row[model] = _dphi(topo, _mid_params(topo, model))
        row["retention"] = (
            row["realistic"] / row["ideal"] if row["ideal"] > 1e-9 else float("nan")
        )
        out[topo] = row
    return out


def part_c_arm_ratio_sweep() -> dict:
    """dphi vs arm ratio. rho=1 is the pre-T0.4 mid-action geometry."""
    rows = []
    for rho in (1.0, 1.02, 1.05, 1.1, 1.25, 1.5, 2.0, 3.0):
        row = {"arm_ratio": rho}
        for model in ("ideal", "realistic"):
            p = dict(_mid_params("Switched_Line", model))
            l_short = parse_spice_number(p["L_short_mm"])
            p["L_short_mm"] = f"{l_short:.6f}"
            p["L_long_mm"] = f"{l_short * rho:.6f}"
            row[f"dphi_{model}"] = _dphi("Switched_Line", p)
        row["retention"] = (
            row["dphi_realistic"] / row["dphi_ideal"]
            if row["dphi_ideal"] > 1e-9 else float("nan")
        )
        rows.append(row)
    nominal = {}
    for model in ("ideal", "realistic"):
        p = _mid_params("Switched_Line", model)
        nominal[f"dphi_{model}"] = _dphi("Switched_Line", p)
        nominal["arm_ratio_at_mid_action"] = (
            parse_spice_number(p["L_long_mm"]) / parse_spice_number(p["L_short_mm"])
        )
    nominal["retention"] = nominal["dphi_realistic"] / nominal["dphi_ideal"]
    return {"sweep": rows, "t04_nominal": nominal}


def part_d_allpass(k: int = 512) -> dict:
    keys = TOPOLOGY_PARAMS["All_Pass"]
    rng = np.random.default_rng(0)
    out = {}
    for model in ("ideal", "realistic"):
        mid = _mid_params("All_Pass", model)
        mid_dphi = _dphi("All_Pass", mid)
        best, best_a = -1.0, None
        vals = []
        for _ in range(k):
            a = rng.random(len(keys))
            p = action_to_params(
                a, "All_Pass", {"fc_ghz": FC, "tech": 0},
                bounds="electrical", switch_model=model,
            )
            try:
                d = _dphi("All_Pass", p)
            except Exception:
                continue
            vals.append(d)
            if d > best:
                best, best_a = d, a.tolist()
        arr = np.array(vals)
        # Is the midpoint symmetric? A/B halves identical -> dphi identically 0.
        sym = all(
            abs(parse_spice_number(mid[a]) - parse_spice_number(mid[b])) < 1e-12
            for a, b in (("L_apA_nh", "L_apB_nh"),
                         ("C_brA_pf", "C_brB_pf"),
                         ("C_cA_pf", "C_cB_pf"))
        )
        out[model] = {
            "midpoint_dphi_deg": mid_dphi,
            "midpoint_branches_identical": sym,
            "box_best_dphi_deg": best,
            "box_best_action": best_a,
            "box_median_dphi_deg": float(np.median(arr)),
            "box_frac_over_45deg": float(np.mean(arr > 45.0)),
            "k": len(vals),
        }
    return out


def part_e_collapse_boundary() -> dict:
    """How bad must the off-state be before dphi actually collapses?

    Sweeps C_off past every value in TECH_SWITCH so the claim "realistic is not
    broken" carries a stated margin instead of resting on one operating point.
    """
    from sim.switch_model import TECH_SWITCH, r_off_eff

    c_offs_ff = [20, 25, 50, 100, 200, 400, 650, 1000, 2000, 5000]
    rows = []
    for fc in (2.4, 10.0, 28.0, 40.0):
        base = _mid_params("Switched_Line", "ideal")
        keys = TOPOLOGY_PARAMS["Switched_Line"]
        base = action_to_params(
            np.full(len(keys), 0.5), "Switched_Line", {"fc_ghz": fc, "tech": 0},
            bounds="electrical", switch_model="ideal",
        )

        def dphi_at(fc_local, params):
            _, a = solve_sparams("Switched_Line", params, fc_local, state=0)
            _, b = solve_sparams("Switched_Line", params, fc_local, state=1)
            p0 = math.degrees(math.atan2(a.imag, a.real))
            p1 = math.degrees(math.atan2(b.imag, b.real))
            return abs(((p1 - p0 + 180.0) % 360.0) - 180.0)

        d_ideal = dphi_at(fc, base)
        for c_ff in c_offs_ff:
            p = dict(base)
            r_off = r_off_eff(c_ff * 1e-15, fc)
            p["R_off"] = f"{r_off:.4e}"
            d = dphi_at(fc, p)
            rows.append({
                "fc_ghz": fc,
                "c_off_ff": c_ff,
                "r_off_ohm": r_off,
                "dphi_ideal": d_ideal,
                "dphi_deg": d,
                "retention": d / d_ideal if d_ideal > 1e-9 else float("nan"),
            })
    # Smallest C_off at which retention drops below 0.5, per fc.
    boundary = {}
    for fc in (2.4, 10.0, 28.0, 40.0):
        bad = [r for r in rows if r["fc_ghz"] == fc and r["retention"] < 0.5]
        boundary[str(fc)] = min((r["c_off_ff"] for r in bad), default=None)
    return {
        "sweep": rows,
        "collapse_c_off_ff_by_fc": boundary,
        "tech_c_off_ff": {
            str(k): v[2] * 1e15 for k, v in TECH_SWITCH.items()
        },
    }


def main() -> int:
    res = {"fc_ghz": FC, "bounds": "electrical", "spec": SPEC}

    print("=== A. reward 2x2 at mid-action (fc=28 GHz) ===")
    res["A_reward_matrix"] = part_a_reward_matrix()
    for k, v in res["A_reward_matrix"]["cells"].items():
        print(f"  {k:42s} score={v['score']:+.4f}  rms={v['rms_phase_err_deg']:6.2f}deg"
              f"  il={v['il_db']:5.2f}dB  rl={v['rl_db']:6.2f}dB")
    print(f"  crossover: {res['A_reward_matrix']['crossover']}")

    print("\n=== B. dphi at mid-action: ideal vs realistic ===")
    res["B_dphi_table"] = part_b_dphi_table()
    for topo, row in res["B_dphi_table"].items():
        print(f"  {topo:18s} ideal={row['ideal']:7.2f}deg  "
              f"realistic={row['realistic']:7.2f}deg  retention={row['retention']:.3f}")

    print("\n=== C. mechanism: dphi vs Switched_Line arm ratio ===")
    res["C_arm_ratio"] = part_c_arm_ratio_sweep()
    print(f"  {'rho':>6s}  {'dphi_ideal':>11s}  {'dphi_real':>10s}  {'retention':>9s}")
    for r in res["C_arm_ratio"]["sweep"]:
        print(f"  {r['arm_ratio']:6.2f}  {r['dphi_ideal']:11.3f}  "
              f"{r['dphi_realistic']:10.3f}  {r['retention']:9.3f}")
    nom = res["C_arm_ratio"]["t04_nominal"]
    print(f"  T0.4 mid-action rho={nom['arm_ratio_at_mid_action']:.3f}: "
          f"ideal={nom['dphi_ideal']:.2f}deg realistic={nom['dphi_realistic']:.2f}deg "
          f"retention={nom['retention']:.3f}")

    print("\n=== D. All_Pass: is dphi=0 a switch problem? ===")
    res["D_allpass"] = part_d_allpass()
    for model, v in res["D_allpass"].items():
        print(f"  {model:10s} midpoint={v['midpoint_dphi_deg']:.3f}deg "
              f"(branches_identical={v['midpoint_branches_identical']}) "
              f"box_best={v['box_best_dphi_deg']:.2f}deg "
              f"median={v['box_median_dphi_deg']:.2f}deg "
              f"frac>45deg={v['box_frac_over_45deg']:.3f}")

    print("\n=== E. how bad must the off-state be to collapse dphi? ===")
    res["E_collapse_boundary"] = part_e_collapse_boundary()
    print(f"  {'fc':>6s} {'C_off(fF)':>10s} {'R_off(ohm)':>11s} {'retention':>9s}")
    for r in res["E_collapse_boundary"]["sweep"]:
        if r["c_off_ff"] in (20, 200, 650, 2000, 5000):
            print(f"  {r['fc_ghz']:6.1f} {r['c_off_ff']:10d} "
                  f"{r['r_off_ohm']:11.1f} {r['retention']:9.3f}")
    print(f"  collapse threshold (retention<0.5) by fc: "
          f"{res['E_collapse_boundary']['collapse_c_off_ff_by_fc']}")
    print(f"  C_off actually in TECH_SWITCH: "
          f"{res['E_collapse_boundary']['tech_c_off_ff']}")

    # ---- verdict ----
    b = res["B_dphi_table"]
    sl_ret = b["Switched_Line"]["retention"]
    sf_ret = b["Switched_Filter"]["retention"]
    collapse_reproduces = sl_ret < 0.5
    res["verdict"] = {
        "t01c_collapse_reproduces": bool(collapse_reproduces),
        "switched_line_retention": sl_ret,
        "switched_filter_retention": sf_ret,
        "allpass_is_switch_limited": bool(
            res["D_allpass"]["ideal"]["midpoint_dphi_deg"] > 1.0
        ),
        "decision": (
            "ADD seventh topology" if collapse_reproduces
            else "ABANDON seventh topology: no collapse to remedy under T0.4 bounds"
        ),
    }
    print("\n=== VERDICT ===")
    print(f"  T0.1c collapse reproduces: {collapse_reproduces}")
    print(f"  Switched_Line dphi retention under realistic: {sl_ret:.3f}")
    print(f"  Switched_Filter dphi retention under realistic: {sf_ret:.3f}")
    print(f"  All_Pass limited by switches: "
          f"{res['verdict']['allpass_is_switch_limited']}")
    print(f"  -> {res['verdict']['decision']}")

    out = os.path.join(REPO_ROOT, "results", "joint", "seventh_topology_decision.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as fh:
        json.dump(res, fh, indent=2)
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
