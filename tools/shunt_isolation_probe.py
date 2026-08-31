"""T0.1c: off-isolation table + series-shunt rescue for branch-select topos.

Reproduces the series-switch isolation numbers, then measures Δφ under
ideal vs realistic R_off with and without state-dependent shunt-to-ground
switches on the off arm (textbook series-shunt SPDT).

All_Pass is a bridged-T: two shunt positions are tried before concluding
it cannot be fixed.
"""
from __future__ import annotations

import json
import math
import os
import sys
from contextlib import contextmanager

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from env.graph_utils import TOPOLOGY_PARAMS
from env.netlist_graph import TOPOLOGY_NETLIST, Dev, _normalize_name
from sim.mna_scorer import parse_spice_number, solve_sparams
from sim.switch_model import r_off_eff
from train_diffusion import action_to_params


def series_isolation_db(z_off_mag: float, z0: float = 50.0) -> float:
    """|S21| for a series R between two Z0 ports."""
    s21 = (2.0 * z0) / (2.0 * z0 + z_off_mag)
    return 20.0 * math.log10(max(s21, 1e-30))


def isolation_table() -> list[dict]:
    rows = []
    for label, r_or_c, kind in (
        ("R_off=10k", 1e4, "R"),
        ("C_off=20fF", 20e-15, "C"),
        ("C_off=650fF", 650e-15, "C"),
    ):
        for fc in (2.4, 28.0):
            z = r_or_c if kind == "R" else r_off_eff(r_or_c, fc)
            rows.append({
                "model": label,
                "fc_ghz": fc,
                "z_off_ohm": z,
                "isolation_db": series_isolation_db(z),
            })
    return rows


def _mid_params(topo: str, fc: float, switch_model: str, tech: int = 0) -> dict:
    keys = TOPOLOGY_PARAMS[topo]
    action = np.full(len(keys), 0.5)
    return action_to_params(
        action, topo, {"fc_ghz": fc, "tech": tech},
        bounds="electrical", switch_model=switch_model,
    )


def _dphi(topo: str, params: dict, fc: float) -> float:
    try:
        _, s21_0 = solve_sparams(topo, params, fc, state=0)
        _, s21_1 = solve_sparams(topo, params, fc, state=1)
        ph0 = math.degrees(math.atan2(s21_0.imag, s21_0.real))
        ph1 = math.degrees(math.atan2(s21_1.imag, s21_1.real))
        return abs(((ph1 - ph0 + 180.0) % 360.0) - 180.0)
    except Exception as e:
        print(f"  [warn] _dphi failed: {e}")
        return float("nan")


@contextmanager
def _with_shunt_switches(topo: str, devices: dict[str, Dev]):
    """Temporarily add state-dependent shunt R_switch devices; restore after."""
    name = _normalize_name(topo)
    netlist = TOPOLOGY_NETLIST[name]
    added = list(devices.keys())
    for dname, dev in devices.items():
        netlist[dname] = dev
    try:
        yield
    finally:
        for dname in added:
            netlist.pop(dname, None)


# Series-shunt: shunt closed on the OFF arm (complementary to series path select).
SHUNT_CONFIGS = {
    "Switched_Line": {
        # State 0 selects short → shunt long arm; state 1 selects long → shunt short.
        "off_arm_mids": {
            "Sh_long_a": Dev("R_switch", ("a_long", "0"),
                             switch_param="R_sh_long", on_in_states=(0,)),
            "Sh_long_b": Dev("R_switch", ("b_long", "0"),
                             switch_param="R_sh_long", on_in_states=(0,)),
            "Sh_short_a": Dev("R_switch", ("a_short", "0"),
                              switch_param="R_sh_short", on_in_states=(1,)),
            "Sh_short_b": Dev("R_switch", ("b_short", "0"),
                              switch_param="R_sh_short", on_in_states=(1,)),
        },
    },
    "Switched_Filter": {
        "off_arm_mids": {
            "Sh_lpf_a": Dev("R_switch", ("a_lpf", "0"),
                            switch_param="R_sh_lpf", on_in_states=(0,)),
            "Sh_lpf_b": Dev("R_switch", ("b_lpf", "0"),
                            switch_param="R_sh_lpf", on_in_states=(0,)),
            "Sh_hpf_a": Dev("R_switch", ("a_hpf", "0"),
                            switch_param="R_sh_hpf", on_in_states=(1,)),
            "Sh_hpf_b": Dev("R_switch", ("b_hpf", "0"),
                            switch_param="R_sh_hpf", on_in_states=(1,)),
        },
    },
    "All_Pass": {
        # Position A: outside the LC section (branch I/O).
        "branch_io": {
            "Sh_B_in": Dev("R_switch", ("b_in", "0"),
                           switch_param="R_sh_B", on_in_states=(0,)),
            "Sh_B_out": Dev("R_switch", ("b_out", "0"),
                            switch_param="R_sh_B", on_in_states=(0,)),
            "Sh_A_in": Dev("R_switch", ("a_in", "0"),
                           switch_param="R_sh_A", on_in_states=(1,)),
            "Sh_A_out": Dev("R_switch", ("a_out", "0"),
                            switch_param="R_sh_A", on_in_states=(1,)),
        },
        # Position B: mid-node of series L's (inside bridged-T resonance).
        "mid_nodes": {
            "Sh_mB": Dev("R_switch", ("m_B", "0"),
                         switch_param="R_sh_mB", on_in_states=(0,)),
            "Sh_mA": Dev("R_switch", ("m_A", "0"),
                         switch_param="R_sh_mA", on_in_states=(1,)),
        },
    },
}


def probe_topo(topo: str, fc: float = 28.0) -> dict:
    out = {"topology": topo, "fc_ghz": fc, "variants": {}}
    for model in ("ideal", "realistic"):
        params = _mid_params(topo, fc, model)
        out["variants"][f"{model}_no_shunt"] = {
            "R_off": float(parse_spice_number(params["R_off"])),
            "dphi_deg": _dphi(topo, params, fc),
        }

    params = _mid_params(topo, fc, "realistic")
    for cname, devices in SHUNT_CONFIGS[topo].items():
        with _with_shunt_switches(topo, devices):
            out["variants"][f"realistic_shunt_{cname}"] = {
                "devices": list(devices.keys()),
                "dphi_deg": _dphi(topo, params, fc),
            }
    return out


def main():
    iso = isolation_table()
    print("=== Series-switch isolation (analytic) ===")
    for r in iso:
        print(f"  {r['model']:14s} @ {r['fc_ghz']:5.1f} GHz  "
              f"Z_off={r['z_off_ohm']:8.1f} Ω  isol={r['isolation_db']:6.1f} dB")

    results = {"isolation": iso, "probes": []}
    for topo in ("Switched_Line", "Switched_Filter", "All_Pass"):
        print(f"\n=== {topo} Δφ @ 28 GHz ===")
        p = probe_topo(topo, 28.0)
        results["probes"].append(p)
        for k, v in p["variants"].items():
            print(f"  {k:40s}  Δφ={v['dphi_deg']:7.2f}°")

    out_path = os.path.join(REPO_ROOT, "results", "harness", "shunt_isolation_probe.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nWrote {out_path}")

    sl = results["probes"][0]
    base = sl["variants"]["realistic_no_shunt"]["dphi_deg"]
    rescued = sl["variants"]["realistic_shunt_off_arm_mids"]["dphi_deg"]
    ideal_d = sl["variants"]["ideal_no_shunt"]["dphi_deg"]
    print(f"\nDecision gate (Switched_Line): "
          f"ideal={ideal_d:.1f}° realistic={base:.1f}° shunt={rescued:.1f}°")

    # A rescue is only meaningful if the baseline collapsed in the first place.
    # Omitting this precondition made the gate fire on a healthy baseline.
    if base > 0.5 * ideal_d:
        print(f"VERDICT: no collapse to rescue (realistic retains "
              f"{base / ideal_d:.1%} of ideal Δφ). Shunt is unnecessary; adding it "
              f"costs insertion loss for isolation the circuit does not need. "
              f"See results/joint/seventh_topology_decision.json.")
        return 0
    if rescued > 0.5 * ideal_d and rescued > 30.0:
        print("VERDICT: baseline collapsed and shunt restores Δφ → add seventh topology.")
        return 0
    print("VERDICT: baseline collapsed and shunt did not restore enough Δφ; "
          "investigate before adding topo.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
