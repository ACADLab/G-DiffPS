"""Electrical action bounds: role-aware C windows; template defaults in-range."""
from __future__ import annotations

import math
import os
import re
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from env.graph_utils import TOPOLOGY_PARAMS
from sim.physics_priors import check_physics_priors
from train_diffusion import action_to_params

FC_LIST = (2.4, 28.0, 38.0)
TOL = 0.05  # relative tolerance on window edges

# Perturbative / shunt / tuning caps (absolute legacy windows under electrical)
PERTURBATIVE_PF = frozenset({"C_load_pf", "C_base_pf", "C_tune_pf"})
# Ratio-encoded All-Pass coupling caps (not independent electrical windows)
RATIO_PF = frozenset({"C_cA_pf", "C_cB_pf"})
# Skip switch Rs — not action dimensions (tech constants)
SKIP_KEYS = frozenset({"R_on", "R_off"}) | RATIO_PF

# Templates are written for ~28 GHz; scale geometry to other fc for the check.
TEMPLATE_FC_GHZ = 28.0


def _parse_spice_number(raw: str) -> float | None:
    s = raw.strip().lower()
    try:
        if s.endswith("meg"):
            return float(s[:-3]) * 1e6
        if len(s) > 1 and s[-1] == "k" and s[:-1][0].isdigit():
            return float(s[:-1]) * 1e3
        if len(s) > 1 and s[-1] in "unp" and (s[0].isdigit() or s[0] == "."):
            return float(s[:-1]) * {"u": 1e-6, "n": 1e-9, "p": 1e-12}[s[-1]]
        return float(s)
    except ValueError:
        return None


def _parse_template_params(topology_name: str) -> dict[str, float]:
    path = os.path.join(REPO_ROOT, "specset", "templates", f"{topology_name.lower()}.sp")
    text = open(path).read()
    vals: dict[str, float] = {}
    for line in text.splitlines():
        if not line.lstrip().upper().startswith(".PARAM"):
            continue
        for m in re.finditer(
            r"([A-Za-z_][A-Za-z0-9_]*)\s*=\s*([0-9.+-]+(?:[eE][+-]?\d+)?[a-zA-Z]*)",
            line,
        ):
            parsed = _parse_spice_number(m.group(2))
            if parsed is not None:
                vals[m.group(1)] = parsed
    return vals


def _parse_param_float(val) -> float:
    s = str(val).strip().lower()
    if s.endswith("meg"):
        return float(s[:-3]) * 1e6
    if s.endswith("k"):
        return float(s[:-1]) * 1e3
    return float(s)


def _electrical_window(topology_name: str, key: str, fc_ghz: float) -> tuple[float, float]:
    keys = TOPOLOGY_PARAMS[topology_name]
    idx = keys.index(key)
    a0 = np.full(len(keys), 0.5, dtype=np.float64)
    a1 = a0.copy()
    a0[idx] = 0.0
    a1[idx] = 1.0
    spec = {"fc_ghz": fc_ghz}
    lo = _parse_param_float(
        action_to_params(a0, topology_name, spec, bounds="electrical")[key]
    )
    hi = _parse_param_float(
        action_to_params(a1, topology_name, spec, bounds="electrical")[key]
    )
    return (min(lo, hi), max(lo, hi))


def _expected_template_value(key: str, raw: float, fc_ghz: float) -> float:
    """Map 28 GHz template numbers onto other carrier frequencies when needed."""
    if key in PERTURBATIVE_PF or key.startswith("Z0") or key in ("G_I_scale", "G_Q_scale"):
        return raw
    # Resonant C/L and TL lengths in the templates are 28 GHz nominals.
    if key.endswith(("_pf", "_nh", "_mm")):
        return raw * (TEMPLATE_FC_GHZ / fc_ghz)
    return raw


def test_template_defaults_inside_electrical_windows():
    for topo, keys in TOPOLOGY_PARAMS.items():
        tmpl = _parse_template_params(topo)
        for key in keys:
            if key in SKIP_KEYS:
                continue
            assert key in tmpl, f"{topo}: missing .PARAM {key}"
            for fc in FC_LIST:
                lo, hi = _electrical_window(topo, key, fc)
                expected = _expected_template_value(key, tmpl[key], fc)
                # Switched_Line short/long are Δφ-partitioned, not λ/4-centered;
                # naive 28→fc template scaling is not meaningful for them.
                if topo == "Switched_Line" and key in ("L_short_mm", "L_long_mm"):
                    continue
                # All_Pass sections are deliberately centred *below* resonance
                # (ALLPASS_CENTRE_OFFSET) and drawn as a (centre, ratio) pair,
                # so they are not centred on the template's near-symmetric
                # default. At the top of the band the lower section approaches
                # the 5 fF floor and the window narrows further -- physics, not
                # drift. The nominal design is pinned instead by
                # tests/test_no_silent_clamping.test_allpass_midpoint_*.
                if topo == "All_Pass" and key in (
                    "L_apA_nh", "L_apB_nh", "C_brA_pf", "C_brB_pf",
                    "C_cA_pf", "C_cB_pf",
                ):
                    continue
                assert lo * (1.0 - TOL) <= expected <= hi * (1.0 + TOL), (
                    f"{topo}.{key} @ {fc} GHz: template-equivalent {expected:.6g} "
                    f"not in electrical window [{lo:.6g}, {hi:.6g}]"
                )


def test_l_quarter_mid_action_is_lam4():
    """Log-symmetric [0.4, 2.5]×λ/4 puts a=0.5 exactly on λ/4 (90°)."""
    lam4 = 47.43 / 28.0
    for topo in ("Loaded_Line", "Reflection_Type", "Vector_Modulator"):
        keys = TOPOLOGY_PARAMS[topo]
        action = np.full(len(keys), 0.5, dtype=np.float64)
        params = action_to_params(
            action, topo, {"fc_ghz": 28.0}, bounds="electrical"
        )
        L = _parse_param_float(params["L_quarter_mm"])
        assert abs(L - lam4) / lam4 < 0.02, f"{topo}: L={L:.4f} vs lam4={lam4:.4f}"


def test_switched_line_short_long_partition():
    """Electrical path must restore L_short < L_long partition."""
    keys = TOPOLOGY_PARAMS["Switched_Line"]
    action = np.full(len(keys), 0.5, dtype=np.float64)
    # Extremes: short at hi, long at lo — still short ≤ long at the shared edge.
    action[keys.index("L_short_mm")] = 1.0
    action[keys.index("L_long_mm")] = 0.0
    params = action_to_params(
        action, "Switched_Line", {"fc_ghz": 28.0}, bounds="electrical"
    )
    short = _parse_param_float(params["L_short_mm"])
    long = _parse_param_float(params["L_long_mm"])
    assert short <= long + 1e-9, f"short={short} > long={long}"
    lo_s, hi_s = _electrical_window("Switched_Line", "L_short_mm", 28.0)
    lo_l, hi_l = _electrical_window("Switched_Line", "L_long_mm", 28.0)
    lam4 = 47.43 / 28.0
    assert abs(hi_s - 0.8 * lam4) / lam4 < 0.05
    assert abs(lo_l - 0.8 * lam4) / lam4 < 0.05
    assert hi_s <= lo_l + 1e-6


def test_c_load_mid_action_sub_resonant_at_28ghz():
    keys = TOPOLOGY_PARAMS["Loaded_Line"]
    action = np.full(len(keys), 0.5, dtype=np.float64)
    params = action_to_params(
        action, "Loaded_Line", {"fc_ghz": 28.0}, bounds="electrical"
    )
    c_load = _parse_param_float(params["C_load_pf"])
    # Resonant blanket mid would be ~C0≈0.114…0.29; shunt mid must stay small.
    assert c_load < 0.15, f"C_load mid-action too large under electrical: {c_load}"


def test_c_load_mid_can_pass_physics_priors():
    keys = TOPOLOGY_PARAMS["Loaded_Line"]
    action = np.full(len(keys), 0.5, dtype=np.float64)
    # Mid-action L_quarter is λ/4 under log [0.4, 2.5]×λ/4
    action[keys.index("Z0_line")] = 0.5
    action[keys.index("L_quarter_mm")] = 0.5
    action[keys.index("C_load_pf")] = 0.5
    params = action_to_params(
        action, "Loaded_Line", {"fc_ghz": 28.0, "tech": 0},
        bounds="electrical", switch_model="ideal",
    )
    assert check_physics_priors("Loaded_Line", params, 28.0), params
    assert "R_on" in params and "R_off" in params


def test_reflection_type_mid_passes_physics_priors():
    """Electrical midpoint must keep Z0_branch/Z0_main in the hybrid window."""
    keys = TOPOLOGY_PARAMS["Reflection_Type"]
    params = action_to_params(
        np.full(len(keys), 0.5), "Reflection_Type",
        {"fc_ghz": 28.0, "tech": 0},
        bounds="electrical", switch_model="ideal",
    )
    z0m = _parse_param_float(params["Z0_main"])
    z0b = _parse_param_float(params["Z0_branch"])
    ratio = z0b / z0m
    assert 0.60 <= ratio <= 0.85, (z0m, z0b, ratio)
    assert check_physics_priors("Reflection_Type", params, 28.0), params


def test_perturbative_caps_not_c0_centered():
    """Mid of shunt/tune windows must stay well below series-resonant C0 at 28 GHz."""
    omega = 2.0 * math.pi * 28e9
    c0 = (1.0 / (omega * 50.0)) * 1e12
    for topo, key in (
        ("Loaded_Line", "C_load_pf"),
        ("Reflection_Type", "C_base_pf"),
        ("Reflection_Type", "C_tune_pf"),
    ):
        lo, hi = _electrical_window(topo, key, 28.0)
        mid = math.sqrt(lo * hi)
        assert mid < 0.5 * c0 or mid < 0.15, (
            f"{topo}.{key} mid={mid:.4f} still near C0={c0:.4f}"
        )


if __name__ == "__main__":
    test_template_defaults_inside_electrical_windows()
    print("OK template defaults in electrical windows")
    test_l_quarter_mid_action_is_lam4()
    print("OK L_quarter mid = λ/4")
    test_switched_line_short_long_partition()
    print("OK Switched_Line short/long partition")
    test_c_load_mid_action_sub_resonant_at_28ghz()
    print("OK C_load mid < 0.15 pF @ 28 GHz")
    test_c_load_mid_can_pass_physics_priors()
    print("OK C_load mid can pass physics priors")
    test_perturbative_caps_not_c0_centered()
    print("OK perturbative caps not C0-centered")
    print("ALL ELECTRICAL-BOUNDS TESTS PASSED")
