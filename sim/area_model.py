"""Board-level microstrip area model for TOPOLOGY_NETLIST circuits.

Medium (FRAMEWORK §4.2):
  eps_r = 3.2, h = 0.254 mm (10 mil)
  -> W(50 ohm) = 0.611 mm, eps_eff = 2.55

TL length anchor for the sanity table / rank budgets matches the SPICE
template constant λ/4[mm] = 47.43 / fc_GHz (ε_eff ≈ 2.5), so table lengths
1.694 mm @ 28 GHz and 19.76 mm @ 2.4 GHz reproduce exactly.

Bounding-box areas. Reflection_Type is the branchline square L_quarter_mm^2
(arms are NOT summed). Discrete passives and switches use fixed footprints.
"""
from __future__ import annotations

import math
from typing import Optional

from env.netlist_graph import TOPOLOGY_NETLIST, _normalize_name, nominal_params

# ---------------------------------------------------------------------------
# Medium constants (board microstrip). No free parameters in the sanity table.
# ---------------------------------------------------------------------------
EPS_R = 3.2
H_MM = 0.254
W50_MM = 0.611
EPS_EFF = 2.55
# Sanity-table / SPICE-template quarter-wave constant (mm·GHz).
LAM4_MM_GHZ = 47.43

# Footprints (mm^2). Switch package keepout by tech (0=PIN, 1=GaAs, 2=SOI).
# tech=0 / 1 keep 4.0 so the sanity table stays exact; SOI is slightly smaller.
SWITCH_FOOTPRINT_MM2 = {0: 4.0, 1: 4.0, 2: 3.6}
SWITCH_MM2 = SWITCH_FOOTPRINT_MM2[0]  # PIN / default alias
PASSIVE_0201_MM2 = 0.36
PASSIVE_0402_MM2 = 0.64  # unused by sanity rows
VCVS_MM2 = 4.0           # unused by sanity rows


def lam4_mm(fc_ghz: float) -> float:
    """Quarter-wave length in mm at fc_ghz (sanity-table / template constant)."""
    return LAM4_MM_GHZ / float(fc_ghz)


def hammerstad_w_mm(z0: float, h_mm: float = H_MM, eps_r: float = EPS_R) -> float:
    """Microstrip width. W(50 Ω) is pinned to W50_MM for the sanity table."""
    if abs(float(z0) - 50.0) < 1e-6:
        return W50_MM
    a = (z0 / 60.0) * math.sqrt((eps_r + 1) / 2) + (eps_r - 1) / (eps_r + 1) * (
        0.23 + 0.11 / eps_r
    )
    w_h = 8.0 * math.exp(a) / (math.exp(2 * a) - 2.0)
    return w_h * h_mm


def _param_float(params: dict, key: str, default: float) -> float:
    v = params.get(key, default)
    if v is None:
        return default
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip().lower().replace(" ", "")
    for suf, mul in (("meg", 1e6), ("k", 1e3), ("m", 1e-3), ("u", 1e-6),
                     ("n", 1e-9), ("p", 1e-12), ("f", 1e-15)):
        if s.endswith(suf):
            return float(s[: -len(suf)]) * mul
    return float(s)


def estimate_area_mm2(
    topology: str,
    params: Optional[dict] = None,
    fc_ghz: float = 28.0,
    bounds: str = "electrical",
    switch_model: str = "ideal",
    tech: int = 0,
) -> float:
    """Bounding-box area for a topology at the given (or nominal) sizing."""
    name = _normalize_name(topology)
    if params is None:
        params = nominal_params(
            name, {"fc_ghz": fc_ghz}, bounds=bounds, switch_model=switch_model,
        )
    netlist = TOPOLOGY_NETLIST[name]
    switch_fp = SWITCH_FOOTPRINT_MM2.get(int(tech), SWITCH_MM2)

    if name == "Reflection_Type":
        # Branchline hybrid: bounding box is L_quarter_mm^2 (arms not summed).
        l_q = _param_float(params, "L_quarter_mm", lam4_mm(fc_ghz))
        area = l_q * l_q
        for _dname, dev in netlist.items():
            if dev.dtype == "R_switch":
                area += switch_fp
            elif dev.dtype in ("C", "L"):
                area += PASSIVE_0201_MM2
            elif dev.dtype == "VCVS":
                area += VCVS_MM2
        return float(area)

    area = 0.0
    seen_tline_keys: set[str] = set()
    for _dname, dev in netlist.items():
        if dev.dtype == "TLine":
            length_key = dev.sizes
            if length_key in seen_tline_keys:
                continue
            seen_tline_keys.add(length_key)
            length = _param_float(params, length_key, lam4_mm(fc_ghz))
            z0_key = (dev.aux_sizes or ("Z0_line",))[0]
            z0 = _param_float(params, z0_key, 50.0)
            w = hammerstad_w_mm(z0)
            area += length * w
        elif dev.dtype == "R_switch":
            area += switch_fp
        elif dev.dtype in ("C", "L"):
            area += PASSIVE_0201_MM2
        elif dev.dtype == "VCVS":
            area += VCVS_MM2
    return float(area)


def ideal_lambda4_params(topology: str, fc_ghz: float) -> dict:
    """Params that pin TL lengths to the sanity-table λ/4 (and SL partition)."""
    name = _normalize_name(topology)
    lam4 = lam4_mm(fc_ghz)
    params = nominal_params(name, {"fc_ghz": fc_ghz}, bounds="electrical")
    out = dict(params)
    for k in list(out.keys()):
        if not k.endswith("_mm"):
            continue
        if name == "Switched_Line":
            if k == "L_short_mm":
                out[k] = 0.55 * lam4
            elif k == "L_long_mm":
                out[k] = 1.65 * lam4
            else:
                out[k] = lam4
        else:
            out[k] = lam4
    # Force Z0 to 50 for Loaded_Line / VM sanity rows (W50 pin).
    if "Z0_line" in out:
        out["Z0_line"] = 50.0
    return out


def reference_areas_mm2(fc_ghz: float, tech: int = 0) -> dict[str, float]:
    """Ideal λ/4 area per topology — used for rank-anchored budgets.

    ``tech`` selects switch package footprint via SWITCH_FOOTPRINT_MM2
    (PIN/GaAs 4.0 mm², SOI 3.6 mm²). Default tech=0 preserves the sanity table.
    """
    out = {}
    for name in TOPOLOGY_NETLIST:
        params = ideal_lambda4_params(name, fc_ghz)
        out[name] = estimate_area_mm2(name, params, fc_ghz=fc_ghz, tech=tech)
    return out


# Sanity-table targets from the work order (exact acceptance).
SANITY_TABLE = {
    ("Loaded_Line", 28.0): 9.76,
    ("Loaded_Line", 38.0): 9.48,
    ("Loaded_Line", 10.0): 11.62,
    ("Reflection_Type", 2.4): 399.9,
    ("Reflection_Type", 10.0): 31.9,
    ("All_Pass", 28.0): 18.88,
    ("Switched_Filter", 28.0): 18.16,
}
