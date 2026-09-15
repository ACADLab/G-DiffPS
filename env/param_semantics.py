"""Design-variable semantics: role, bounds, current value, coupling group.

Action tokens are (device, parameter-role), not anonymous device scalars.
Kept out of ``train_diffusion`` so the graph builder can import it without a
cycle (``nominal_params`` already lazy-imports the decoder).

All_Pass is special: the electrical decoder maps action slots as
(centre, ratio) pairs, not six independent physical windows. R3 / D1a
must encode those control variables, otherwise the actor sees the wrong
object.
"""
from __future__ import annotations

import math
from functools import lru_cache

import numpy as np

PARAM_ROLES = [
    "length", "impedance", "capacitance", "inductance",
    "centre", "ratio", "coupling", "gain", "switch_width", "other",
]
PARAM_ROLE_IDX = {r: i for i, r in enumerate(PARAM_ROLES)}
N_PARAM_ROLES = len(PARAM_ROLES)

# Extra scalar channels concatenated onto the role one-hot for the actor.
# [value_01, is_log, log10(lo), log10(hi), group_id]
PARAM_AUX_DIM = 5
PARAM_CONTEXT_DIM = N_PARAM_ROLES + PARAM_AUX_DIM

# Graph node feature width for R3 parameter nodes.
PARAM_NODE_IN = PARAM_CONTEXT_DIM + 1  # + mutable

_PARAM_ROLE_BY_KEY = {
    "L_quarter_mm": "length", "L_short_mm": "length", "L_long_mm": "length",
    "Z0_line": "impedance", "Z0_main": "impedance", "Z0_branch": "impedance",
    "C_load_pf": "capacitance", "C_base_pf": "capacitance", "C_tune_pf": "capacitance",
    "C_hpf_pf": "capacitance", "C_lpf_pf": "capacitance",
    "L_hpf_nh": "inductance", "L_lpf_nh": "inductance",
    # All_Pass electrical decoder: A-slot = centre, B-slot = section ratio,
    # C_c* = coupling k against its own bridge cap.
    "L_apA_nh": "centre", "C_brA_pf": "centre",
    "L_apB_nh": "ratio", "C_brB_pf": "ratio",
    "C_cA_pf": "coupling", "C_cB_pf": "coupling",
    "G_I_scale": "gain", "G_Q_scale": "gain",
    "W_um": "switch_width", "L_um": "length", "nf": "other",
}

# Parameters decoded as a coupled pair share a coupling group.
_COUPLED_GROUPS = {
    "L_apA_nh": 1, "L_apB_nh": 1,
    "C_brA_pf": 2, "C_brB_pf": 2,
    "C_cA_pf": 3, "C_cB_pf": 4,
}

_ALLPASS_CENTRE_KEYS = {"L_apA_nh", "C_brA_pf"}
_ALLPASS_RATIO_KEYS = {"L_apB_nh", "C_brB_pf"}
_ALLPASS_COUPLING_KEYS = {"C_cA_pf", "C_cB_pf"}


def param_role_name(key: str) -> str:
    return _PARAM_ROLE_BY_KEY.get(key, "other")


def param_role_onehot(key: str) -> np.ndarray:
    v = np.zeros(N_PARAM_ROLES, dtype=np.float32)
    v[PARAM_ROLE_IDX[param_role_name(key)]] = 1.0
    return v


def coupled_group_id(key: str) -> int:
    return int(_COUPLED_GROUPS.get(key, 0))


def sized_param_roles(topology_name: str) -> np.ndarray:
    from env.netlist_graph import sized_devices
    return np.stack(
        [param_role_onehot(p) for _, p in sized_devices(topology_name)], axis=0,
    )


def _is_log_key(key: str) -> bool:
    if key.startswith("Z0") or key in ("G_I_scale", "G_Q_scale"):
        return False
    if key in ("L_short_mm", "L_long_mm"):
        return False
    if key in _ALLPASS_COUPLING_KEYS:
        return False  # k = C_c/C_br is linear in [1.2, ALLPASS_CC_MAX]
    if key in _ALLPASS_RATIO_KEYS:
        return True  # section ratio is log-uniform
    return True


def _allpass_ratio_bounds() -> tuple[float, float]:
    from train_diffusion import _allpass_ratio_range
    return _allpass_ratio_range()


def _allpass_coupling_bounds() -> tuple[float, float]:
    from train_diffusion import ALLPASS_CC_MAX
    return 1.2, float(ALLPASS_CC_MAX)


def _allpass_centre_bounds(
    kind: str, fc_ghz: float,
) -> tuple[float, float]:
    """Physical centre window for L or C at fc (mid of A/B geometric mean)."""
    from train_diffusion import (
        ALLPASS_CC_MAX, PHYSICAL_LIMITS_BY_SUFFIX, _allpass_centre_offset,
    )
    omega = 2.0 * math.pi * fc_ghz * 1e9
    Z0 = 50.0
    if kind == "L":
        centre = (Z0 / omega) * 1e9 * _allpass_centre_offset()
        clo, chi = PHYSICAL_LIMITS_BY_SUFFIX["_nh"]
        top = 1.0
    else:
        centre = (1.0 / (omega * Z0)) * 1e12 * _allpass_centre_offset()
        clo, chi = PHYSICAL_LIMITS_BY_SUFFIX["_pf"]
        top = ALLPASS_CC_MAX
    # Same half-width logic as _allpass_section at ratio = ALLPASS_SECTION_RATIO
    from train_diffusion import ALLPASS_SECTION_RATIO
    s = math.sqrt(max(ALLPASS_SECTION_RATIO, 1e-12))
    half = max(0.0, min(
        1.0,
        math.log10(max(centre / (clo * s), 1.0)),
        math.log10(max((chi / top) / centre, 1.0)),
    ))
    c_l = math.log10(max(centre, 1e-12))
    lo = 10 ** (c_l - half)
    hi = 10 ** (c_l + half)
    if hi <= lo:
        hi = lo + 1e-12
    return lo, hi


def invert_allpass_controls(params: dict) -> dict[str, float]:
    """Map physical section values → semantic control coordinates.

    Returns per-key semantic value used for value_01:
      L_apA / C_brA → geometric-mean centre
      L_apB / C_brB → section ratio B/A
      C_cA / C_cB   → coupling k = C_c / C_br
    """
    from env.netlist_graph import params_numeric
    p = params_numeric(params)
    l_a = float(p.get("L_apA_nh", 0.0))
    l_b = float(p.get("L_apB_nh", 0.0))
    c_a = float(p.get("C_brA_pf", 0.0))
    c_b = float(p.get("C_brB_pf", 0.0))
    cc_a = float(p.get("C_cA_pf", 0.0))
    cc_b = float(p.get("C_cB_pf", 0.0))
    return {
        "L_apA_nh": math.sqrt(max(l_a * l_b, 1e-30)),
        "L_apB_nh": l_b / max(l_a, 1e-30),
        "C_brA_pf": math.sqrt(max(c_a * c_b, 1e-30)),
        "C_brB_pf": c_b / max(c_a, 1e-30),
        "C_cA_pf": cc_a / max(c_a, 1e-30),
        "C_cB_pf": cc_b / max(c_b, 1e-30),
    }


def param_window(
    topology_name: str,
    key: str,
    spec_dict: dict | None = None,
    bounds: str = "electrical",
    switch_model: str = "ideal",
) -> tuple[float, float, bool]:
    """Physical or semantic (lo, hi, is_log) for one action key."""
    name = topology_name
    if name == "All_Pass" and bounds == "electrical":
        fc = float((spec_dict or {}).get("fc_ghz", 28.0))
        if key in _ALLPASS_CENTRE_KEYS:
            kind = "L" if key.startswith("L_") else "C"
            lo, hi = _allpass_centre_bounds(kind, fc)
            return lo, hi, True
        if key in _ALLPASS_RATIO_KEYS:
            lo, hi = _allpass_ratio_bounds()
            return lo, hi, True
        if key in _ALLPASS_COUPLING_KEYS:
            lo, hi = _allpass_coupling_bounds()
            return lo, hi, False
    return _param_window_cached(
        topology_name, key,
        float((spec_dict or {}).get("fc_ghz", 28.0)),
        int((spec_dict or {}).get("tech", 0)),
        bounds, switch_model,
    )


@lru_cache(maxsize=512)
def _param_window_cached(
    topology_name: str, key: str, fc_ghz: float, tech: int,
    bounds: str, switch_model: str,
) -> tuple[float, float, bool]:
    from env.graph_utils import TOPOLOGY_PARAMS
    from env.netlist_graph import params_numeric
    from train_diffusion import action_to_params

    keys = TOPOLOGY_PARAMS[topology_name]
    idx = keys.index(key)
    spec = {"fc_ghz": fc_ghz, "tech": tech}
    a0 = np.full(len(keys), 0.5, dtype=np.float64)
    a1 = a0.copy()
    a0[idx] = 0.0
    a1[idx] = 1.0
    lo = float(params_numeric(action_to_params(
        a0, topology_name, spec, sizing="log", bounds=bounds,
        switch_model=switch_model,
    ))[key])
    hi = float(params_numeric(action_to_params(
        a1, topology_name, spec, sizing="log", bounds=bounds,
        switch_model=switch_model,
    ))[key])
    if lo > hi:
        lo, hi = hi, lo
    if abs(hi - lo) < 1e-18:
        hi = lo + 1e-12
    return lo, hi, _is_log_key(key)


def value_01(value: float, lo: float, hi: float, is_log: bool) -> float:
    v = float(value)
    if is_log:
        lv = math.log10(max(abs(v), 1e-18))
        l0 = math.log10(max(abs(lo), 1e-18))
        l1 = math.log10(max(abs(hi), 1e-18))
        if abs(l1 - l0) < 1e-12:
            return 0.5
        return float(np.clip((lv - l0) / (l1 - l0), 0.0, 1.0))
    return float(np.clip((v - lo) / (hi - lo), 0.0, 1.0))


def param_aux_features(
    topology_name: str,
    key: str,
    value: float,
    spec_dict: dict | None = None,
    bounds: str = "electrical",
    switch_model: str = "ideal",
) -> np.ndarray:
    lo, hi, is_log = param_window(
        topology_name, key, spec_dict, bounds=bounds, switch_model=switch_model,
    )
    return np.array([
        value_01(value, lo, hi, is_log),
        1.0 if is_log else 0.0,
        math.log10(max(abs(lo), 1e-18)),
        math.log10(max(abs(hi), 1e-18)),
        float(coupled_group_id(key)) / 4.0,
    ], dtype=np.float32)


def param_context_row(
    topology_name: str,
    key: str,
    value: float,
    spec_dict: dict | None = None,
    bounds: str = "electrical",
    switch_model: str = "ideal",
) -> np.ndarray:
    """Role one-hot + bounds/value aux. Length PARAM_CONTEXT_DIM."""
    return np.concatenate([
        param_role_onehot(key),
        param_aux_features(
            topology_name, key, value, spec_dict,
            bounds=bounds, switch_model=switch_model,
        ),
    ]).astype(np.float32)


def param_context_matrix(
    topology_name: str,
    spec_dict: dict | None = None,
    params: dict | None = None,
    bounds: str = "electrical",
    switch_model: str = "ideal",
) -> np.ndarray:
    """[N_sized, PARAM_CONTEXT_DIM] aligned with ``sized_devices``."""
    from env.netlist_graph import nominal_params, params_numeric, sized_devices

    if params is None:
        params = nominal_params(
            topology_name, spec_dict, bounds=bounds, switch_model=switch_model,
        )
    numeric = params_numeric(params)
    if topology_name == "All_Pass" and bounds == "electrical":
        semantic = invert_allpass_controls(numeric)
    else:
        semantic = None
    rows = []
    for _, key in sized_devices(topology_name):
        if semantic is not None and key in semantic:
            val = float(semantic[key])
        else:
            val = float(numeric.get(key, 0.0))
        rows.append(param_context_row(
            topology_name, key, val, spec_dict,
            bounds=bounds, switch_model=switch_model,
        ))
    return np.stack(rows, axis=0)


def param_node_features(
    topology_name: str,
    spec_dict: dict | None = None,
    params: dict | None = None,
    bounds: str = "electrical",
    switch_model: str = "ideal",
) -> tuple[np.ndarray, list[str], list[str]]:
    """R3 parameter-node table: (X, param_keys, owner_device_names)."""
    from env.netlist_graph import sized_devices

    ctx = param_context_matrix(
        topology_name, spec_dict, params=params,
        bounds=bounds, switch_model=switch_model,
    )
    mutable = np.ones((ctx.shape[0], 1), dtype=np.float32)
    x = np.concatenate([ctx, mutable], axis=1)
    sized = sized_devices(topology_name)
    keys = [p for _, p in sized]
    owners = [d for d, _ in sized]
    return x, keys, owners
