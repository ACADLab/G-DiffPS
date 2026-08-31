#!/usr/bin/env python3
"""H0+ smoke tests for topology registry and generic MNA scoring."""
from __future__ import annotations

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np

from env.netlist_graph import Dev
from env.topology_registry import register_topology, unregister_topology
from sim.mna_scorer import mna_score, solve_sparams
from train_diffusion import action_to_params


def _score_at_fc(name: str, fc: float) -> float:
    spec = {"fc_ghz": fc, "tech": 0, "max_il_db": 3.0, "min_rl_db": 10.0,
            "rms_phase_err_deg": 10.0, "max_area_mm2": 500.0, "rms_gain_err_db": 1.0}
    from env.graph_utils import TOPOLOGY_PARAMS
    act = np.full(len(TOPOLOGY_PARAMS[name]), 0.5)
    params = action_to_params(act, name, spec, bounds="electrical", switch_model="ideal")
    return mna_score(name, params, spec)


def test_series_l_shunt_switched_c() -> None:
    """Throwaway: series L + shunt switched C — not in the original six."""
    name = "_Smoke_SerL_ShuntSwC"
    netlist = {
        "Lmain": Dev("L", ("in", "mid"), sizes="L_smoke_nh"),
        "Rsw": Dev("R_switch", ("mid", "nld"), switch_param="R_sw",
                   on_in_states=(0,)),
        "Csh": Dev("C", ("nld", "0"), sizes="C_smoke_pf"),
        "Rout": Dev("R_fixed", ("mid", "out")),
    }
    register_topology(name, netlist, ("in", "out"), 2,
                      ["L_smoke_nh", "C_smoke_pf"], -90.0, overwrite=True)
    try:
        for fc in (7.96, 28.0, 38.0):
            s = _score_at_fc(name, fc)
            assert np.isfinite(s), f"non-finite score at {fc} GHz"
        print("  series L + shunt switched C: OK")
    finally:
        unregister_topology(name)


def test_shorted_stub() -> None:
    """TLine stub to ground exercises the b<0 TLine stamp branch."""
    name = "_Smoke_ShuntStub"
    netlist = {
        "Tstub": Dev("TLine", ("in", "0"), sizes="L_stub_mm", aux_sizes=("Z0_stub",)),
        "Rth": Dev("R_fixed", ("in", "out")),
        "Rsw": Dev("R_switch", ("out", "n2"), switch_param="R_sw", on_in_states=(1,)),
        "Lser": Dev("L", ("n2", "0"), sizes="L_tail_nh"),
    }
    register_topology(name, netlist, ("in", "out"), 2,
                      ["L_stub_mm", "Z0_stub", "L_tail_nh"], -90.0, overwrite=True)
    try:
        for fc in (7.96, 28.0, 38.0):
            s = _score_at_fc(name, fc)
            assert np.isfinite(s), f"non-finite at {fc}"
        print("  shorted TL stub: OK")
    finally:
        unregister_topology(name)


def test_no_path_state() -> None:
    """Topology with no in→out path in state 1 — rejected by validity filter."""
    from topology.compose import validate_composed, compose_cascade, Section, SecKind
    sec = Section(kind=SecKind.SW_SER_BYPASS, polarity=1)
    ct = compose_cascade((sec,), "_Smoke_NoPath")
    ok, reason = validate_composed(ct)
    # Bypass-only may still have path; use explicit disconnected layout
    netlist = {
        "Rsw0": Dev("R_switch", ("in", "mid"), switch_param="R_sw", on_in_states=(0,)),
        "Lser": Dev("L", ("mid", "out"), sizes="L_np_nh"),
        "Rsw1": Dev("R_switch", ("in", "niso"), switch_param="R_sw", on_in_states=(1,)),
        "Ciso": Dev("C", ("niso", "0"), sizes="C_np_pf"),
    }
    ct2 = type(ct)(name="_Smoke_NoPath2", sections=(sec,), netlist=netlist,
                   param_keys=["L_np_nh", "C_np_pf"], device_count=4, has_switch=True)
    ok2, reason2 = validate_composed(ct2)
    assert not ok2, f"expected no-path rejection, got {reason2}"
    print(f"  no-path state rejection ({reason2}): OK")


def main() -> int:
    print("H0+ registry smoke tests:")
    test_series_l_shunt_switched_c()
    test_shorted_stub()
    test_no_path_state()
    print("all smoke tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
