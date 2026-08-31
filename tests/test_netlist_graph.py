"""Validate TOPOLOGY_NETLIST against SPICE templates."""
from __future__ import annotations

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from env.netlist_graph import (
    TOPOLOGY_NETLIST,
    assert_matches_template,
    build_circuit_graph,
    connected_component_count,
    device_names,
    graph_device_names,
    nominal_params,
    sized_devices,
)

# device.x layout: 0..6 type one-hot, then
IDX_SIZED, IDX_SHUNT, IDX_NTERM, IDX_SWITCH = 7, 8, 9, 10
IDX_LOG_FC, IDX_ELEC, IDX_Z0 = 11, 12, 13
IDX_D_IN_MIN, IDX_D_IN_MAX = 14, 15
IDX_D_OUT_MIN, IDX_D_OUT_MAX = 16, 17
IDX_D_GND_MIN, IDX_D_GND_MAX = 18, 19
DEVICE_WIDTH = 7 + 13
EDGE_WIDTH = 6


TEMPLATES = {
    "Loaded_Line": "loaded_line.sp",
    "Switched_Line": "switched_line.sp",
    "Reflection_Type": "reflection_type.sp",
    "Switched_Filter": "switched_filter.sp",
    "Vector_Modulator": "vector_modulator.sp",
    "All_Pass": "all_pass.sp",
}


def test_incidence_matches_templates():
    for topo, fname in TEMPLATES.items():
        path = os.path.join(REPO_ROOT, "specset", "templates", fname)
        assert_matches_template(topo, path)


def test_single_connected_component():
    for topo in TOPOLOGY_NETLIST:
        assert connected_component_count(topo) == 1, topo


def test_build_circuit_graph_shapes():
    for topo in TOPOLOGY_NETLIST:
        g = build_circuit_graph(topo, {"fc_ghz": 28.0}, state=0)
        assert g["net"].x.shape[1] == 5
        assert g["device"].x.shape[1] == DEVICE_WIDTH
        assert g["device", "connects", "net"].edge_attr.shape[1] == EDGE_WIDTH
        assert g["device", "connects", "net"].edge_index.shape[0] == 2
        assert g["net", "rev_connects", "device"].edge_index.shape[0] == 2
        # Ground is always net 0
        assert g["net"].names[0] == "0"
        assert float(g["net"].x[0, 0]) == 1.0


def test_frequency_changes_features():
    g1 = build_circuit_graph("Loaded_Line", {"fc_ghz": 2.4}, state=0)
    g2 = build_circuit_graph("Loaded_Line", {"fc_ghz": 38.0}, state=0)
    assert not torch_allclose(g1["device"].x[:, IDX_LOG_FC], g2["device"].x[:, IDX_LOG_FC])


def torch_allclose(a, b):
    import torch
    return torch.allclose(a, b)


def test_switch_state_flips():
    g0 = build_circuit_graph("Switched_Line", {"fc_ghz": 28.0}, state=0)
    g1 = build_circuit_graph("Switched_Line", {"fc_ghz": 28.0}, state=1)
    assert not torch_allclose(g0["device"].x[:, IDX_SWITCH], g1["device"].x[:, IDX_SWITCH])


def test_port_devices_present_and_last():
    """Ports are appended after the real devices so name->row lookups hold."""
    for topo in TOPOLOGY_NETLIST:
        g = build_circuit_graph(topo, {"fc_ghz": 28.0}, state=0)
        names = list(g["device"].names)
        assert names == graph_device_names(topo)
        assert names[-2:] == ["P_in", "P_out"]
        assert names[: len(device_names(topo))] == device_names(topo)
        # Port impedance is carried, normalised against the 50-ohm reference.
        assert float(g["device"].x[-1, IDX_Z0]) == 0.0

        g_no = build_circuit_graph(topo, {"fc_ghz": 28.0}, state=0, include_ports=False)
        assert list(g_no["device"].names) == device_names(topo)


def test_sizing_changes_electrical_feature():
    """elec must respond to device values, not just type and frequency."""
    spec = {"fc_ghz": 28.0}
    base = nominal_params("Loaded_Line", spec)
    big = dict(base)
    big["C_load_pf"] = float(base.get("C_load_pf", 0.05)) * 10.0

    g0 = build_circuit_graph("Loaded_Line", spec, state=0, params=base)
    g1 = build_circuit_graph("Loaded_Line", spec, state=0, params=big)
    assert not torch_allclose(g0["device"].x[:, IDX_ELEC], g1["device"].x[:, IDX_ELEC])


def test_switch_electrical_size_tracks_impedance():
    """Off-state switches must read as a large impedance, not just a flipped bit."""
    import torch

    spec = {"fc_ghz": 28.0}
    names = graph_device_names("Switched_Line")
    i_short = names.index("R_in_short")
    g0 = build_circuit_graph("Switched_Line", spec, state=0)
    g1 = build_circuit_graph("Switched_Line", spec, state=1)
    # State 0 closes R_in_short (R_on), state 1 opens it (R_off).
    on_elec = float(g0["device"].x[i_short, IDX_ELEC])
    off_elec = float(g1["device"].x[i_short, IDX_ELEC])
    assert off_elec > on_elec + 1.0, (on_elec, off_elec)


def test_pin_distances_are_order_invariant():
    """Aggregating per-pin distances by (min, max) must not depend on pin order."""
    from env.netlist_graph import TOPOLOGY_NETLIST as NL, Dev

    topo = "Loaded_Line"
    g = build_circuit_graph(topo, {"fc_ghz": 28.0}, state=0)
    names = list(g["device"].names)
    i = names.index("C_in_load")
    orig = NL[topo]["C_in_load"]
    flipped = Dev(orig.dtype, tuple(reversed(orig.nets)), sizes=orig.sizes,
                  switch_param=orig.switch_param, on_in_states=orig.on_in_states,
                  aux_sizes=orig.aux_sizes, is_control_pins=orig.is_control_pins)
    NL[topo]["C_in_load"] = flipped
    try:
        g2 = build_circuit_graph(topo, {"fc_ghz": 28.0}, state=0)
    finally:
        NL[topo]["C_in_load"] = orig
    cols = slice(IDX_D_IN_MIN, IDX_D_GND_MAX + 1)
    assert torch_allclose(g["device"].x[i, cols], g2["device"].x[i, cols])


def test_sized_devices_cover_params():
    from env.graph_utils import TOPOLOGY_PARAMS
    for topo, keys in TOPOLOGY_PARAMS.items():
        sized = [p for _, p in sized_devices(topo)]
        for k in keys:
            assert k in sized, f"{topo}: missing sized param {k}"


if __name__ == "__main__":
    test_incidence_matches_templates()
    print("OK incidence")
    test_single_connected_component()
    print("OK connected")
    test_build_circuit_graph_shapes()
    print("OK shapes")
    test_frequency_changes_features()
    print("OK frequency")
    test_switch_state_flips()
    print("OK switch state")
    test_port_devices_present_and_last()
    print("OK port devices")
    test_sizing_changes_electrical_feature()
    print("OK sizing drives elec")
    test_switch_electrical_size_tracks_impedance()
    print("OK switch impedance")
    test_pin_distances_are_order_invariant()
    print("OK pin-order invariance")
    test_sized_devices_cover_params()
    print("OK sized params")
    print("ALL TESTS PASSED")
