"""Phase-1 diagnostics and typed-graph construction."""
from __future__ import annotations

import os
import sys

import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from env.netlist_graph import (
    N_NET_ROLES,
    TERMINAL_ROLE_IDX,
    TOPOLOGY_NETLIST,
    build_circuit_graph,
    device_terminals,
    incidence_from_graph,
)
from models.circuit_encoder import CircuitEncoder, CircuitTypedEncoder
from tools.graph_diagnostics import (
    g1_roundtrip,
    g2_permutation,
    g3_terminal_swap,
    g7_perturbation,
    permute_hetero,
    TEMPLATES,
)

SPEC = {"fc_ghz": 28.0}


def test_g1_all_topologies():
    for topo in TEMPLATES:
        r = g1_roundtrip(topo)
        assert r["pass"], r


def test_graph_incidence_matches_declared():
    for topo in TOPOLOGY_NETLIST:
        g = build_circuit_graph(topo, SPEC, state=0, include_ports=False)
        recovered = incidence_from_graph(g)
        declared = {
            k: tuple("0" if n in ("0", "GND") else n for n in dev.nets)
            for k, dev in TOPOLOGY_NETLIST[topo].items()
        }
        assert recovered == declared, (topo, recovered, declared)


def test_typed_net_role_width():
    g = build_circuit_graph("Loaded_Line", SPEC, state=0, typed=True)
    assert g["net"].x.shape[1] == 5 + N_NET_ROLES
    assert g["device", "connects", "net"].edge_type.ndim == 1
    assert int(g["device", "connects", "net"].edge_type.max()) < len(TERMINAL_ROLE_IDX)


def test_typed_vcvs_roles():
    g = build_circuit_graph("Vector_Modulator", SPEC, state=0, typed=True)
    names = list(g["device"].names)
    ei = g["device", "connects", "net"].edge_index
    et = g["device", "connects", "net"].edge_type
    di = names.index("E_I")
    roles = sorted(int(t) for t, src in zip(et.tolist(), ei[0].tolist()) if src == di)
    expect = sorted(
        TERMINAL_ROLE_IDX[r]
        for r in ("out_p", "out_n", "ctrl_p", "ctrl_n")
    )
    assert roles == expect


def test_g2_untyped_permutation():
    enc = CircuitEncoder()
    enc.eval()
    r = g2_permutation(enc, "Loaded_Line", typed=False, n_trials=4)
    assert r["pass"], r


def test_g2_typed_permutation():
    enc = CircuitTypedEncoder(use_pe=True)
    enc.eval()
    r = g2_permutation(enc, "Loaded_Line", typed=True, n_trials=4)
    assert r["pass"], r


def test_g3_runs():
    torch.manual_seed(0)
    enc = CircuitEncoder()
    enc.eval()
    r = g3_terminal_swap(enc, typed=False)
    assert "cases" in r
    torch.manual_seed(0)
    enc_t = CircuitTypedEncoder(use_pe=True)
    enc_t.eval()
    rt = g3_terminal_swap(enc_t, typed=True)
    assert rt["mos_present"]
    polar = next(c for c in rt["cases"] if c["name"] == "vcvs_out_ctrl")
    mos = next(c for c in rt["cases"] if c["name"] == "mos_gd")
    cap = next(c for c in rt["cases"] if c["name"] == "passive_cap_pins")
    assert polar["rel_l2"] > 1e-6
    assert mos["rel_l2"] > 1e-6
    assert polar["rel_l2"] > cap["rel_l2"]
    # Untyped encoder is more sensitive to a null capacitor pin-swap than
    # to a VCVS output/control swap — the Phase-0 gap G3 is meant to catch.
    ru = g3_terminal_swap(enc, typed=False)
    u_polar = next(c for c in ru["cases"] if c["name"] == "vcvs_out_ctrl")
    u_cap = next(c for c in ru["cases"] if c["name"] == "passive_cap_pins")
    assert u_polar["rel_l2"] < u_cap["rel_l2"]


def test_g7_runs():
    enc = CircuitTypedEncoder(use_pe=False)
    enc.eval()
    r = g7_perturbation(enc, typed=True)
    assert len(r["rows"]) == 5
    assert all(row["rel_l2"] >= 0.0 for row in r["rows"])


def test_permute_preserves_counts():
    g = build_circuit_graph("All_Pass", SPEC, state=0, typed=True)
    rng = __import__("numpy").random.default_rng(1)
    g2 = permute_hetero(g, rng)
    assert g2["net"].x.shape == g["net"].x.shape
    assert g2["device"].x.shape == g["device"].x.shape
    assert g2["device", "connects", "net"].edge_index.shape == (
        g["device", "connects", "net"].edge_index.shape
    )


def test_device_terminals_untyped_omits_gate():
    from sim.sky130.realizable import loaded_line_sky130_graph

    nl, _ = loaded_line_sky130_graph()
    mos = nl["M_in_path"]
    untyped = device_terminals(mos, typed=False)
    typed = device_terminals(mos, typed=True)
    assert len(untyped) == 2
    assert [r for _, r in typed] == ["drain", "gate", "source", "body"]
    assert "g_in" in [n for n, _ in typed]


def test_motifs_on_all_six():
    from env.rf_motifs import TOPOLOGY_MOTIFS, motif_instances
    from env.netlist_graph import TOPOLOGY_NETLIST, build_circuit_graph, device_names

    for topo in TOPOLOGY_NETLIST:
        inst = motif_instances(topo, device_names(topo))
        assert inst, topo
        assert len(inst) == len(TOPOLOGY_MOTIFS[topo]), (topo, inst)
        g = build_circuit_graph(topo, SPEC, state=0, typed=True)
        assert "motif" in g.node_types
        assert g["motif"].x.size(0) == len(inst)
        enc = CircuitTypedEncoder(use_pe=True)
        enc.eval()
        z = enc(topo, SPEC)
        assert z.shape == (1, 64)
    enc = CircuitTypedEncoder(use_pe=True)
    enc.eval()
    z = enc("Loaded_Line", SPEC)
    assert z.shape == (1, 64)
    z2, h = enc("Vector_Modulator", SPEC, return_device=True)
    assert z2.shape == (1, 64)
    assert h.dim() == 2


if __name__ == "__main__":
    test_g1_all_topologies()
    print("OK g1")
    test_graph_incidence_matches_declared()
    print("OK incidence")
    test_typed_net_role_width()
    test_typed_vcvs_roles()
    print("OK typed")
    test_g2_untyped_permutation()
    test_g2_typed_permutation()
    print("OK g2")
    test_g3_runs()
    print("OK g3")
    test_g7_runs()
    print("OK g7")
    test_device_terminals_untyped_omits_gate()
    print("OK mos terminals")
    test_motifs_on_all_six()
    print("OK motifs")
    print("ALL TESTS PASSED")
