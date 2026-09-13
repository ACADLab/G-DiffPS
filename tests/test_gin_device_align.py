"""GIN node embeddings must align to sized_devices by name, not i % n_nodes."""
from __future__ import annotations

import os
import sys

import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from env.graph_utils import (
    GIN_NODE_NAMES,
    get_topology_graph,
    gin_device_rows,
    gin_node_index,
)
from env.netlist_graph import sized_devices


def test_gin_node_count_matches_graph():
    for topo, names in GIN_NODE_NAMES.items():
        g = get_topology_graph(topo)
        assert g.x.size(0) == len(names), (topo, g.x.size(0), len(names))


def test_every_sized_device_resolves_to_a_gin_node():
    for topo in GIN_NODE_NAMES:
        g = get_topology_graph(topo)
        sized = sized_devices(topo)
        rows = gin_device_rows(g.x, topo, sized)
        assert len(rows) == len(sized), topo
        for (dname, _), row in zip(sized, rows):
            idx = gin_node_index(topo, dname)
            assert torch.equal(row, g.x[idx]), (topo, dname, idx)


def test_all_pass_does_not_cycle_switch_into_inductor_slot():
    """Regression: i % n_nodes mapped L_apA_ser1 onto the input switch."""
    topo = "All_Pass"
    sized = sized_devices(topo)
    assert sized[0][0] == "L_apA_ser1"
    assert gin_node_index(topo, "L_apA_ser1") == 1
    assert gin_node_index(topo, "L_apA_ser2") == 1
    # Cycling would have used node 0 (R_in_apA, a Switch).
    assert GIN_NODE_NAMES[topo][0] == "R_in_apA"
    assert GIN_NODE_NAMES[topo][1] == "L_apA"
