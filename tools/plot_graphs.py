"""Side-by-side visualization: current component-adjacency vs bipartite net/device graphs."""
from __future__ import annotations

import os
import sys

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from env.graph_utils import get_topology_graph, NODE_TYPES
from env.netlist_graph import (
    DEVICE_TYPES, N_DEVICE_TYPES, TOPOLOGY_NETLIST, PORT_NETS,
    build_circuit_graph, connected_component_count,
)

TYPE_REV = {tuple(v): k for k, v in NODE_TYPES.items()}
COLORS = {
    "TLine": "#4C72B0", "Switch": "#DD8452", "Cap": "#55A868",
    "Ind": "#C44E52", "Res": "#8172B3", "R_switch": "#DD8452",
    "R_fixed": "#8172B3", "C": "#55A868", "L": "#C44E52",
    "VCVS": "#937860", "Port": "#000000", "net": "#CCCCCC",
    "GND": "#222222", "PORT_IN": "#2ca02c", "PORT_OUT": "#d62728",
}


def current_graph_nx(topology: str) -> tuple[nx.Graph, dict]:
    data = get_topology_graph(topology)
    G = nx.Graph()
    labels = {}
    for i, feat in enumerate(data.x.tolist()):
        t = TYPE_REV.get(tuple(feat), "?")
        G.add_node(i, ntype=t)
        labels[i] = f"{i}:{t}"
    ei = data.edge_index.tolist()
    for a, b in zip(ei[0], ei[1]):
        if a < b:
            G.add_edge(a, b)
    return G, labels


def bipartite_graph_nx(topology: str, state: int = 0) -> tuple[nx.Graph, dict, set, set]:
    data = build_circuit_graph(topology, {"fc_ghz": 28.0}, state=state)
    G = nx.Graph()
    labels = {}
    net_nodes, dev_nodes = set(), set()
    for i, name in enumerate(data["net"].names):
        ntype = "GND" if name == "0" else (
            "PORT_IN" if name == PORT_NETS[topology][0] else (
                "PORT_OUT" if name == PORT_NETS[topology][1] else "net"
            )
        )
        nid = f"n:{name}"
        G.add_node(nid, ntype=ntype, bipartite=0)
        labels[nid] = name
        net_nodes.add(nid)
    for i, name in enumerate(data["device"].names):
        # Read the type from the one-hot rather than the netlist table, so the
        # synthetic port terminations resolve too.
        dtype = DEVICE_TYPES[int(data["device"].x[i, :N_DEVICE_TYPES].argmax())]
        did = f"d:{name}"
        G.add_node(did, ntype=dtype, bipartite=1)
        labels[did] = name
        dev_nodes.add(did)
    ei = data["device", "connects", "net"].edge_index.tolist()
    for d_i, n_i in zip(ei[0], ei[1]):
        G.add_edge(f"d:{data['device'].names[d_i]}", f"n:{data['net'].names[n_i]}")
    return G, labels, net_nodes, dev_nodes


def _components(G: nx.Graph) -> int:
    return nx.number_connected_components(G)


def plot_topology(topology: str, out_dir: str):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: current
    G0, lab0 = current_graph_nx(topology)
    ax = axes[0]
    pos = nx.spring_layout(G0, seed=42)
    node_colors = [COLORS.get(G0.nodes[n]["ntype"], "#888") for n in G0.nodes]
    nx.draw_networkx_nodes(G0, pos, ax=ax, node_color=node_colors, node_size=500)
    nx.draw_networkx_edges(G0, pos, ax=ax, alpha=0.5)
    nx.draw_networkx_labels(G0, pos, labels=lab0, ax=ax, font_size=7)
    ax.set_title(
        f"CURRENT component-adjacency\n"
        f"n={G0.number_of_nodes()} e={G0.number_of_edges()} "
        f"cc={_components(G0)}"
    )
    ax.axis("off")

    # Right: bipartite
    G1, lab1, nets, devs = bipartite_graph_nx(topology)
    ax = axes[1]
    # Manual bipartite layout
    pos = {}
    for i, n in enumerate(sorted(nets)):
        pos[n] = (0.0, i)
    for i, d in enumerate(sorted(devs)):
        pos[d] = (1.0, i * len(nets) / max(len(devs), 1))
    node_colors = [COLORS.get(G1.nodes[n]["ntype"], "#888") for n in G1.nodes]
    nx.draw_networkx_nodes(G1, pos, ax=ax, node_color=node_colors, node_size=400)
    nx.draw_networkx_edges(G1, pos, ax=ax, alpha=0.4)
    nx.draw_networkx_labels(G1, pos, labels=lab1, ax=ax, font_size=6)
    ax.set_title(
        f"PROPOSED bipartite device/net\n"
        f"nets={len(nets)} devices={len(devs)} e={G1.number_of_edges()} "
        f"cc={connected_component_count(topology)}"
    )
    ax.axis("off")

    fig.suptitle(topology.replace("_", " "), fontsize=14, fontweight="bold")
    fig.tight_layout()
    path = os.path.join(out_dir, f"{topology.lower()}.png")
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"Wrote {path}")


def main():
    out_dir = os.path.join(REPO_ROOT, "results", "graphs")
    os.makedirs(out_dir, exist_ok=True)
    for topo in TOPOLOGY_NETLIST:
        plot_topology(topo, out_dir)
    print(f"All graphs written to {out_dir}")


if __name__ == "__main__":
    main()
