#!/usr/bin/env python3
"""Render G-DiffPS circuit graphs (the same objects get_topology_graph builds).

Does not import torch. Node types, labels, and undirected edges are copied
from env/graph_utils.py so the figures stay in lockstep with the GNN input.

Usage:
    python3 scripts/viz_topology_graphs.py
    python3 scripts/viz_topology_graphs.py --out results/circuit_graphs
"""
from __future__ import annotations

import argparse
from pathlib import Path

from graphviz import Graph

# One-hot order in NODE_TYPES: TLine, Switch, Cap, Ind, Res
TYPE_STYLE = {
    "TLine":  {"shape": "box",     "fillcolor": "#D6E8F5", "color": "#2E79B5", "fontcolor": "#1A4A70"},
    "Switch": {"shape": "diamond", "fillcolor": "#F5E4D6", "color": "#C06028", "fontcolor": "#6A3010"},
    "Cap":    {"shape": "ellipse", "fillcolor": "#D6EFE4", "color": "#1F8A65", "fontcolor": "#0D4A36"},
    "Ind":    {"shape": "hexagon", "fillcolor": "#E0E4F5", "color": "#5A6CC0", "fontcolor": "#2A3470"},
    "Res":    {"shape": "box",     "fillcolor": "#E8E8EC", "color": "#6B6B7B", "fontcolor": "#3A3A44"},
}

# (id, type, short_label, role) — comments from get_topology_graph
TOPOLOGIES = {
    "Loaded_Line": {
        "title": "Loaded Line  —  λ/4 TLine + switched shunt RC",
        "mechanism": "λ/4 TL + switched shunt RC",
        "params": "Z0_line, L_quarter_mm, C_load_pf, R_on, R_off",
        "nodes": [
            (0, "TLine",  "TLine",   "main λ/4 line"),
            (1, "Cap",    "C_in",    "input shunt load"),
            (2, "Switch", "SW_in",   "input R_on/R_off"),
            (3, "Cap",    "C_out",   "output shunt load"),
            (4, "Switch", "SW_out",  "output R_on/R_off"),
        ],
        # Bidirectional pairs from edge_index, stored once as undirected
        "edges": [(0, 1), (0, 2), (1, 2), (0, 3), (0, 4), (3, 4)],
        "clusters": {
            "input shunt": [1, 2],
            "output shunt": [3, 4],
        },
        "rankdir": "LR",
        "engine": "dot",
    },
    "Switched_Line": {
        "title": "Switched Line  —  dual TL path selection",
        "mechanism": "Dual TL path selection",
        "params": "Z0_line, L_short_mm, L_long_mm, R_on, R_off",
        "nodes": [
            (0, "Switch", "SW_in_s",  "short-path input"),
            (1, "Switch", "SW_in_l",  "long-path input"),
            (2, "TLine",  "T_short",  "short delay line"),
            (3, "TLine",  "T_long",   "long delay line"),
            (4, "Switch", "SW_out_s", "short-path output"),
            (5, "Switch", "SW_out_l", "long-path output"),
        ],
        "edges": [(0, 2), (2, 4), (1, 3), (3, 5)],
        "clusters": {
            "short path": [0, 2, 4],
            "long path": [1, 3, 5],
        },
        "rankdir": "LR",
        "engine": "dot",
    },
    "Reflection_Type": {
        "title": "Reflection Type  —  branchline hybrid + reflective caps",
        "mechanism": "Branchline hybrid + reflective caps",
        "params": "Z0_main, Z0_branch, L_quarter_mm, C_base_pf, C_tune_pf, R_on, R_off",
        "nodes": [
            (0, "TLine",  "T_top",    "coupler top"),
            (1, "TLine",  "T_bot",    "coupler bottom"),
            (2, "TLine",  "T_left",   "coupler left"),
            (3, "TLine",  "T_right",  "coupler right"),
            (4, "Cap",    "C_baseA",  "base cap path A"),
            (5, "Cap",    "C_tuneA",  "tune cap path A"),
            (6, "Switch", "SW_A",     "path A switch"),
            (7, "Cap",    "C_baseB",  "base cap path B"),
            (8, "Cap",    "C_tuneB",  "tune cap path B"),
            (9, "Switch", "SW_B",     "path B switch"),
        ],
        "edges": [
            (0, 2), (0, 3), (1, 2), (1, 3),          # coupler square
            (0, 4), (3, 4), (0, 5), (3, 5), (5, 6),  # reflective load A
            (1, 7), (2, 7), (1, 8), (2, 8), (8, 9),  # reflective load B
        ],
        "clusters": {
            "branchline coupler": [0, 1, 2, 3],
            "load A": [4, 5, 6],
            "load B": [7, 8, 9],
        },
        "rankdir": "TB",
        "engine": "dot",
    },
    "Switched_Filter": {
        "title": "Switched Filter  —  HPF / LPF switched π sections",
        "mechanism": "HPF/LPF switched pi-sections",
        "params": "C_hpf_pf, L_hpf_nh, L_lpf_nh, C_lpf_pf, R_on, R_off",
        "nodes": [
            (0, "Switch", "SW_in_H",  "HPF input"),
            (1, "Ind",    "Lp_H_in",  "HPF shunt L in"),
            (2, "Cap",    "C_H_ser",  "HPF series C"),
            (3, "Ind",    "Lp_H_out", "HPF shunt L out"),
            (4, "Switch", "SW_out_H", "HPF output"),
            (5, "Switch", "SW_in_L",  "LPF input"),
            (6, "Cap",    "Cp_L_in",  "LPF shunt C in"),
            (7, "Ind",    "L_L_ser",  "LPF series L"),
            (8, "Cap",    "Cp_L_out", "LPF shunt C out"),
            (9, "Switch", "SW_out_L", "LPF output"),
        ],
        "edges": [
            (0, 1), (0, 2), (2, 3), (2, 4),  # HPF
            (5, 6), (5, 7), (7, 8), (7, 9),  # LPF
        ],
        "clusters": {
            "HPF path": [0, 1, 2, 3, 4],
            "LPF path": [5, 6, 7, 8, 9],
        },
        "rankdir": "LR",
        "engine": "dot",
    },
    "Vector_Modulator": {
        "title": "Vector Modulator  —  I/Q VCVS sum",
        "mechanism": "I/Q VCVS sum (16-state)",
        "params": "Z0_line, L_quarter_mm, G_I_scale, G_Q_scale, R_on, R_off",
        "nodes": [
            (0, "TLine", "T_quad", "quadrature TLine"),
            (1, "Res",   "R_term", "quad terminator"),
            (2, "Res",   "E_I",    "I-channel VCVS"),
            (3, "Res",   "E_Q",    "Q-channel VCVS"),
            (4, "Res",   "R_drv",  "output driver"),
        ],
        "edges": [(0, 1), (0, 2), (0, 3), (2, 3), (2, 4), (3, 4)],
        "clusters": {},
        "rankdir": "TB",
        "engine": "dot",
    },
    "All_Pass": {
        "title": "All Pass  —  two cascaded bridged-T LC sections",
        "mechanism": "Bridged-T LC sections",
        "params": "L_apA_nh, C_brA_pf, C_cA_pf, L_apB_nh, C_brB_pf, C_cB_pf, R_on, R_off",
        "nodes": [
            (0, "Switch", "SW_in_A",  "section A input"),
            (1, "Ind",    "L_apA",    "series L A"),
            (2, "Cap",    "C_brA",    "bridge C A"),
            (3, "Cap",    "C_cA",     "center shunt C A"),
            (4, "Switch", "SW_out_A", "section A output"),
            (5, "Switch", "SW_in_B",  "section B input"),
            (6, "Ind",    "L_apB",    "series L B"),
            (7, "Cap",    "C_brB",    "bridge C B"),
            (8, "Cap",    "C_cB",     "center shunt C B"),
            (9, "Switch", "SW_out_B", "section B output"),
        ],
        "edges": [
            (0, 1), (0, 2), (1, 3), (1, 2), (2, 4), (1, 4),  # bridged-T A
            (5, 6), (5, 7), (6, 8), (6, 7), (7, 9), (6, 9),  # bridged-T B
        ],
        "clusters": {
            "bridged-T A": [0, 1, 2, 3, 4],
            "bridged-T B": [5, 6, 7, 8, 9],
        },
        "rankdir": "LR",
        "engine": "dot",
    },
}


def _node_label(nid: int, ntype: str, short: str, role: str) -> str:
    return f"{short}\\n{role}\\n[{nid} {ntype}]"


def build_graph(name: str, spec: dict) -> Graph:
    g = Graph(
        name=name,
        filename=name,
        engine=spec["engine"],
        format="png",
    )
    g.attr(
        rankdir=spec["rankdir"],
        splines="true",
        overlap="false",
        nodesep="0.55",
        ranksep="0.65",
        pad="0.3",
        bgcolor="white",
        fontname="Helvetica",
        label=f"{spec['title']}\\nGNN input: {len(spec['nodes'])} nodes, "
              f"{2 * len(spec['edges'])} directed edges (bidirectional)  |  "
              f"params: {spec['params']}",
        labelloc="t",
        fontsize="14",
        fontcolor="#222222",
    )
    g.attr("node", style="filled", fontname="Helvetica", fontsize="10", penwidth="1.6")
    g.attr("edge", color="#888899", penwidth="1.4", arrowsize="0.6")
    g.attr("graph", fontname="Helvetica")

    for nid, ntype, short, role in spec["nodes"]:
        style = TYPE_STYLE[ntype]
        g.node(
            str(nid),
            label=_node_label(nid, ntype, short, role),
            **style,
        )

    for u, v in spec["edges"]:
        g.edge(str(u), str(v), dir="none")

    for i, (cluster_name, member_ids) in enumerate(spec["clusters"].items()):
        with g.subgraph(name=f"cluster_{i}") as c:
            c.attr(
                label=cluster_name,
                color="#CCCCD4",
                style="rounded",
                fontcolor="#555566",
                fontsize="11",
                penwidth="1.0",
            )
            for nid in member_ids:
                c.node(str(nid))

    return g


def build_legend() -> Graph:
    g = Graph(name="legend", filename="legend", engine="dot", format="png")
    g.attr(rankdir="LR", bgcolor="white", pad="0.2",
           label="Node types (5-D one-hot in x)", labelloc="t", fontname="Helvetica")
    g.attr("node", style="filled", fontname="Helvetica", fontsize="11", penwidth="1.6")
    order = ["TLine", "Switch", "Cap", "Ind", "Res"]
    onehot = {
        "TLine":  "[1,0,0,0,0]",
        "Switch": "[0,1,0,0,0]",
        "Cap":    "[0,0,1,0,0]",
        "Ind":    "[0,0,0,1,0]",
        "Res":    "[0,0,0,0,1]",
    }
    for t in order:
        g.node(t, label=f"{t}\\n{onehot[t]}", **TYPE_STYLE[t])
    for a, b in zip(order, order[1:]):
        g.edge(a, b, style="invis")
    return g


def build_pipeline() -> Graph:
    """How get_topology_graph() feeds the GNN."""
    g = Graph(name="pipeline", filename="how_graph_is_created", engine="dot", format="png")
    g.attr(rankdir="LR", bgcolor="white", pad="0.35", nodesep="0.45", ranksep="0.7",
           label="How a circuit graph is created  (env/graph_utils.py → TopologyEncoder)",
           labelloc="t", fontsize="14", fontname="Helvetica")
    g.attr("node", fontname="Helvetica", fontsize="11", style="filled", penwidth="1.4")
    g.attr("edge", color="#888899", penwidth="1.6")

    box = dict(shape="box", fillcolor="#F4F4F7", color="#6B6B7B", fontcolor="#222222")
    g.node("name", "topology_name\\nstr, one of 6 families", **box)
    g.node("norm", "normalize\\n.lower().replace('_','')", **box)
    g.node("x", "node matrix x\\n[N, 5] one-hot type", shape="box", fillcolor="#D6E8F5", color="#2E79B5")
    g.node("e", "edge_index\\n[2, E] bidirectional", shape="box", fillcolor="#D6EFE4", color="#1F8A65")
    g.node("data", "PyG Data(x, edge_index)", **box)
    g.node("deg", "inject degree\\ncat([x, deg]) → [N, 6]", **box)
    g.node("gin", "3× GINConv\\n+ sum-pool + Linear", shape="box", fillcolor="#E0E4F5", color="#5A6CC0")
    g.node("z", "z_topo ∈ R^64", shape="ellipse", fillcolor="#F5E4D6", color="#C06028")

    g.edges([
        ("name", "norm"),
        ("norm", "x"),
        ("norm", "e"),
        ("x", "data"),
        ("e", "data"),
        ("data", "deg"),
        ("deg", "gin"),
        ("gin", "z"),
    ])
    return g


def write_all(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    pipeline = build_pipeline()
    for fmt in ("png", "svg"):
        pipeline.format = fmt
        pipeline.render(filename=str(out_dir / "00_how_graph_is_created"), cleanup=True)

    legend = build_legend()
    for fmt in ("png", "svg"):
        legend.format = fmt
        legend.render(filename=str(out_dir / "00_legend"), cleanup=True)

    for i, (name, spec) in enumerate(TOPOLOGIES.items(), start=1):
        g = build_graph(name, spec)
        stem = f"{i:02d}_{name}"
        g.save(str(out_dir / f"{stem}.dot"))
        for fmt in ("png", "svg"):
            g.format = fmt
            g.render(filename=str(out_dir / stem), cleanup=True)
        n, e = len(spec["nodes"]), len(spec["edges"])
        print(f"  {name:<20} {n} nodes, {e} undirected / {2*e} directed edges  →  {stem}.png")

    print(f"\nWrote GraphViz figures to {out_dir}")


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=repo / "results" / "circuit_graphs")
    args = ap.parse_args()
    write_all(args.out)


if __name__ == "__main__":
    main()
