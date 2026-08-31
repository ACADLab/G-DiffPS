"""Build results/graphs/comparison.html: old vs new graphs + improvement tables."""
from __future__ import annotations

import json
import os
import sys

import networkx as nx

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from env.graph_utils import get_topology_graph, NODE_TYPES
from env.netlist_graph import (
    TOPOLOGY_NETLIST, PORT_NETS, build_circuit_graph, connected_component_count,
)

TYPE_REV = {tuple(v): k for k, v in NODE_TYPES.items()}
TOPOS = list(TOPOLOGY_NETLIST.keys())


def _old_stats(topology: str) -> dict:
    data = get_topology_graph(topology)
    G = nx.Graph()
    for i in range(data.x.shape[0]):
        G.add_node(i)
    ei = data.edge_index.tolist()
    for a, b in zip(ei[0], ei[1]):
        if a < b:
            G.add_edge(a, b)
    return {
        "n": G.number_of_nodes(),
        "e": G.number_of_edges(),
        "cc": nx.number_connected_components(G),
    }


def _new_stats(topology: str) -> dict:
    data = build_circuit_graph(topology, {"fc_ghz": 28.0}, state=0)
    n_nets = len(data["net"].names)
    n_devs = len(data["device"].names)
    e = int(data["device", "connects", "net"].edge_index.shape[1])
    return {
        "nets": n_nets,
        "devices": n_devs,
        "e": e,
        "cc": connected_component_count(topology),
        "ports": PORT_NETS[topology],
    }


def build_html(out_path: str, rank_path: str | None = None) -> str:
    rank = {}
    if rank_path and os.path.exists(rank_path):
        with open(rank_path) as fh:
            rank = json.load(fh)

    rows = []
    for topo in TOPOS:
        old = _old_stats(topo)
        new = _new_stats(topo)
        png = f"{topo.lower()}.png"
        rows.append((topo, old, new, png))

    struct_rows = "\n".join(
        f"<tr><td>{t}</td>"
        f"<td>{o['n']}</td><td>{o['e']}</td><td>{o['cc']}</td>"
        f"<td>{n['nets']}</td><td>{n['devices']}</td><td>{n['e']}</td>"
        f"<td>{n['cc']}</td>"
        f"<td>{'yes' if o['cc'] > 1 and n['cc'] == 1 else ('—' if o['cc'] == 1 else 'no')}</td>"
        f"</tr>"
        for t, o, n, _ in rows
    )

    rank_rows = ""
    if rank:
        for name, m in rank.items():
            rank_rows += (
                f"<tr><td>{name}</td><td>{m.get('rank')}</td>"
                f"<td>{m.get('participation_ratio', 0):.3f}</td>"
                f"<td>{m.get('within_topo_var', 0):.4g}</td>"
                f"<td>{m.get('n_unique_vectors')}</td></tr>\n"
            )

    cards = []
    for topo, old, new, png in rows:
        cards.append(f"""
<section class="card">
  <h2>{topo.replace('_', ' ')}</h2>
  <p class="meta">old: n={old['n']} e={old['e']} cc={old['cc']}
     &nbsp;|&nbsp; new: nets={new['nets']} devices={new['devices']}
     e={new['e']} cc={new['cc']} ports={new['ports']}</p>
  <img src="{png}" alt="{topo} old vs new graph" loading="lazy"/>
</section>""")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>G-DiffPS graph comparison: component-adjacency vs bipartite net/device</title>
<style>
  :root {{
    --bg: #f7f5f0; --ink: #1a1a1a; --muted: #555; --line: #d4cfc4;
    --accent: #0b5fff;
  }}
  body {{
    margin: 0; font-family: "IBM Plex Sans", "Source Sans 3", sans-serif;
    background: var(--bg); color: var(--ink); line-height: 1.45;
  }}
  header {{
    padding: 2rem 1.5rem 1rem; max-width: 1100px; margin: 0 auto;
  }}
  h1 {{ font-size: 1.75rem; margin: 0 0 0.4rem; letter-spacing: -0.02em; }}
  .lede {{ color: var(--muted); max-width: 62ch; }}
  main {{ max-width: 1100px; margin: 0 auto; padding: 0 1.5rem 3rem; }}
  .card {{ margin: 1.75rem 0; }}
  .card h2 {{ font-size: 1.15rem; margin: 0 0 0.25rem; }}
  .meta {{ color: var(--muted); font-size: 0.85rem; margin: 0 0 0.6rem; }}
  .card img {{
    width: 100%; height: auto; display: block;
    border: 1px solid var(--line); background: #fff;
  }}
  table {{
    width: 100%; border-collapse: collapse; font-size: 0.9rem;
    margin: 0.75rem 0 1.5rem; background: #fff;
  }}
  th, td {{ border: 1px solid var(--line); padding: 0.45rem 0.6rem; text-align: left; }}
  th {{ background: #ebe6dc; font-weight: 600; }}
  h3 {{ margin-top: 2rem; }}
  code {{ font-size: 0.85em; }}
</style>
</head>
<body>
<header>
  <h1>Circuit graph rebuild</h1>
  <p class="lede">
    Left: legacy component-adjacency graphs used by the paper GIN encoder
    (missing GND/ports; three topologies disconnected). Right: bipartite
    device/net incidence transcribed from SPICE templates (KCL/KVL structure).
  </p>
</header>
<main>
{''.join(cards)}

<h3>Structure improvement</h3>
<table>
  <thead>
    <tr>
      <th>Topology</th>
      <th>Old n</th><th>Old e</th><th>Old cc</th>
      <th>New nets</th><th>New devices</th><th>New e</th><th>New cc</th>
      <th>Connectivity fixed</th>
    </tr>
  </thead>
  <tbody>
{struct_rows}
  </tbody>
</table>

<h3>Encoder rank diagnostic</h3>
<p class="lede">From <code>results/rank_diagnostic/rank_report.json</code>
(500 random specs). Legacy GIN embeddings collapse to ≤6 unique vectors;
circuit encoder expands rank and within-topology variance.</p>
<table>
  <thead>
    <tr>
      <th>Encoder</th><th>Rank</th><th>Participation ratio</th>
      <th>Within-topo var</th><th>Unique vectors</th>
    </tr>
  </thead>
  <tbody>
{rank_rows if rank_rows else '<tr><td colspan="5">rank_report.json not found</td></tr>'}
  </tbody>
</table>
</main>
</body>
</html>
"""
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as fh:
        fh.write(html)
    return out_path


def main():
    out = os.path.join(REPO_ROOT, "results", "graphs", "comparison.html")
    rank = os.path.join(REPO_ROOT, "results", "rank_diagnostic", "rank_report.json")
    path = build_html(out, rank)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
