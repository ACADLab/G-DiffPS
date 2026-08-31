"""Table 6 replacement: numerical rank of z_topo over a spec sweep.

Establishes that dying-ReLU / SAGE+LN / GIN on the current graph are pinned
at rank <= 6 by the input, while the circuit encoder's rank grows with
spec diversity.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, global_mean_pool, global_add_pool
from torch_geometric.utils import degree

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from env.graph_utils import get_topology_graph, TOPOLOGY_PARAMS
from env.netlist_graph import TOPOLOGY_NETLIST
from models.gnn_encoder import TopologyEncoder
from models.circuit_encoder import CircuitEncoder


class DyingReLUEncoder(nn.Module):
    """Intentionally brittle encoder that collapses embeddings (App A)."""

    def __init__(self, in_channels=5, hidden=64, out=64):
        super().__init__()
        self.lin1 = nn.Linear(in_channels, hidden)
        self.lin2 = nn.Linear(hidden, out)

    def forward(self, x, edge_index, batch=None):
        h = F.relu(self.lin1(x))
        h = F.relu(self.lin2(h))
        return global_mean_pool(h, batch)


class SAGELNEncoder(nn.Module):
    """Mean-pool GraphSAGE + post-pool LayerNorm (collapses geometry)."""

    def __init__(self, in_channels=5, hidden=64, out=64):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden)
        self.conv2 = SAGEConv(hidden, out)
        self.ln = nn.LayerNorm(out)

    def forward(self, x, edge_index, batch=None):
        h = F.relu(self.conv1(x, edge_index))
        h = self.conv2(h, edge_index)
        z = global_mean_pool(h, batch)
        return self.ln(z)


def participation_ratio(Z: np.ndarray) -> float:
    """PR = (sum s_i)^2 / sum s_i^2 of singular values — effective rank."""
    s = np.linalg.svd(Z, compute_uv=False)
    s = s[s > 1e-12]
    if s.size == 0:
        return 0.0
    return float((s.sum() ** 2) / (s ** 2).sum())


def numerical_rank(Z: np.ndarray, tol: float = 1e-3) -> int:
    s = np.linalg.svd(Z, compute_uv=False)
    return int(np.sum(s > tol * s.max())) if s.size else 0


def sweep_gin_family(encoder, n_specs=500, seed=0):
    rng = np.random.default_rng(seed)
    topos = list(TOPOLOGY_PARAMS.keys())
    graphs = {t: get_topology_graph(t) for t in topos}
    rows, labels = [], []
    encoder.eval()
    with torch.no_grad():
        for _ in range(n_specs):
            t = topos[int(rng.integers(0, len(topos)))]
            # Spec is ignored by these encoders — that is the point
            g = graphs[t]
            z = encoder(g.x, g.edge_index).squeeze(0).cpu().numpy()
            rows.append(z)
            labels.append(t)
    Z = np.stack(rows, axis=0)
    return Z, np.array(labels)


def sweep_circuit(encoder, n_specs=500, seed=0):
    rng = np.random.default_rng(seed)
    topos = list(TOPOLOGY_NETLIST.keys())
    rows, labels = [], []
    encoder.eval()
    encoder.clear_cache()
    with torch.no_grad():
        for _ in range(n_specs):
            t = topos[int(rng.integers(0, len(topos)))]
            fc = float(10 ** rng.uniform(np.log10(1.0), np.log10(40.0)))
            z = encoder(t, {"fc_ghz": fc}).squeeze(0).cpu().numpy()
            rows.append(z)
            labels.append(t)
    return np.stack(rows, axis=0), np.array(labels)


def within_topo_variance(Z, labels):
    vars_ = []
    for t in np.unique(labels):
        sub = Z[labels == t]
        if len(sub) < 2:
            continue
        vars_.append(float(sub.var(axis=0).mean()))
    return float(np.mean(vars_)) if vars_ else 0.0


def main():
    n = 500
    results = {}

    for name, enc in [
        ("dying_relu", DyingReLUEncoder()),
        ("sage_ln", SAGELNEncoder()),
        ("gin", TopologyEncoder()),
    ]:
        Z, labels = sweep_gin_family(enc, n_specs=n)
        results[name] = {
            "rank": numerical_rank(Z),
            "participation_ratio": participation_ratio(Z),
            "within_topo_var": within_topo_variance(Z, labels),
            "n_unique_vectors": int(np.unique(np.round(Z, 6), axis=0).shape[0]),
            "shape": list(Z.shape),
        }
        print(f"{name:12s} rank={results[name]['rank']:3d}  "
              f"PR={results[name]['participation_ratio']:.2f}  "
              f"within_var={results[name]['within_topo_var']:.3e}  "
              f"unique={results[name]['n_unique_vectors']}")

    circ = CircuitEncoder()
    Z, labels = sweep_circuit(circ, n_specs=n)
    results["circuit"] = {
        "rank": numerical_rank(Z),
        "participation_ratio": participation_ratio(Z),
        "within_topo_var": within_topo_variance(Z, labels),
        "n_unique_vectors": int(np.unique(np.round(Z, 6), axis=0).shape[0]),
        "shape": list(Z.shape),
    }
    print(f"{'circuit':12s} rank={results['circuit']['rank']:3d}  "
          f"PR={results['circuit']['participation_ratio']:.2f}  "
          f"within_var={results['circuit']['within_topo_var']:.3e}  "
          f"unique={results['circuit']['n_unique_vectors']}")

    out = os.path.join(REPO_ROOT, "results", "rank_diagnostic", "rank_report.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
