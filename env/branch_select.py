"""Branch-selecting topologies: one definition, three consequences.

Definition
----------
A topology is *branch-selecting* if its state map acts as a **permutation of
the device set** rather than as a change to device parameters. Concretely: the
multiset of per-device feature vectors under state s is a permutation of the
multiset under state s' (exact automorphism at identical sizing), so any
permutation-invariant pool of those features is state-blind.

Three consequences follow from that definition (not three independent facts):

1. ``cc = 2`` under device–device adjacency (the branches share no net).
2. Permutation-invariant state pooling is exactly state-blind (T6 theorem).
3. Measured Δφ depends entirely on off-branch isolation (finite R_off leaks).

Membership is decidable from the netlist: check whether flipping the state
permutes the device-feature multiset. Any new topology in the class must show
all three symptoms; any topology outside it must show none.
"""
from __future__ import annotations

import math
from typing import Iterable

import torch

from env.netlist_graph import (
    TOPOLOGY_NETLIST, N_STATES, build_circuit_graph, nominal_params,
    _normalize_name, graph_device_names,
)


# Canonical members under the definition above. Loaded_Line / Reflection_Type /
# Vector_Modulator tune parameters rather than permute devices.
BRANCH_SELECT_TOPOLOGIES: frozenset[str] = frozenset({
    "Switched_Line",
    "Switched_Filter",
    "All_Pass",
})


def _device_feature_rows(data) -> torch.Tensor:
    """[N_dev, F] feature matrix in graph device-row order."""
    return data["device"].x.detach().cpu()


def device_feature_multiset_equal(
    x_a: torch.Tensor, x_b: torch.Tensor, rtol: float = 1e-5, atol: float = 1e-6
) -> bool:
    """True iff the rows of x_a are a permutation of the rows of x_b."""
    if x_a.shape != x_b.shape:
        return False
    # Sort by a stable hash of each row so we compare multisets.
    def key(x: torch.Tensor) -> torch.Tensor:
        # Weighted sum of columns — enough to order distinct rows stably.
        w = torch.arange(1, x.size(1) + 1, dtype=x.dtype)
        return (x * w).sum(dim=-1)

    ia = torch.argsort(key(x_a))
    ib = torch.argsort(key(x_b))
    return torch.allclose(x_a[ia], x_b[ib], rtol=rtol, atol=atol)


def state_acts_as_device_permutation(
    topology_name: str,
    state_a: int = 0,
    state_b: int = 1,
    spec: dict | None = None,
    bounds: str = "electrical",
    switch_model: str = "ideal",
) -> bool:
    """Netlist decision procedure for branch-select membership.

    Builds the circuit graph at two states under identical (nominal) sizing
    and asks whether the device-feature rows are a permutation of each other.
    """
    name = _normalize_name(topology_name)
    spec = dict(spec or {"fc_ghz": 28.0, "tech": 0})
    params = nominal_params(name, spec, bounds=bounds, switch_model=switch_model)
    g_a = build_circuit_graph(name, state=state_a, params=params, bounds=bounds)
    g_b = build_circuit_graph(name, state=state_b, params=params, bounds=bounds)
    return device_feature_multiset_equal(
        _device_feature_rows(g_a), _device_feature_rows(g_b)
    )


def is_branch_selecting(topology_name: str, **kwargs) -> bool:
    """Decide membership from the netlist (not from a hand-maintained list)."""
    name = _normalize_name(topology_name)
    n = N_STATES.get(name, 2)
    if n < 2:
        return False
    # Use states 0 and 1 (for VM those are adjacent phase steps, not endpoints).
    return state_acts_as_device_permutation(name, 0, 1, **kwargs)


def classify_all(**kwargs) -> dict[str, bool]:
    return {t: is_branch_selecting(t, **kwargs) for t in TOPOLOGY_NETLIST}
