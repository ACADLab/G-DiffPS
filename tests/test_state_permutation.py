"""State-permutation blindness of symmetric pooling.

Claim under test, stated as a theorem rather than a measurement:

    Let a phase state transition s -> s' act on the device set as a
    permutation pi, so that h_{s'}[d] = h_s[pi(d)] for every device d. Then for
    any permutation-invariant readout P,

        P({h_s[d]}) = P({h_{s'}[d]})

    exactly -- independent of features, weights, architecture or training. A
    symmetric state pool is therefore *blind* to that transition.

Sum pooling is permutation invariant, so differencing after the device pool
returns zero. Differencing per aligned device row and pooling second does not.
Branch-switched networks -- switched-line, switched-filter, bridged-T all-pass
-- are exactly the circuits whose states are related this way, which is to say
the theorem applies to the canonical phase shifter rather than to a corner case.

The per-device difference is only well defined because every state of one
topology shares a single netlist, so device row i means the same device in
every state. Contrasting across *topologies* would break that alignment
silently, which is why `test_device_rows_align_across_states` guards it.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from env.netlist_graph import (
    TOPOLOGY_NETLIST, build_circuit_graph, nominal_params,
)
from models.circuit_encoder import CircuitEncoder

FC = 28.0
# Topologies whose two states exchange one branch for another.
BRANCH_SWAP = ["Switched_Line", "Switched_Filter", "All_Pass"]
# Of those, the ones that become exact graph automorphisms when the two branches
# are sized identically. Note this is a property of the *sizing*, not of the
# nominal point: `symmetrized_params` forces it below rather than relying on a
# nominal point happening to be degenerate. Switched_Line's nominal point is no
# longer symmetric (T0.4 partitions the arms 0.3-0.8 / 0.8-2.5 lam/4, which is
# what gives it any Delta-phi at all), and All_Pass's still is (a defect tracked
# in results/joint/SEVENTH_TOPOLOGY_DECISION.md). Neither should decide the test.
EXACT_AUTOMORPHISM = ["Switched_Line", "All_Pass"]

# Branch parameter pairs that must be equal for the state swap to be an
# automorphism of the sized graph.
_BRANCH_PAIRS = {
    "Switched_Line": [("L_short_mm", "L_long_mm")],
    "All_Pass": [("L_apA_nh", "L_apB_nh"),
                 ("C_brA_pf", "C_brB_pf"),
                 ("C_cA_pf", "C_cB_pf")],
}


def symmetrized_params(topology, spec=None):
    """nominal_params with the swapped branches forced to equal sizing."""
    params = dict(nominal_params(topology, spec or {"fc_ghz": FC}))
    for a, b in _BRANCH_PAIRS.get(topology, []):
        if a in params and b in params:
            params[b] = params[a]
    return params


def _encoder():
    torch.manual_seed(0)
    enc = CircuitEncoder()
    enc.eval()
    return enc


def _state_rows(enc, topology, params, state):
    g = build_circuit_graph(topology, {"fc_ghz": FC}, state=state, params=params)
    with torch.no_grad():
        _, h = enc.encode_state(g)
    return h


def _diffs(enc, topology, params):
    """(difference after device pool, difference before device pool)."""
    h0 = _state_rows(enc, topology, params, 0)
    h1 = _state_rows(enc, topology, params, 1)
    pooled = float((h0.sum(0) - h1.sum(0)).norm())
    per_device = float((h0 - h1).abs().sum(0).norm())
    return pooled, per_device


def permutes_device_features(topology, params) -> bool:
    """Does the state change leave the multiset of device features fixed?"""
    def multiset(state):
        x = build_circuit_graph(
            topology, {"fc_ghz": FC}, state=state, params=params
        )["device"].x.numpy()
        return sorted(tuple(np.round(r, 6)) for r in x)
    return multiset(0) == multiset(1)


def test_branch_swap_topologies_permute_device_features():
    """The precondition of the theorem holds for exactly the branch-swap set."""
    for topo in TOPOLOGY_NETLIST:
        params = symmetrized_params(topo)
        got = permutes_device_features(topo, params)
        want = topo in BRANCH_SWAP
        assert got == want, f"{topo}: permutes={got}, expected {want}"


def test_t04_partition_breaks_the_switched_line_automorphism():
    """Switched_Line's nominal point must NOT be an automorphism.

    Equal arms mean zero differential phase. The pooled difference vanishing at
    nominal would therefore be a symptom of a broken topology, not a property
    worth preserving -- this is the same failure mode as the All_Pass midpoint.
    """
    enc = _encoder()
    pooled, _ = _diffs(enc, "Switched_Line", nominal_params(
        "Switched_Line", {"fc_ghz": FC}))
    assert pooled > 1e-3, (
        "Switched_Line's nominal arms are symmetric again; the T0.4 partition "
        "has regressed and the topology has no differential phase at nominal"
    )


def test_symmetric_pool_is_blind_to_exact_permutations():
    """Pooling before differencing loses the state change entirely.

    Evaluated at explicitly symmetrized sizing, so the theorem is tested where
    its precondition provably holds rather than wherever the nominal point lands.
    """
    enc = _encoder()
    for topo in EXACT_AUTOMORPHISM:
        params = symmetrized_params(topo)
        pooled, per_device = _diffs(enc, topo, params)
        assert pooled < 1e-4, f"{topo}: pooled diff {pooled} should vanish"
        assert per_device > 1.0, f"{topo}: per-device diff {per_device} too small"
        assert per_device > 1e4 * max(pooled, 1e-12), topo


def test_per_device_contrast_survives_asymmetric_sizing():
    """With the branches sized apart the automorphism breaks, but the pooled
    difference stays orders of magnitude below the per-device one."""
    from train_diffusion import TOPOLOGY_PARAMS, action_to_params

    enc = _encoder()
    rng = np.random.default_rng(0)
    for topo in EXACT_AUTOMORPHISM:
        action = rng.random(len(TOPOLOGY_PARAMS[topo])).astype(np.float32)
        params = action_to_params(action, topo, {"fc_ghz": FC},
                                  sizing="log", bounds="electrical")
        pooled, per_device = _diffs(enc, topo, params)
        assert per_device > 5.0 * pooled, (topo, pooled, per_device)


def test_device_rows_align_across_states():
    """Per-device differencing is only sound while row i is the same device in
    every state. This is the invariant that breaks if states are ever drawn
    from different topologies."""
    for topo in TOPOLOGY_NETLIST:
        params = nominal_params(topo, {"fc_ghz": FC})
        names = None
        for state in (0, 1):
            g = build_circuit_graph(topo, {"fc_ghz": FC}, state=state, params=params)
            if names is None:
                names = list(g["device"].names)
            else:
                assert list(g["device"].names) == names, topo


if __name__ == "__main__":
    test_branch_swap_topologies_permute_device_features()
    print("OK permutation precondition")
    test_symmetric_pool_is_blind_to_exact_permutations()
    print("OK symmetric pool is state-blind")
    test_t04_partition_breaks_the_switched_line_automorphism()
    print("OK T0.4 partition breaks the nominal automorphism")
    test_per_device_contrast_survives_asymmetric_sizing()
    print("OK contrast survives asymmetric sizing")
    test_device_rows_align_across_states()
    print("OK device row alignment")
    print("ALL STATE-PERMUTATION TESTS PASSED")
