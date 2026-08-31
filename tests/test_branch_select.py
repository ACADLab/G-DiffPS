"""Unit tests for the branch-select class definition."""
from __future__ import annotations

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from env.branch_select import (
    BRANCH_SELECT_TOPOLOGIES,
    classify_all,
    is_branch_selecting,
    state_acts_as_device_permutation,
)
from env.netlist_graph import TOPOLOGY_NETLIST


def test_membership_matches_canonical_set():
    decided = classify_all(switch_model="ideal")
    for topo in TOPOLOGY_NETLIST:
        expect = topo in BRANCH_SELECT_TOPOLOGIES
        assert decided[topo] is expect, (
            f"{topo}: decided={decided[topo]} expected={expect}"
        )


def test_parameter_tuning_topologies_are_outside():
    for topo in ("Loaded_Line", "Reflection_Type", "Vector_Modulator"):
        assert not is_branch_selecting(topo, switch_model="ideal")


def test_switched_line_is_branch_selecting():
    assert is_branch_selecting("Switched_Line", switch_model="ideal")
    assert state_acts_as_device_permutation("Switched_Line")


if __name__ == "__main__":
    test_membership_matches_canonical_set()
    print("OK membership == canonical set")
    test_parameter_tuning_topologies_are_outside()
    print("OK parameter-tuning outside")
    test_switched_line_is_branch_selecting()
    print("OK Switched_Line is branch-selecting")
    print("ALL BRANCH-SELECT TESTS PASSED")
