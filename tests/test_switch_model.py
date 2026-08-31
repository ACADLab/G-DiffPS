"""Switch model: tech constants, ideal vs realistic R_off_eff."""
from __future__ import annotations

import math
import os
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from env.graph_utils import TOPOLOGY_PARAMS, SLOT_ACTION_DIM
from sim.switch_model import r_off_eff, switch_params
from train_diffusion import action_to_params


def test_slot_action_dim_is_max_params():
    assert SLOT_ACTION_DIM == max(len(v) for v in TOPOLOGY_PARAMS.values())
    assert SLOT_ACTION_DIM == 6  # All_Pass
    for keys in TOPOLOGY_PARAMS.values():
        assert "R_on" not in keys and "R_off" not in keys


def test_ideal_matches_templates():
    p = switch_params(tech=0, fc_ghz=28.0, model="ideal")
    assert abs(p["R_on"] - 3.0) < 1e-9
    assert abs(p["R_off"] - 1e4) < 1e-6


def test_realistic_roff_at_28ghz():
    # C_off=20 fF → 1/(2π·28e9·20e-15) ≈ 284 Ω
    expected = 1.0 / (2.0 * math.pi * 28e9 * 20e-15)
    assert abs(r_off_eff(20e-15, 28.0) - expected) / expected < 1e-9
    p = switch_params(tech=0, fc_ghz=28.0, model="realistic")
    assert abs(p["R_off"] - expected) / expected < 0.01
    assert abs(p["R_off"] - 284.0) / 284.0 < 0.02


def test_action_to_params_injects_switch():
    keys = TOPOLOGY_PARAMS["Switched_Line"]
    action = np.full(len(keys), 0.5)
    p = action_to_params(
        action, "Switched_Line", {"fc_ghz": 28.0, "tech": 0},
        bounds="electrical", switch_model="ideal",
    )
    assert float(p["R_on"]) == 3.0
    assert abs(float(p["R_off"]) - 1e4) < 1.0
    p2 = action_to_params(
        action, "Switched_Line", {"fc_ghz": 28.0, "tech": 0},
        bounds="electrical", switch_model="realistic",
    )
    assert abs(float(p2["R_off"]) - 284.0) < 10.0


if __name__ == "__main__":
    test_slot_action_dim_is_max_params()
    print("OK slot action dim")
    test_ideal_matches_templates()
    print("OK ideal")
    test_realistic_roff_at_28ghz()
    print("OK realistic R_off @ 28 GHz")
    test_action_to_params_injects_switch()
    print("OK action_to_params injects switch")
    print("ALL SWITCH MODEL TESTS PASSED")
