"""Unit tests for differentiable MNA scorer vs analytic / SPICE priors."""
from __future__ import annotations

import math
import os
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from sim.mna_scorer import solve_sparams, mna_evaluate, sparams_to_metrics
from sim.physics_priors import compute_loaded_line_s_params, compute_switched_line_s_params


def test_loaded_line_matches_analytic():
    params = {
        "Z0_line": "50", "L_quarter_mm": "1.69", "C_load_pf": "0.04",
        "R_on": "3", "R_off": "10000",
    }
    fc = 28.0
    for state in (0, 1):
        s11, s21 = solve_sparams("Loaded_Line", params, fc, state=state)
        a11, a21 = compute_loaded_line_s_params(
            50.0, 1.69, 0.04, 3.0, 10000.0, fc * 1e9, state,
        )
        a11 = complex(a11)
        a21 = complex(a21)
        assert abs(s11 - a11) < 0.05, (state, s11, a11)
        assert abs(s21 - a21) < 0.05, (state, s21, a21)


def test_switched_line_matches_analytic():
    params = {
        "Z0_line": "50", "L_short_mm": "1.69", "L_long_mm": "3.38",
        "R_on": "3", "R_off": "10000",
    }
    fc = 28.0
    for state in (0, 1):
        s11, s21 = solve_sparams("Switched_Line", params, fc, state=state)
        a11, a21 = compute_switched_line_s_params(
            50.0, 1.69, 3.38, 3.0, 10000.0, fc * 1e9, state,
        )
        a11, a21 = complex(a11), complex(a21)
        assert abs(s11 - a11) < 0.08, (state, s11, a11)
        assert abs(s21 - a21) < 0.08, (state, s21, a21)


def test_all_pass_solves_at_2p4():
    # Electrical-scale L ~ 3.3 nH at 2.4 GHz
    params = {
        "L_apA_nh": "3.32", "C_brA_pf": "1.33", "C_cA_pf": "2.66",
        "L_apB_nh": "3.32", "C_brB_pf": "1.33", "C_cB_pf": "2.66",
        "R_on": "3", "R_off": "10000",
    }
    spec = {
        "fc_ghz": 2.4, "max_il_db": 5.0, "min_rl_db": 10.0,
        "rms_phase_err_deg": 5.0, "rms_gain_err_db": 1.0,
    }
    score, metrics = mna_evaluate("All_Pass", params, spec)
    assert metrics is not None
    assert score > -5.0
    s11, s21 = solve_sparams("All_Pass", params, 2.4, state=0)
    assert abs(s21) > 0.1


def test_vector_modulator_states():
    params = {
        "Z0_line": "50", "L_quarter_mm": "1.69",
        "G_I_scale": "1.0", "G_Q_scale": "1.0",
        "R_on": "3", "R_off": "10000",
    }
    s11_0, s21_0 = solve_sparams("Vector_Modulator", params, 28.0, state=0)
    s11_4, s21_4 = solve_sparams("Vector_Modulator", params, 28.0, state=4)
    # State 0 ~ 0 deg, state 4 ~ -90 deg — phases should differ materially
    p0 = math.degrees(math.atan2(s21_0.imag, s21_0.real))
    p4 = math.degrees(math.atan2(s21_4.imag, s21_4.real))
    assert abs(((p4 - p0 + 180) % 360) - 180) > 40


def test_score_finite_all_topos():
    from env.graph_utils import TOPOLOGY_PARAMS
    spec = {
        "fc_ghz": 28.0, "max_il_db": 5.0, "min_rl_db": 10.0,
        "rms_phase_err_deg": 5.0, "rms_gain_err_db": 1.0,
    }
    for topo in TOPOLOGY_PARAMS:
        params = {k: "1.0" for k in TOPOLOGY_PARAMS[topo]}
        # Override with sane mid values where needed
        for k in params:
            if k.startswith("Z0"):
                params[k] = "50"
            elif k.endswith("_mm"):
                params[k] = "1.69"
            elif k.endswith("_pf"):
                params[k] = "0.1"
            elif k.endswith("_nh"):
                params[k] = "0.3"
            elif "G_" in k:
                params[k] = "1.0"
            elif k == "R_on":
                params[k] = "3"
            elif k == "R_off":
                params[k] = "10000"
        score, _ = mna_evaluate(topo, params, spec)
        assert np.isfinite(score), topo


if __name__ == "__main__":
    test_loaded_line_matches_analytic()
    print("OK loaded_line")
    test_switched_line_matches_analytic()
    print("OK switched_line")
    test_all_pass_solves_at_2p4()
    print("OK all_pass 2.4")
    test_vector_modulator_states()
    print("OK vector_modulator")
    test_score_finite_all_topos()
    print("OK all topos")
    print("ALL MNA TESTS PASSED")
