"""Regression: All-Pass electrical nominal must pass the physics prior.

The repaired (centre, ratio) decoder places section A at ω_res/ω_fc ≈ 8.94.
The prior previously allowed only ≤5× and rejected this good design.
"""
from __future__ import annotations

import os
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from env.graph_utils import TOPOLOGY_PARAMS
from sim.physics_priors import check_physics_priors
from train_diffusion import action_to_params


def _mid_params(fc_ghz: float) -> dict:
    keys = TOPOLOGY_PARAMS["All_Pass"]
    a = np.full(len(keys), 0.5, dtype=np.float32)
    return action_to_params(
        a, "All_Pass", {"fc_ghz": fc_ghz, "tech": 0},
        sizing="log", bounds="electrical", switch_model="ideal",
    )


def test_allpass_nominal_passes_prior_at_carriers():
    for fc in (2.4, 10.0, 28.0):
        params = _mid_params(fc)
        assert check_physics_priors("All_Pass", params, fc, pmax_mw=50.0), (
            f"All_Pass electrical midpoint rejected at fc={fc} GHz; params={params}"
        )


def test_allpass_nominal_mna_is_good():
    """Independent scorer: midpoint should be low-IL / high-RL / low phase err."""
    from sim.mna_scorer import mna_evaluate

    for fc in (2.4, 10.0, 28.0):
        params = _mid_params(fc)
        spec = {
            "fc_ghz": fc,
            "bw_pct": 20.0,
            "rms_phase_err_deg": 5.0,
            "rms_gain_err_db": 1.0,
            "max_il_db": 3.0,
            "min_rl_db": 12.0,
            "pmax_mw": 15.0,
            "max_area_mm2": 100.0,
            "tech": 0,
        }
        score, metrics = mna_evaluate("All_Pass", params, spec)
        assert metrics is not None, f"MNA failed at {fc} GHz (score={score})"
        assert float(metrics["il_db"]) < 1.0, metrics
        assert abs(float(metrics["rl_db"])) > 15.0, metrics
        assert abs(float(metrics["rms_phase_err_deg"])) < 10.0, metrics


def test_allpass_wild_values_still_rejected():
    """Prior must still reject clearly non-physical All_Pass sizings."""
    bad = {
        "L_apA_nh": "1e-6", "C_brA_pf": "10.0", "C_cA_pf": "20.0",
        "L_apB_nh": "1e-6", "C_brB_pf": "10.0", "C_cB_pf": "20.0",
        "R_on": "3", "R_off": "10k",
    }
    assert not check_physics_priors("All_Pass", bad, 28.0)


if __name__ == "__main__":
    test_allpass_nominal_passes_prior_at_carriers()
    test_allpass_nominal_mna_is_good()
    test_allpass_wild_values_still_rejected()
    print("OK allpass nominal prior regression")
