"""T1.5d: power is a hard prior gate; continuous w_power is removed."""
from __future__ import annotations

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from env.reward import WEIGHTS_AREA, compute_sim_reward
from sim.mna_scorer import estimate_pwr_mw, vm_drive_pwr_mw
from sim.physics_priors import check_physics_priors


def test_estimate_pwr_mw_quarantined():
    try:
        estimate_pwr_mw("Vector_Modulator", {"G_I_scale": 1.0, "G_Q_scale": 1.0})
        raise AssertionError("expected RuntimeError")
    except RuntimeError:
        pass


def test_weights_area_has_no_power():
    assert WEIGHTS_AREA.power == 0.0
    assert abs(WEIGHTS_AREA.area - 0.18) < 1e-9


def test_vm_pmax_gate():
    params = {"G_I_scale": 1.0, "G_Q_scale": 1.0, "Z0_line": 50.0, "L_quarter_mm": 1.69}
    worst = max(vm_drive_pwr_mw(params, state=s) for s in range(16))
    assert worst > 0.0
    assert check_physics_priors("Vector_Modulator", params, 28.0, pmax_mw=0.01) is False
    assert worst <= 100.0


def test_area_term_in_reward():
    metrics = {
        "rms_phase_err_deg": 2.0,
        "il_db": 1.0,
        "rl_db": 20.0,
        "gain_err_db": 0.3,
        "area_mm2": 10.0,
    }
    spec = {
        "rms_phase_err_deg": 5.0,
        "max_il_db": 5.0,
        "min_rl_db": 10.0,
        "rms_gain_err_db": 1.0,
        "pmax_mw": 15.0,
        "max_area_mm2": 20.0,
    }
    r_small = compute_sim_reward(metrics, spec, weights=WEIGHTS_AREA)
    r_big = compute_sim_reward({**metrics, "area_mm2": 19.0}, spec, weights=WEIGHTS_AREA)
    assert r_small > r_big


if __name__ == "__main__":
    test_estimate_pwr_mw_quarantined()
    test_weights_area_has_no_power()
    test_vm_pmax_gate()
    test_area_term_in_reward()
    print("OK power/area T1.5d")
