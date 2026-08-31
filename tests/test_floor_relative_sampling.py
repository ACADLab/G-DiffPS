"""House convention: fields with a physical floor are sampled relative to it.

Guards the two fields that follow it (rms_phase_err_deg over the quantization
floor, max_area_mm2 over the rank-ordered reference areas) and the invariant that
protected the area anchors when the sizing rule changed under T0.4.
"""
from __future__ import annotations

import os
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from sim.area_model import SANITY_TABLE, lam4_mm, reference_areas_mm2
from specset.schema import (
    PHASE_KAPPA_RANGE,
    SPEC_BOUNDS,
    effective_phase_floor_deg,
    is_feasible_phase_err,
    quantization_floor_deg,
    sample_rms_phase_err_deg,
)

BITS = (0, 3, 4, 5, 6)
COVERAGES = (90.0, 135.0, 180.0, 270.0, 360.0)


def test_phase_draws_are_never_infeasible():
    """Floor-relative sampling cannot produce a spec below the quantization floor."""
    rng = np.random.default_rng(0)
    for bits in BITS:
        for cov in COVERAGES:
            if bits >= 5 and cov < 180.0:
                continue  # excluded by the S2 coupling rule
            for _ in range(400):
                rms, kappa = sample_rms_phase_err_deg(bits, cov, rng)
                spec = {"phase_bits": bits, "phase_coverage_deg": cov,
                        "rms_phase_err_deg": rms}
                assert is_feasible_phase_err(spec), (
                    f"bits={bits} cov={cov} rms={rms:.4f} kappa={kappa:.3f} "
                    f"is below the feasibility margin"
                )
                lo, hi = PHASE_KAPPA_RANGE
                assert lo - 1e-9 <= kappa <= hi + 1e-9, kappa


def test_phase_floor_reference_values():
    """Pin the floor table the convention was specified against."""
    expected = {
        (4, 180.0): 3.2476,
        (5, 180.0): 1.6238,
        (6, 180.0): 0.8119,
        (3, 360.0): 12.9904,
    }
    for (bits, cov), want in expected.items():
        got = quantization_floor_deg(bits, cov)
        assert abs(got - want) < 1e-3, (bits, cov, got, want)


def test_analog_has_a_floor_too():
    """phase_bits=0 has no quantization floor but must not fall back to a box."""
    assert quantization_floor_deg(0, 180.0) is None
    f0 = effective_phase_floor_deg(0, 180.0)
    f6 = effective_phase_floor_deg(6, 180.0)
    assert abs(f0 - f6) < 1e-12, (
        "analog proxy floor should match the finest digital grid"
    )
    # And it must scale with coverage, not sit at a constant.
    assert effective_phase_floor_deg(0, 360.0) > 1.9 * f0


def test_spec_bounds_cover_the_reachable_span():
    """SPEC_BOUNDS for a derived field must not clip what the rule can produce."""
    lo_b, hi_b = SPEC_BOUNDS["rms_phase_err_deg"]
    k_lo, k_hi = PHASE_KAPPA_RANGE
    reach_lo, reach_hi = float("inf"), 0.0
    for bits in BITS:
        for cov in COVERAGES:
            if bits >= 5 and cov < 180.0:
                continue
            f = effective_phase_floor_deg(bits, cov)
            reach_lo = min(reach_lo, f * k_lo)
            reach_hi = max(reach_hi, f * k_hi)
    assert lo_b <= reach_lo, (lo_b, reach_lo)
    assert hi_b >= reach_hi, (hi_b, reach_hi)


def test_area_anchors_are_independent_of_the_sizing_rule():
    """T0.4 moved nominal TL length 1.45 -> 1.00 lam/4.

    reference_areas_mm2 pins lam/4 explicitly rather than reading the action
    mid-point, which is why max_area_mm2 anchors survived that change. Lock it in:
    the anchors must reproduce the published sanity table exactly.
    """
    for (topo, fc), want in SANITY_TABLE.items():
        got = reference_areas_mm2(fc, tech=0)[topo]
        # 0.5% tolerance absorbs the rounding in the published table.
        assert abs(got - want) <= max(0.05, 0.005 * want), (topo, fc, got, want)


def test_t04_nominal_is_log_centred_on_quarter_wave():
    """Mid-action must sit at lam/4, not at the arithmetic mean of the window."""
    from env.graph_utils import TOPOLOGY_PARAMS
    from train_diffusion import action_to_params

    fc = 28.0
    lam4 = lam4_mm(fc)
    for topo in ("Loaded_Line", "Reflection_Type", "Vector_Modulator"):
        keys = TOPOLOGY_PARAMS[topo]
        params = action_to_params(
            np.full(len(keys), 0.5), topo, {"fc_ghz": fc, "tech": 0},
            bounds="electrical", switch_model="ideal",
        )
        ratio = float(params["L_quarter_mm"]) / lam4
        assert abs(ratio - 1.0) < 0.01, (
            f"{topo} mid-action L_quarter is {ratio:.3f} x lam/4; T0.4 log "
            f"scaling has regressed (linear scaling would give 1.45)"
        )


if __name__ == "__main__":
    test_phase_draws_are_never_infeasible()
    print("OK phase draws never infeasible")
    test_phase_floor_reference_values()
    print("OK phase floor reference table")
    test_analog_has_a_floor_too()
    print("OK analog proxy floor")
    test_spec_bounds_cover_the_reachable_span()
    print("OK SPEC_BOUNDS covers reachable span")
    test_area_anchors_are_independent_of_the_sizing_rule()
    print("OK area anchors reproduce sanity table")
    test_t04_nominal_is_log_centred_on_quarter_wave()
    print("OK T0.4 log-centred nominal")
