"""switch_model='realistic' must be a usable benchmark, not a broken one.

T0.1c reported that realistic switches collapse the branch-selecting topologies
(Switched_Line Δφ 65.7° → 3.4°) and a seventh topology was proposed to fix it.
That collapse does not reproduce: see results/joint/seventh_topology_decision.json.
These tests keep it from coming back silently, and pin the margin so a future
change to TECH_SWITCH that would actually break the benchmark fails here.
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from env.graph_utils import TOPOLOGY_PARAMS
from sim.mna_scorer import solve_sparams
from sim.switch_model import TECH_SWITCH, r_off_eff
from train_diffusion import action_to_params

# Below this off-state impedance, series-switch leakage starts eating Δφ.
# Measured collapse boundary (retention < 0.5) is ~28 Ω at 28 GHz.
MIN_SAFE_R_OFF_OHM = 100.0

BRANCH_SELECT = ("Switched_Line", "Switched_Filter")
FCS = (2.4, 10.0, 28.0, 40.0)


def _dphi(topo: str, fc: float, tech: int, switch_model: str) -> float:
    keys = TOPOLOGY_PARAMS[topo]
    params = action_to_params(
        np.full(len(keys), 0.5), topo, {"fc_ghz": fc, "tech": tech},
        bounds="electrical", switch_model=switch_model,
    )
    _, s21_0 = solve_sparams(topo, params, fc, state=0)
    _, s21_1 = solve_sparams(topo, params, fc, state=1)
    p0 = math.degrees(math.atan2(s21_0.imag, s21_0.real))
    p1 = math.degrees(math.atan2(s21_1.imag, s21_1.real))
    return abs(((p1 - p0 + 180.0) % 360.0) - 180.0)


def test_branch_select_keeps_phase_under_realistic():
    """Δφ under realistic switches stays within 10% of ideal."""
    for topo in BRANCH_SELECT:
        for tech in sorted(TECH_SWITCH):
            for fc in FCS:
                d_ideal = _dphi(topo, fc, tech, "ideal")
                d_real = _dphi(topo, fc, tech, "realistic")
                assert d_ideal > 30.0, (
                    f"{topo} @ {fc} GHz has no usable Δφ even under ideal "
                    f"switches ({d_ideal:.2f}°) — sizing bounds problem, "
                    f"not a switch problem"
                )
                retention = d_real / d_ideal
                assert retention > 0.9, (
                    f"{topo} tech={tech} @ {fc} GHz: Δφ retention "
                    f"{retention:.3f} (ideal {d_ideal:.2f}° → realistic "
                    f"{d_real:.2f}°). The T0.1c collapse has reappeared; a "
                    f"series-shunt variant may now be justified."
                )


def test_tech_off_impedance_has_margin():
    """Every shipped tech sits well clear of the measured collapse boundary."""
    for tech in sorted(TECH_SWITCH):
        for fc in (28.0, 40.0):
            _, _, c_off = TECH_SWITCH[tech]
            r_off = r_off_eff(c_off, fc)
            assert r_off > MIN_SAFE_R_OFF_OHM, (
                f"tech={tech} C_off={c_off * 1e15:.0f} fF gives "
                f"R_off_eff={r_off:.1f} Ω at {fc} GHz, below the "
                f"{MIN_SAFE_R_OFF_OHM:.0f} Ω safety floor. Branch-select "
                f"topologies will lose Δφ."
            )


def test_switched_line_arm_partition_is_disjoint():
    """T0.4: short and long arms must not share a sizing window.

    When both arms fall through to one rule they are equal at mid-action, Δφ is
    zero, and the topology looks broken for reasons unrelated to switches.
    """
    fc = 28.0
    keys = TOPOLOGY_PARAMS["Switched_Line"]
    params = action_to_params(
        np.full(len(keys), 0.5), "Switched_Line", {"fc_ghz": fc, "tech": 0},
        bounds="electrical", switch_model="ideal",
    )
    l_short = float(params["L_short_mm"])
    l_long = float(params["L_long_mm"])
    assert l_long > 1.5 * l_short, (
        f"mid-action arms nearly equal (short={l_short:.4f} mm, "
        f"long={l_long:.4f} mm): the T0.4 arm partition has regressed"
    )


def test_allpass_midpoint_is_a_real_design():
    """All_Pass's box midpoint must be a working phase shifter.

    It used to read Δφ = 0.00° under *both* switch models, because its two
    bridged-T sections were sized from one window and were therefore identical
    at a = 0.5 — the two switch states were the same circuit. That made the
    topology look broken for reasons unrelated to switches, and it fed a
    degenerate graph to the topology encoder. ALLPASS_SECTION_SPLIT offsets the
    section windows; this pins the result.
    """
    fc = 28.0
    keys = TOPOLOGY_PARAMS["All_Pass"]
    for model in ("ideal", "realistic"):
        d = _dphi("All_Pass", fc, 0, model)
        assert 70.0 < d < 115.0, (
            f"All_Pass mid-action Δφ is {d:.2f}° under {model}; expected near "
            f"the 90° ideal step. If this reads ~0°, the section split has "
            f"regressed and the two states are the same circuit."
        )

    # And it must still be well matched there — the old midpoint was not
    # (RL 6.3 dB), so zero phase was not its only problem.
    params = action_to_params(
        np.full(len(keys), 0.5), "All_Pass", {"fc_ghz": fc, "tech": 0},
        bounds="electrical", switch_model="ideal",
    )
    s11, _ = solve_sparams("All_Pass", params, fc, state=0)
    rl_db = -20.0 * math.log10(max(abs(s11), 1e-12))
    assert rl_db > 15.0, f"All_Pass midpoint return loss {rl_db:.2f} dB"


def test_allpass_reparameterization_preserves_the_achievable_set():
    """The (centre, ratio) reparameterization must not shrink what is reachable.

    The point of reparameterizing rather than moving the bounds is that the
    envelope archive stays valid: r_star is a max over achievable *metrics*, so
    it survives a change of coordinates but not a change of achievable set.
    """
    fc = 28.0
    keys = TOPOLOGY_PARAMS["All_Pass"]
    rng = np.random.default_rng(0)
    best_dphi, best_rl = 0.0, 0.0
    for _ in range(1024):
        params = action_to_params(
            rng.random(len(keys)), "All_Pass", {"fc_ghz": fc, "tech": 0},
            bounds="electrical", switch_model="ideal",
        )
        try:
            s11a, s0 = solve_sparams("All_Pass", params, fc, state=0)
            _, s1 = solve_sparams("All_Pass", params, fc, state=1)
        except Exception:
            continue
        p0 = math.degrees(math.atan2(s0.imag, s0.real))
        p1 = math.degrees(math.atan2(s1.imag, s1.real))
        best_dphi = max(best_dphi, abs(((p1 - p0 + 180.0) % 360.0) - 180.0))
        best_rl = max(best_rl, -20.0 * math.log10(max(abs(s11a), 1e-12)))
    assert best_dphi > 175.0, (
        f"All_Pass can only reach {best_dphi:.1f}° after reparameterization; "
        f"the achievable set shrank and the envelope archive is now stale"
    )
    # Loose: 1024 random draws under-sample the frontier (the archive reaches
    # 38.7 dB). This guards against the match collapsing, not against drift.
    assert best_rl > 25.0, f"All_Pass best return loss only {best_rl:.1f} dB"


def test_allpass_sections_are_always_asymmetric():
    """Equal sections are a zero-phase design; the ratio floor excludes them."""
    fc = 28.0
    keys = TOPOLOGY_PARAMS["All_Pass"]
    rng = np.random.default_rng(1)
    for a in rng.random((256, len(keys))):
        p = action_to_params(
            a, "All_Pass", {"fc_ghz": fc, "tech": 0},
            bounds="electrical", switch_model="ideal",
        )
        ratio = float(p["L_apB_nh"]) / float(p["L_apA_nh"])
        assert ratio > 1.05, (
            f"section ratio {ratio:.3f} is ~1, which is the degenerate "
            f"zero-phase design the reparameterization exists to exclude"
        )


if __name__ == "__main__":
    test_branch_select_keeps_phase_under_realistic()
    print("OK branch-select phase retention under realistic switches")
    test_tech_off_impedance_has_margin()
    print("OK tech off-impedance margin")
    test_switched_line_arm_partition_is_disjoint()
    print("OK switched_line arm partition (T0.4)")
    test_allpass_midpoint_is_a_real_design()
    print("OK all_pass midpoint is a real design")
    test_allpass_reparameterization_preserves_the_achievable_set()
    print("OK all_pass achievable set preserved under reparameterization")
    test_allpass_sections_are_always_asymmetric()
    print("OK all_pass sections always asymmetric")
