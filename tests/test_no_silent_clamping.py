"""Sampling windows must not run past the netlist's physical limits.

`clamp_spice_value` caps inductors and capacitors at 10 and TL lengths at 50 mm.
Under bounds='electrical' the windows are multiples of the resonant value at fc,
which scales as 1/fc, so at low carriers the window used to overrun the cap: 72%
of Switched_Line draws and 77% of Switched_Filter draws clamped at 1.26 GHz,
collapsing much of the action box onto the same few circuits.

The windows are now clipped to the reachable range before a value is drawn.
These tests pin that, and pin the mirrored limit table against the real clamp.
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.graph_utils import TOPOLOGY_PARAMS
from netlist.llm_netlist_gen import clamp_spice_value
from train_diffusion import PHYSICAL_LIMITS_BY_SUFFIX, action_to_params

FC_LIST = (1.0, 1.26, 2.0, 5.0, 14.0, 28.0, 40.0)
# Fraction of draws allowed to touch a boundary. Non-zero because a window whose
# end legitimately coincides with a limit will sit on it -- at 40 GHz a resonant
# cap is 0.08 pF and All_Pass's lower section sits at 8.9 fF, so its window
# genuinely runs down to the 5 fF floor. The failure this guards against is bulk
# truncation, which ran to 77%.
MAX_BOUNDARY_TOUCH = 0.05


def test_mirrored_limits_match_the_real_clamp():
    """The audit's table and clamp_spice_value must not drift apart."""
    for suf, (lo, hi) in PHYSICAL_LIMITS_BY_SUFFIX.items():
        key = f"X{suf}"
        assert float(clamp_spice_value(key, str(hi * 10.0))) == hi, (
            f"{suf}: clamp_spice_value caps at something other than {hi}"
        )
        assert float(clamp_spice_value(key, str(lo / 10.0))) == lo, (
            f"{suf}: clamp_spice_value floors at something other than {lo}"
        )


def test_no_topology_clamps_in_bulk():
    rng = np.random.default_rng(0)
    worst = ("", 0.0)
    for topo, keys in TOPOLOGY_PARAMS.items():
        for fc in FC_LIST:
            hits = 0
            n = 256
            for a in rng.random((n, len(keys))):
                p = action_to_params(a, topo, {"fc_ghz": fc, "tech": 0},
                                     bounds="electrical", switch_model="ideal")
                for k in keys:
                    suf = next((s for s in PHYSICAL_LIMITS_BY_SUFFIX
                                if k.lower().endswith(s)), None)
                    if suf is None:
                        continue
                    lo, hi = PHYSICAL_LIMITS_BY_SUFFIX[suf]
                    v = float(p[k])
                    if v >= hi * (1 - 1e-6) or v <= lo * (1 + 1e-6):
                        hits += 1
                        break
            frac = hits / n
            if frac > worst[1]:
                worst = (f"{topo} @ {fc} GHz", frac)
    assert worst[1] <= MAX_BOUNDARY_TOUCH, (
        f"{worst[0]}: {worst[1]:.1%} of draws land on a physical limit; the "
        f"electrical window is overrunning the clamp again"
    )


def test_allpass_midpoint_survives_the_clip():
    """Clipping must not move the nominal All_Pass design."""
    from sim.mna_scorer import solve_sparams
    keys = TOPOLOGY_PARAMS["All_Pass"]
    for fc in (1.0, 5.0, 28.0, 40.0):
        p = action_to_params(np.full(len(keys), 0.5), "All_Pass",
                             {"fc_ghz": fc, "tech": 0},
                             bounds="electrical", switch_model="ideal")
        _, s0 = solve_sparams("All_Pass", p, fc, state=0)
        _, s1 = solve_sparams("All_Pass", p, fc, state=1)
        p0 = math.degrees(math.atan2(s0.imag, s0.real))
        p1 = math.degrees(math.atan2(s1.imag, s1.real))
        d = abs(((p1 - p0 + 180.0) % 360.0) - 180.0)
        assert 70.0 < d < 115.0, (
            f"All_Pass nominal Δφ = {d:.1f}° at {fc} GHz, expected ~90°"
        )


if __name__ == "__main__":
    test_mirrored_limits_match_the_real_clamp()
    print("OK mirrored limits match clamp_spice_value")
    test_no_topology_clamps_in_bulk()
    print("OK no topology clamps in bulk")
    test_allpass_midpoint_survives_the_clip()
    print("OK all_pass midpoint survives the window clip")
