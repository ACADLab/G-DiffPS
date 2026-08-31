"""Unit tests for sim/area_model.py sanity table (exact reproduction)."""
from __future__ import annotations

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from sim.area_model import (
    SANITY_TABLE,
    estimate_area_mm2,
    ideal_lambda4_params,
    reference_areas_mm2,
)


def test_sanity_table_exact():
    for (topo, fc), target in SANITY_TABLE.items():
        params = ideal_lambda4_params(topo, fc)
        got = estimate_area_mm2(topo, params, fc_ghz=fc)
        assert abs(got - target) < 0.12, f"{topo}@{fc}: {got} vs {target}"


def test_reference_areas_ordered_at_low_freq():
    areas = reference_areas_mm2(2.4)
    # At 2.4 GHz Reflection_Type (branchline box) dominates.
    assert areas["Reflection_Type"] > areas["All_Pass"]
    assert areas["Reflection_Type"] > areas["Switched_Filter"]


if __name__ == "__main__":
    test_sanity_table_exact()
    test_reference_areas_ordered_at_low_freq()
    print("OK area_model sanity")
