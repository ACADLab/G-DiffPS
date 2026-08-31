"""Unit tests for schema v2 spec normalization and loaders."""
from __future__ import annotations

import json
import os
import sys
import tempfile

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from specset.schema import (
    SCHEMA_VERSION,
    SPEC_DIM,
    SchemaVersionError,
    load_specset,
    normalize_spec,
)


def _base_spec(**overrides):
    spec = {
        "fc_ghz": 28.0,
        "bw_pct": 30.0,
        "phase_coverage_deg": 360.0,
        "phase_bits": 5,
        "rms_phase_err_deg": 5.0,
        "rms_gain_err_db": 1.0,
        "max_il_db": 5.0,
        "min_rl_db": 10.0,
        "vdd": 1.8,
        "pmax_mw": 15.0,
        "tech": 0,
        "app": 2,
        "max_area_mm2": 50.0,
    }
    spec.update(overrides)
    return spec


def test_spec_dim_is_19():
    assert SPEC_DIM == 19
    assert normalize_spec(_base_spec()).shape == (19,)


def test_phase_bits_onehot_equidistant_from_analog():
    """bits=0 must not be nearer bits=3 than bits=6 in normalized space.

    One-hot phase_bits: differing in exactly two positions → L2 = sqrt(2)
    for any distinct pair. Ordinal encoding would put 0 closer to 3 than 6.
    """
    v0 = normalize_spec(_base_spec(phase_bits=0))
    v3 = normalize_spec(_base_spec(phase_bits=3))
    v6 = normalize_spec(_base_spec(phase_bits=6))

    d03 = float(np.linalg.norm(v0 - v3))
    d06 = float(np.linalg.norm(v0 - v6))
    d36 = float(np.linalg.norm(v3 - v6))

    assert abs(d03 - np.sqrt(2.0)) < 1e-5
    assert abs(d06 - np.sqrt(2.0)) < 1e-5
    assert abs(d36 - np.sqrt(2.0)) < 1e-5
    assert abs(d03 - d06) < 1e-6
    # Explicitly: analog is not closer to 3-bit than to 6-bit
    assert not (d03 < d06 - 1e-6)


def test_schema_mismatch_raises():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "bad_specset.json")
        with open(path, "w") as f:
            json.dump(
                {
                    "schema_version": SCHEMA_VERSION - 1,
                    "pool": "train",
                    "specs": [{"id": "x", "spec": _base_spec()}],
                },
                f,
            )
        try:
            load_specset(path)
            raise AssertionError("expected SchemaVersionError")
        except SchemaVersionError:
            pass


if __name__ == "__main__":
    test_spec_dim_is_19()
    test_phase_bits_onehot_equidistant_from_analog()
    test_schema_mismatch_raises()
    print("OK test_spec_normalize")
