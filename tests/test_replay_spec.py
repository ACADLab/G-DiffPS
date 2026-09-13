"""Replay must keep the full spec for circuit-encoder re-encoding."""
from __future__ import annotations

import os
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from train_diffusion import ReplayBuffer, _replay_spec


def test_replay_buffer_roundtrips_spec_dict():
    buf = ReplayBuffer(capacity=4)
    spec_dict = {"fc_ghz": 2.4, "tech": 2, "max_il_db": 4.0}
    buf.push(np.zeros(8, dtype=np.float32), "All_Pass", np.array([0.5]), 1.0,
             fc_ghz=2.4, spec_dict=spec_dict)
    spec_dict["tech"] = 0  # later mutation must not leak
    _, topos, _, _, fcs, specs = buf.sample(1)
    assert topos[0] == "All_Pass"
    assert fcs[0] == 2.4
    assert specs[0]["tech"] == 2
    assert specs[0]["fc_ghz"] == 2.4


def test_replay_spec_keeps_tech():
    out = _replay_spec({"tech": 1, "app": 2}, 14.0)
    assert out["tech"] == 1
    assert out["fc_ghz"] == 14.0
    assert _replay_spec(None, 28.0) == {"fc_ghz": 28.0}
