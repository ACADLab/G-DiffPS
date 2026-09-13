"""Empty specset must not silently serve a dummy spec on reset()."""
from __future__ import annotations

import os
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from env.phaseshifter_env import PhaseShifterEnv


def test_reset_raises_when_dataset_empty():
    env = PhaseShifterEnv()
    env.dataset = []
    env._specset_load_error = "synthetic empty dataset"
    with pytest.raises(RuntimeError, match="dummy spec"):
        env.reset()


def test_reset_raises_when_eval_pool_empty():
    env = PhaseShifterEnv()
    env.pool = "eval"
    env.eval_dataset = []
    with pytest.raises(RuntimeError, match="dummy spec"):
        env.reset()
