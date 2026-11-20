"""balanced_sign_acc must refuse single-class labels."""
import os
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "tools"))

from d5b_role_transfer import balanced_sign_acc


def test_balanced_sign_none_when_single_class():
    y = np.full(20, -1.0)
    yhat = np.full(20, -1.0)
    assert balanced_sign_acc(y, yhat) is None


def test_balanced_sign_both_classes():
    y = np.array([1.0, 1.0, -1.0, -1.0])
    yhat = np.array([1.0, -1.0, -1.0, -1.0])
    assert abs(balanced_sign_acc(y, yhat) - 0.75) < 1e-9
