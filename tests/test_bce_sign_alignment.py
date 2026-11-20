"""Regression test for BCE sign alignment with inverse-normalized yhat."""
import ast
import os
import sys

import torch
import torch.nn.functional as F

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "tools"))

from env.sign_alignment import sign_logits_from_normalized_pred


def test_bce_sign_logits_zero_crossing_alignment():
    """Verify sign_logits has identical zero-crossing to yhat."""
    y = torch.tensor([-2.0, -0.5, 0.5, 3.0, 5.0], dtype=torch.float)
    y_mean = y.mean()
    y_std = y.std()

    pred = torch.linspace(-3.0, 3.0, 100)
    yhat = pred * y_std + y_mean
    sign_logits = sign_logits_from_normalized_pred(pred, y_mean, y_std)

    assert torch.all((yhat > 0) == (sign_logits > 0))
    assert torch.all((yhat < 0) == (sign_logits < 0))

    target = (y - y_mean) / y_std
    perfect_logits = sign_logits_from_normalized_pred(target, y_mean, y_std)
    assert torch.all((perfect_logits > 0) == (y > 0))
    assert torch.all((perfect_logits < 0) == (y < 0))


def test_bce_sign_conflict_resolution():
    """Old BCE on raw pred fights MSE when μ>0; aligned BCE does not."""
    y = torch.tensor([1.0], dtype=torch.float)
    y_mean = torch.tensor(5.0)
    y_std = torch.tensor(2.0)

    pred = torch.tensor([-2.0], requires_grad=True)
    bce_old = F.binary_cross_entropy_with_logits(pred, torch.tensor([1.0]))
    bce_old.backward()
    grad_old = pred.grad.clone()
    assert grad_old.item() < 0

    pred_new = torch.tensor([-2.0], requires_grad=True)
    sign_logits = sign_logits_from_normalized_pred(pred_new, y_mean, y_std)
    bce_new = F.binary_cross_entropy_with_logits(sign_logits, torch.tensor([1.0]))
    bce_new.backward()
    grad_new = pred_new.grad.clone()
    assert abs(grad_new.item()) < abs(grad_old.item())


def _train_calls_use_helper(module_path: str) -> bool:
    with open(module_path) as fh:
        tree = ast.parse(fh.read())
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            name = getattr(func, "attr", None) or getattr(func, "id", None)
            if name == "sign_logits_from_normalized_pred":
                return True
    return False


def test_d4_and_d5b_call_shared_helper():
    d4_path = os.path.join(REPO_ROOT, "tools", "d4_local_sensitivity.py")
    d5b_path = os.path.join(REPO_ROOT, "tools", "d5b_role_transfer.py")
    assert _train_calls_use_helper(d4_path)
    assert _train_calls_use_helper(d5b_path)
