"""Align sign-BCE logits with inverse-normalized magnitude predictions.

Training targets magnitude MSE on mean-centered labels:
    target = (y - y_mean) / y_std
    yhat   = pred * y_std + y_mean

Sign labels are on raw y (positive if y > 0). Applying BCE to ``pred``
directly fights MSE whenever y_mean != 0, because pred=0 is not the
physical zero. Use these helpers so BCE sees the same zero-crossing as yhat.
"""
from __future__ import annotations

import torch


def sign_logits_from_normalized_pred(
    pred: torch.Tensor,
    y_mean: torch.Tensor,
    y_std: torch.Tensor,
) -> torch.Tensor:
    """Logits whose sign matches ``yhat = pred * y_std + y_mean``.

    Algebra: sign(yhat) == sign(pred + y_mean / y_std) when y_std > 0.
    """
    return pred + (y_mean / y_std)
