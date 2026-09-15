"""Reflection_Type phase-target audit.

``rms_phase_err_deg`` against ideal −22.5° is a discontinuous / poorly
interpolable supervised target for this topology: action-only MLPs reach
high train R² and near-zero/negative test R². Raw Δφ is memorisable but
also fails same-topology interpolation under the current electrical box.

D3/D5 gates for Reflection_Type therefore treat phase_err as a known-hard
target and require strong R² on IL / RL / gain (and optionally raw Δφ
diagnostics), not on rms_phase_err alone.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from env.graph_utils import TOPOLOGY_PARAMS
from sim.mna_scorer import mna_evaluate, solve_sparams, sparams_to_metrics
from specset.schema import TRAIN_SPECSET_PATH
from train_diffusion import action_to_params


def _wrap(d):
    return ((d + 180.0) % 360.0) - 180.0


def _spec():
    with open(TRAIN_SPECSET_PATH) as fh:
        doc = json.load(fh)
    spec = dict(doc["specs"][0]["spec"])
    spec["fc_ghz"] = 28.0
    return spec


def test_reflection_phase_err_has_variance_but_l_quarter_is_discontinuous():
    spec = _spec()
    keys = TOPOLOGY_PARAMS["Reflection_Type"]
    mid = np.full(len(keys), 0.5)
    p0 = action_to_params(mid, "Reflection_Type", spec, bounds="electrical")
    _, a0 = mna_evaluate("Reflection_Type", p0, spec)
    assert a0 is not None
    # Shortening L_quarter by 0.15 in action space jumps phase error sharply.
    a = mid.copy()
    a[keys.index("L_quarter_mm")] = 0.35
    p1 = action_to_params(a, "Reflection_Type", spec, bounds="electrical")
    _, a1 = mna_evaluate("Reflection_Type", p1, spec)
    assert abs(a1["rms_phase_err_deg"] - a0["rms_phase_err_deg"]) > 20.0


def test_reflection_raw_dphi_spans_wide_range():
    spec = _spec()
    keys = TOPOLOGY_PARAMS["Reflection_Type"]
    rng = np.random.default_rng(0)
    dphis = []
    for _ in range(80):
        act = rng.random(len(keys))
        p = action_to_params(act, "Reflection_Type", spec, bounds="electrical")
        m0 = sparams_to_metrics(*solve_sparams("Reflection_Type", p, 28.0, 0))
        m1 = sparams_to_metrics(*solve_sparams("Reflection_Type", p, 28.0, 1))
        dphis.append(_wrap(m1["phase_deg"] - m0["phase_deg"]))
    arr = np.asarray(dphis)
    assert arr.std() > 10.0
    assert arr.max() - arr.min() > 90.0


def test_reflection_il_is_a_reasonable_supervised_target():
    """Smoke: IL varies smoothly enough that a tiny MLP can fit train data."""
    import torch
    import torch.nn as nn

    spec = _spec()
    keys = TOPOLOGY_PARAMS["Reflection_Type"]
    rng = np.random.default_rng(1)
    X, y = [], []
    for _ in range(300):
        a = rng.random(len(keys))
        p = action_to_params(a, "Reflection_Type", spec, bounds="electrical")
        _, agg = mna_evaluate("Reflection_Type", p, spec)
        if agg is None:
            continue
        X.append(a)
        y.append(agg["il_db"])
    X = torch.tensor(np.asarray(X), dtype=torch.float)
    y = torch.tensor(np.asarray(y), dtype=torch.float)
    net = nn.Sequential(
        nn.Linear(5, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 1),
    )
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    for _ in range(1500):
        loss = ((net(X).squeeze(-1) - y) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        pred = net(X).squeeze(-1)
        ss_res = float(((y - pred) ** 2).sum())
        ss_tot = float(((y - y.mean()) ** 2).sum())
        r2 = 1.0 - ss_res / ss_tot
    assert r2 > 0.7, r2
