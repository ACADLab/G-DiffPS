"""D2 supervised sanity: memorization and same-topology interpolation.

Predict raw MNA metrics (and aggregate reward) from (graph, sizing).

    python tools/supervised_sanity.py --mode overfit --out results/graphs/d2_overfit.json
    python tools/supervised_sanity.py --mode interp --out results/graphs/d2_interp.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from env.graph_utils import SLOT_ACTION_DIM, TOPOLOGY_PARAMS
from env.netlist_graph import build_circuit_graph, params_numeric
from models.circuit_encoder import make_encoder
from sim.mna_scorer import mna_evaluate
from specset.schema import SPEC_DIM, TRAIN_SPECSET_PATH, normalize_spec
from train_diffusion import action_to_params

METRIC_KEYS = ["il_db", "rl_db", "rms_phase_err_deg", "gain_err_db", "reward"]


def _fixed_spec():
    with open(TRAIN_SPECSET_PATH) as fh:
        doc = json.load(fh)
    spec = dict(doc["specs"][0]["spec"])
    spec["fc_ghz"] = 28.0
    return spec


SPEC = _fixed_spec()


class MetricProbe(nn.Module):
    def __init__(self, in_dim, n_out=len(METRIC_KEYS), hidden=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_out),
        )

    def forward(self, x):
        return self.net(x)


def r2_np(y, yhat):
    y = np.asarray(y, dtype=np.float64)
    yhat = np.asarray(yhat, dtype=np.float64)
    ss_res = float(((y - yhat) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    if ss_tot < 1e-12:
        return 0.0
    return 1.0 - ss_res / ss_tot


def _pad_action(action, dim=SLOT_ACTION_DIM):
    out = np.full(dim, 0.5, dtype=np.float32)
    a = np.asarray(action, dtype=np.float32).reshape(-1)
    out[: min(len(a), dim)] = a[:dim]
    return out


def build_samples(topo, n, rng, spec=SPEC):
    keys = TOPOLOGY_PARAMS[topo]
    rows = []
    attempts = 0
    while len(rows) < n and attempts < n * 12:
        attempts += 1
        action = rng.random(len(keys))
        params = action_to_params(
            action, topo, spec, sizing="log", bounds="electrical",
            switch_model="ideal",
        )
        score, agg = mna_evaluate(topo, params, spec)
        if agg is None:
            continue
        rows.append({
            "action": _pad_action(action),
            "params": params_numeric(params),
            "y": np.array([
                float(agg.get("il_db", 0.0) or 0.0),
                float(agg.get("rl_db", 0.0) or 0.0),
                float(agg.get("rms_phase_err_deg", 0.0) or 0.0),
                float(agg.get("gain_err_db", 0.0) or 0.0),
                float(score),
            ], dtype=np.float32),
        })
    return rows


def _encode_batch(enc, topo, spec, rows, kind, max_dev=16):
    spec_t = torch.tensor(normalize_spec(spec), dtype=torch.float)
    zs, acts, raws = [], [], []
    with torch.no_grad():
        for row in rows:
            if kind == "nominal":
                z = enc(topo, spec).squeeze(0)
            elif kind == "actual":
                z = enc(topo, spec, params=row["params"]).squeeze(0)
            else:
                z = torch.zeros(64)
            zs.append(z)
            acts.append(torch.tensor(row["action"], dtype=torch.float))
            g = build_circuit_graph(
                topo, spec, state=0, typed=True, params=row["params"],
            )
            x = g["device"].x
            pad = torch.zeros(max_dev, x.size(-1))
            pad[: x.size(0)] = x
            raws.append(pad.reshape(-1))
    z = torch.stack(zs, dim=0)
    a = torch.stack(acts, dim=0)
    s = spec_t.unsqueeze(0).expand(len(rows), -1)
    raw = torch.stack(raws, dim=0)
    return z, s, a, raw


def fit_probe(x_train, y_train, x_test, y_test, epochs, lr, seed):
    torch.manual_seed(seed)
    probe = MetricProbe(x_train.size(1), y_train.size(1))
    opt = torch.optim.Adam(probe.parameters(), lr=lr)
    y_mean = y_train.mean(dim=0)
    y_std = y_train.std(dim=0).clamp_min(1e-3)
    for _ in range(epochs):
        probe.train()
        pred = probe(x_train)
        loss = ((pred - (y_train - y_mean) / y_std) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    probe.eval()
    with torch.no_grad():
        yhat = probe(x_test) * y_std + y_mean
    yhat_np = yhat.cpu().numpy()
    y_np = y_test.cpu().numpy()
    per = {METRIC_KEYS[i]: r2_np(y_np[:, i], yhat_np[:, i]) for i in range(len(METRIC_KEYS))}
    return per, float(((yhat - y_test) ** 2).mean())


def features(z, s, a, condition, raw=None):
    if condition == "action_only":
        return torch.cat([s, a], dim=-1)
    if condition == "z_nominal_action":
        return torch.cat([z, s, a], dim=-1)
    if condition == "z_actual":
        return torch.cat([z, s], dim=-1)
    if condition == "z_actual_action":
        return torch.cat([z, s, a], dim=-1)
    if condition == "raw_device_x":
        return torch.cat([raw, s], dim=-1)
    raise ValueError(condition)


def run(mode, topo, n, epochs, seed, out):
    rng = np.random.default_rng(seed)
    rows = build_samples(topo, n, rng)
    if len(rows) < max(20, n // 2):
        raise RuntimeError(f"only {len(rows)} MNA samples for {topo}")

    enc = make_encoder("circuit-typed")
    enc.eval()
    z_nom, s, a, _ = _encode_batch(enc, topo, SPEC, rows, "nominal")
    z_act, _, _, raw = _encode_batch(enc, topo, SPEC, rows, "actual")
    y = torch.stack([torch.tensor(r["y"]) for r in rows], dim=0)

    if mode == "overfit":
        tr, te = slice(None), slice(None)
    else:
        n_tr = int(0.8 * len(rows))
        tr, te = slice(0, n_tr), slice(n_tr, None)

    conditions = {
        "action_only": (None, s, a, None),
        "z_nominal_action": (z_nom, s, a, None),
        "z_actual": (z_act, s, a, None),
        "z_actual_action": (z_act, s, a, None),
        "raw_device_x": (None, s, a, raw),
    }
    report = {
        "mode": mode,
        "topology": topo,
        "n": len(rows),
        "epochs": epochs,
        "n_train": len(rows) if mode == "overfit" else int(0.8 * len(rows)),
        "conditions": {},
    }
    for name, (z, spec_t, act, raw_x) in conditions.items():
        x = features(
            z if z is not None else torch.zeros(len(rows), 0),
            spec_t, act, name, raw=raw_x,
        )
        per, mse = fit_probe(x[tr], y[tr], x[te], y[te], epochs, 1e-3, seed)
        report["conditions"][name] = {"r2": per, "mse": mse}
        print(name, json.dumps(per))
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w") as fh:
        json.dump(report, fh, indent=2)
    print("wrote", out)
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["overfit", "interp"], default="overfit")
    ap.add_argument("--topology", default="Loaded_Line")
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--epochs", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(
        REPO_ROOT, "results", "graphs", "d2_overfit.json",
    ))
    args = ap.parse_args()
    run(args.mode, args.topology, args.n, args.epochs, args.seed, args.out)


if __name__ == "__main__":
    main()
