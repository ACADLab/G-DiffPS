"""R3 ablations: drop one semantic channel at a time.

Ablations (applied to param context / param-node features):
  full          — role + value + bounds + log flag + coupling (+ RWSE on devices)
  no_role       — zero role one-hot
  no_value      — zero current-value channel
  no_bounds     — zero lo/hi channels
  no_log_flag   — zero is_log channel
  no_coupling   — zero coupling-group channel
  no_rwse       — typed encoder without positional encodings

Same D3 protocol: representation-only same-topology interpolation.

    python tools/r3_ablations.py --n 200 --epochs 250 \\
        --out results/graphs/r3_ablations.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from env.graph_utils import TOPOLOGY_PARAMS
from env.netlist_graph import (
    N_STATES, build_circuit_graph, params_numeric, sized_devices,
)
from env.param_semantics import (
    N_PARAM_ROLES, PARAM_AUX_DIM, PARAM_CONTEXT_DIM, PARAM_NODE_IN,
    param_context_matrix,
)
from models.circuit_encoder import make_encoder
from sim.mna_scorer import mna_evaluate, solve_sparams, sparams_to_metrics
from specset.schema import SPEC_DIM, TRAIN_SPECSET_PATH, normalize_spec
from train_diffusion import action_to_params

# Aux layout: [value_01, is_log, log10(lo), log10(hi), group_id]
_AUX_VALUE, _AUX_LOG, _AUX_LO, _AUX_HI, _AUX_GROUP = 0, 1, 2, 3, 4

METRIC_KEYS = ["il_db", "rl_db", "rms_phase_err_deg", "gain_err_db", "raw_dphi"]
GATE_METRICS = {
    "Loaded_Line": ["il_db", "rl_db", "rms_phase_err_deg", "gain_err_db"],
    "Switched_Line": ["il_db", "rl_db", "rms_phase_err_deg", "gain_err_db"],
    "Reflection_Type": ["il_db", "rl_db", "gain_err_db", "raw_dphi"],
    "Switched_Filter": ["il_db", "rl_db", "rms_phase_err_deg", "gain_err_db"],
    "Vector_Modulator": ["il_db", "rl_db", "rms_phase_err_deg", "gain_err_db"],
    "All_Pass": ["il_db", "rl_db", "rms_phase_err_deg", "gain_err_db"],
}
GATE_THRESH = 0.5
MAX_PARAM_TOKENS = 8
ABLATIONS = (
    "full", "no_role", "no_value", "no_bounds", "no_log_flag", "no_coupling", "no_rwse",
)


def _wrap(d):
    return ((d + 180.0) % 360.0) - 180.0


def _fixed_spec():
    with open(TRAIN_SPECSET_PATH) as fh:
        doc = json.load(fh)
    spec = dict(doc["specs"][0]["spec"])
    spec["fc_ghz"] = 28.0
    return spec


SPEC = _fixed_spec()


def r2_np(y, yhat):
    y = np.asarray(y, dtype=np.float64)
    yhat = np.asarray(yhat, dtype=np.float64)
    ss_res = float(((y - yhat) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    if ss_tot < 1e-12:
        return 0.0
    return 1.0 - ss_res / ss_tot


def _states(topo):
    n = N_STATES[topo]
    if n <= 2:
        return list(range(n))
    return sorted({0, 1, n // 2})


def mask_context(ctx: np.ndarray, ablation: str) -> np.ndarray:
    """Zero selected semantic channels in [N, PARAM_CONTEXT_DIM]."""
    out = ctx.copy()
    if ablation == "no_role":
        out[:, :N_PARAM_ROLES] = 0.0
    elif ablation == "no_value":
        out[:, N_PARAM_ROLES + _AUX_VALUE] = 0.0
    elif ablation == "no_bounds":
        out[:, N_PARAM_ROLES + _AUX_LO] = 0.0
        out[:, N_PARAM_ROLES + _AUX_HI] = 0.0
    elif ablation == "no_log_flag":
        out[:, N_PARAM_ROLES + _AUX_LOG] = 0.0
    elif ablation == "no_coupling":
        out[:, N_PARAM_ROLES + _AUX_GROUP] = 0.0
    return out


def mask_param_x(x: torch.Tensor, ablation: str) -> torch.Tensor:
    """Param node features = context + mutable flag."""
    out = x.clone()
    if ablation == "no_role":
        out[:, :N_PARAM_ROLES] = 0.0
    elif ablation == "no_value":
        out[:, N_PARAM_ROLES + _AUX_VALUE] = 0.0
    elif ablation == "no_bounds":
        out[:, N_PARAM_ROLES + _AUX_LO] = 0.0
        out[:, N_PARAM_ROLES + _AUX_HI] = 0.0
    elif ablation == "no_log_flag":
        out[:, N_PARAM_ROLES + _AUX_LOG] = 0.0
    elif ablation == "no_coupling":
        out[:, N_PARAM_ROLES + _AUX_GROUP] = 0.0
    return out


def build_samples(topo, n, rng):
    keys = TOPOLOGY_PARAMS[topo]
    rows = []
    attempts = 0
    while len(rows) < n and attempts < n * 12:
        attempts += 1
        action = rng.random(len(keys))
        params = action_to_params(
            action, topo, SPEC, sizing="log", bounds="electrical", switch_model="ideal",
        )
        score, agg = mna_evaluate(topo, params, SPEC)
        if agg is None:
            continue
        try:
            m0 = sparams_to_metrics(*solve_sparams(topo, params, SPEC["fc_ghz"], 0))
            m1 = sparams_to_metrics(*solve_sparams(topo, params, SPEC["fc_ghz"], 1))
            dphi = _wrap(m1["phase_deg"] - m0["phase_deg"])
        except Exception:
            dphi = 0.0
        numeric = params_numeric(params)
        states = _states(topo)
        graphs = [
            build_circuit_graph(
                topo, SPEC, state=s, typed=True, params=numeric, param_nodes=True,
            )
            for s in states
        ]
        rows.append({
            "params": numeric,
            "graphs": graphs,
            "context": param_context_matrix(topo, SPEC, params=numeric),
            "y": np.array([
                float(agg.get("il_db", 0.0) or 0.0),
                float(agg.get("rl_db", 0.0) or 0.0),
                float(agg.get("rms_phase_err_deg", 0.0) or 0.0),
                float(agg.get("gain_err_db", 0.0) or 0.0),
                float(dphi),
            ], dtype=np.float32),
        })
    return rows


class Head(nn.Module):
    def __init__(self, in_dim, n_out=len(METRIC_KEYS), hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_out),
        )

    def forward(self, x):
        return self.net(x)


class AblationModel(nn.Module):
    def __init__(self, ablation: str):
        super().__init__()
        self.ablation = ablation
        from models.circuit_encoder import CircuitParamEncoder
        self.enc = CircuitParamEncoder(use_pe=(ablation != "no_rwse"))
        self.token_proj = nn.Sequential(nn.Linear(64, 64), nn.ReLU())
        self.head = Head(MAX_PARAM_TOKENS * 64 + SPEC_DIM)

    def forward_batch(self, rows):
        device = next(self.parameters()).device
        spec_t = torch.tensor(
            np.stack([normalize_spec(SPEC) for _ in rows]), dtype=torch.float, device=device,
        )
        feats = []
        for r in rows:
            h_list, p_list = [], []
            for g0 in r["graphs"]:
                g = g0.clone().to(device)
                if "param" in g.node_types:
                    g["param"].x = mask_param_x(g["param"].x, self.ablation)
                _, h_d, h_p = self.enc.encode_state(g)
                h_list.append(h_d); p_list.append(h_p)
            P = torch.stack(p_list, dim=0)
            p_mean = P.mean(dim=0)
            if P.size(0) >= 2:
                iu = torch.triu_indices(P.size(0), P.size(0), offset=1, device=device)
                p_contrast = (P[iu[0]] - P[iu[1]]).abs().mean(dim=0)
            else:
                p_contrast = torch.zeros_like(p_mean)
            h_param = self.enc.param_proj(torch.cat([p_mean, p_contrast], dim=-1))
            tok = self.token_proj(h_param)
            pad = torch.zeros(MAX_PARAM_TOKENS, 64, device=device)
            pad[: min(tok.size(0), MAX_PARAM_TOKENS)] = tok[:MAX_PARAM_TOKENS]
            feats.append(pad.reshape(-1))
        return self.head(torch.cat([torch.stack(feats, dim=0), spec_t], dim=-1))


def train_ablation(ablation, topo, rows, epochs, seed, batch_size=64):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    n_tr = int(0.8 * len(rows))
    idx = rng.permutation(len(rows))
    train_rows = [rows[i] for i in idx[:n_tr]]
    test_rows = [rows[i] for i in idx[n_tr:]]
    model = AblationModel(ablation)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    y_tr = torch.tensor(np.stack([r["y"] for r in train_rows]), dtype=torch.float)
    y_mean, y_std = y_tr.mean(0), y_tr.std(0).clamp_min(1e-3)
    t0 = time.time()
    model.train()
    for _ in range(epochs):
        order = rng.permutation(len(train_rows))
        for start in range(0, len(train_rows), batch_size):
            batch = [train_rows[i] for i in order[start:start + batch_size]]
            y = torch.tensor(np.stack([r["y"] for r in batch]), dtype=torch.float)
            pred = model.forward_batch(batch)
            loss = ((pred - (y - y_mean) / y_std) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    train_s = time.time() - t0
    model.eval()
    with torch.no_grad():
        pred = model.forward_batch(test_rows)
        yhat = (pred * y_std + y_mean).numpy()
        y = np.stack([r["y"] for r in test_rows])
    per = {METRIC_KEYS[i]: r2_np(y[:, i], yhat[:, i]) for i in range(len(METRIC_KEYS))}
    gate_keys = GATE_METRICS[topo]
    return {
        "n_train": len(train_rows), "n_test": len(test_rows),
        "train_seconds": train_s, "test_r2": per,
        "gate_metrics": gate_keys,
        "gate_ok": all(per[k] > GATE_THRESH for k in gate_keys),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--epochs", type=int, default=250)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--topologies", default="Loaded_Line,All_Pass")
    ap.add_argument("--ablations", default=",".join(ABLATIONS))
    ap.add_argument("--out", default=os.path.join(
        REPO_ROOT, "results", "graphs", "r3_ablations.json",
    ))
    args = ap.parse_args()

    report = {"n": args.n, "epochs": args.epochs, "seed": args.seed, "topologies": {}}
    for topo in args.topologies.split(","):
        topo = topo.strip()
        print(f"\n=== {topo} ===", flush=True)
        rng = np.random.default_rng(args.seed + (hash(topo) % 10000))
        rows = build_samples(topo, args.n, rng)
        print(f"  built {len(rows)} samples", flush=True)
        report["topologies"][topo] = {"ablations": {}}
        for abl in args.ablations.split(","):
            abl = abl.strip()
            print(f"  training {abl}...", flush=True)
            rec = train_ablation(abl, topo, rows, args.epochs, args.seed)
            report["topologies"][topo]["ablations"][abl] = rec
            tr = rec["test_r2"]
            print(
                f"    gate={'PASS' if rec['gate_ok'] else 'FAIL'}  "
                f"IL={tr['il_db']:+.3f} RL={tr['rl_db']:+.3f} ph={tr['rms_phase_err_deg']:+.3f}",
                flush=True,
            )
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(report, fh, indent=2)
    print("wrote", args.out, flush=True)


if __name__ == "__main__":
    main()
