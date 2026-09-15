"""D5: corrected supervised LOOCV (run only after D3/D4 pass).

Train on five topologies; evaluate raw-metric R² and sensitivity sign accuracy
on the held-out topology. Compare gin / circuit-typed / d1a / r3.

    python tools/d5_supervised_loocv.py --n 200 --epochs 250 --seeds 0,1,2 \\
        --out results/graphs/d5_supervised_loocv.json
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

from env.graph_utils import SLOT_ACTION_DIM, TOPOLOGY_PARAMS, get_topology_graph
from env.netlist_graph import (
    N_STATES, build_circuit_graph, device_names, params_numeric, sized_devices,
)
from env.param_semantics import PARAM_CONTEXT_DIM, param_context_matrix
from models.circuit_encoder import make_encoder
from models.gnn_encoder import TopologyEncoder
from sim.mna_scorer import mna_evaluate, solve_sparams, sparams_to_metrics
from specset.schema import SPEC_DIM, TRAIN_SPECSET_PATH, normalize_spec
from train_diffusion import action_to_params

METRIC_KEYS = ["il_db", "rl_db", "rms_phase_err_deg", "gain_err_db", "raw_dphi"]
GATE_METRICS = {
    "Loaded_Line": ["il_db", "rl_db", "rms_phase_err_deg", "gain_err_db"],
    "Switched_Line": ["il_db", "rl_db", "rms_phase_err_deg", "gain_err_db"],
    "Reflection_Type": ["il_db", "rl_db", "gain_err_db", "raw_dphi"],
    "Switched_Filter": ["il_db", "rl_db", "rms_phase_err_deg", "gain_err_db"],
    "Vector_Modulator": ["il_db", "rl_db", "rms_phase_err_deg", "gain_err_db"],
    "All_Pass": ["il_db", "rl_db", "rms_phase_err_deg", "gain_err_db"],
}
MAX_PARAM_TOKENS = 8
DELTA = 0.08


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


def sign_acc(y, yhat, eps=1e-6):
    y = np.asarray(y); yhat = np.asarray(yhat)
    mask = np.abs(y) > eps
    if mask.sum() == 0:
        return 1.0
    return float((np.sign(y[mask]) == np.sign(yhat[mask])).mean())


def _states(topo):
    n = N_STATES[topo]
    if n <= 2:
        return list(range(n))
    return sorted({0, 1, n // 2})


def _metrics(topo, params):
    score, agg = mna_evaluate(topo, params, SPEC)
    if agg is None:
        return None
    try:
        m0 = sparams_to_metrics(*solve_sparams(topo, params, SPEC["fc_ghz"], 0))
        m1 = sparams_to_metrics(*solve_sparams(topo, params, SPEC["fc_ghz"], 1))
        dphi = _wrap(m1["phase_deg"] - m0["phase_deg"])
    except Exception:
        dphi = 0.0
    return np.array([
        float(agg.get("il_db", 0.0) or 0.0),
        float(agg.get("rl_db", 0.0) or 0.0),
        float(agg.get("rms_phase_err_deg", 0.0) or 0.0),
        float(agg.get("gain_err_db", 0.0) or 0.0),
        float(dphi),
    ], dtype=np.float32)


def build_abs_samples(topo, n, rng):
    keys = TOPOLOGY_PARAMS[topo]
    rows = []
    attempts = 0
    while len(rows) < n and attempts < n * 12:
        attempts += 1
        action = rng.random(len(keys)).astype(np.float32)
        params = action_to_params(
            action, topo, SPEC, sizing="log", bounds="electrical", switch_model="ideal",
        )
        m = _metrics(topo, params)
        if m is None:
            continue
        numeric = params_numeric(params)
        graphs = [
            build_circuit_graph(
                topo, SPEC, state=s, typed=True, params=numeric, param_nodes=True,
            )
            for s in _states(topo)
        ]
        rows.append({
            "topo": topo,
            "action_len": len(keys),
            "graphs": graphs,
            "context": param_context_matrix(topo, SPEC, params=numeric),
            "y": m,
        })
    return rows


def build_sens_samples(topo, n, rng, delta=DELTA):
    keys = TOPOLOGY_PARAMS[topo]
    n_act = len(keys)
    rows = []
    attempts = 0
    while len(rows) < n and attempts < n * 20:
        attempts += 1
        base = np.clip(rng.random(n_act).astype(np.float32), delta + 0.02, 1 - delta - 0.02)
        params0 = action_to_params(
            base, topo, SPEC, sizing="log", bounds="electrical", switch_model="ideal",
        )
        m0 = _metrics(topo, params0)
        if m0 is None:
            continue
        i = int(rng.integers(0, n_act))
        a_pos = base.copy(); a_pos[i] = float(np.clip(base[i] + delta, 0, 1))
        mp = _metrics(topo, action_to_params(
            a_pos, topo, SPEC, sizing="log", bounds="electrical", switch_model="ideal",
        ))
        if mp is None:
            continue
        numeric = params_numeric(params0)
        graphs = [
            build_circuit_graph(
                topo, SPEC, state=s, typed=True, params=numeric, param_nodes=True,
            )
            for s in _states(topo)
        ]
        knob = np.zeros(SLOT_ACTION_DIM, dtype=np.float32); knob[i] = 1.0
        rows.append({
            "topo": topo,
            "graphs": graphs,
            "context": param_context_matrix(topo, SPEC, params=numeric),
            "knob": knob,
            "dm": (mp - m0).astype(np.float32),
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


class SharedModel(nn.Module):
    """Topology-shared encoder+head for LOOCV transfer."""

    def __init__(self, variant: str, mode: str = "abs"):
        super().__init__()
        self.variant = variant
        self.mode = mode  # abs | sens
        knob_dim = SLOT_ACTION_DIM if mode == "sens" else 0
        if variant == "gin":
            self.enc = TopologyEncoder()
            self._gin_cache = {}
            in_dim = 64 + SPEC_DIM + knob_dim
        elif variant == "circuit-typed":
            self.enc = make_encoder("circuit-typed")
            in_dim = 64 + SPEC_DIM + knob_dim
        elif variant == "d1a":
            self.enc = make_encoder("circuit-typed")
            self.token_proj = nn.Sequential(
                nn.Linear(64 + PARAM_CONTEXT_DIM, 64), nn.ReLU(),
            )
            in_dim = MAX_PARAM_TOKENS * 64 + SPEC_DIM + knob_dim
        elif variant == "r3":
            self.enc = make_encoder("circuit-typed-param")
            self.token_proj = nn.Sequential(nn.Linear(64, 64), nn.ReLU())
            in_dim = MAX_PARAM_TOKENS * 64 + SPEC_DIM + knob_dim
        else:
            raise ValueError(variant)
        self.head = Head(in_dim)

    def _encode_typed(self, graphs):
        enc = self.enc
        h_list, p_list = [], []
        for g in graphs:
            if self.variant == "r3":
                _, h_d, h_p = enc.encode_state(g)
                h_list.append(h_d); p_list.append(h_p)
            else:
                _, h_d = enc.encode_state(g)
                h_list.append(h_d)
        H = torch.stack(h_list, dim=0)
        h_mean = H.mean(dim=0)
        if H.size(0) >= 2:
            iu = torch.triu_indices(H.size(0), H.size(0), offset=1, device=H.device)
            h_contrast = (H[iu[0]] - H[iu[1]]).abs().mean(dim=0)
        else:
            h_contrast = torch.zeros_like(h_mean)
        z = enc.proj(torch.cat([
            h_mean.sum(dim=0, keepdim=True),
            h_contrast.sum(dim=0, keepdim=True),
        ], dim=-1)).squeeze(0)
        h_dev = enc.dev_proj(torch.cat([h_mean, h_contrast], dim=-1))
        h_param = None
        if p_list:
            P = torch.stack(p_list, dim=0)
            p_mean = P.mean(dim=0)
            if P.size(0) >= 2:
                iu = torch.triu_indices(P.size(0), P.size(0), offset=1, device=P.device)
                p_contrast = (P[iu[0]] - P[iu[1]]).abs().mean(dim=0)
            else:
                p_contrast = torch.zeros_like(p_mean)
            h_param = enc.param_proj(torch.cat([p_mean, p_contrast], dim=-1))
        return z, h_dev, h_param

    def forward_batch(self, rows):
        device = next(self.parameters()).device
        B = len(rows)
        spec_t = torch.tensor(
            np.stack([normalize_spec(SPEC) for _ in rows]), dtype=torch.float, device=device,
        )
        knobs = None
        if self.mode == "sens":
            knobs = torch.tensor(
                np.stack([r["knob"] for r in rows]), dtype=torch.float, device=device,
            )

        if self.variant == "gin":
            zs = []
            for r in rows:
                topo = r["topo"]
                if topo not in self._gin_cache:
                    self._gin_cache[topo] = get_topology_graph(topo)
                g = self._gin_cache[topo]
                z = self.enc(g.x.to(device), g.edge_index.to(device))
                if z.dim() > 1:
                    z = z.squeeze(0)
                zs.append(z)
            h = torch.stack(zs, dim=0)
            parts = [h, spec_t]
            if knobs is not None:
                parts.append(knobs)
            return self.head(torch.cat(parts, dim=-1))

        feats = []
        for r in rows:
            graphs = [g.to(device) for g in r["graphs"]]
            z, h_dev, h_param = self._encode_typed(graphs)
            topo = r["topo"]
            if self.variant == "circuit-typed":
                feats.append(z)
            elif self.variant == "d1a":
                sized = sized_devices(topo)
                name_to_idx = {n: i for i, n in enumerate(device_names(topo))}
                ctx = torch.tensor(r["context"], dtype=torch.float, device=device)
                toks = [
                    torch.cat([h_dev[name_to_idx[d]], ctx[j]], dim=-1)
                    for j, (d, _) in enumerate(sized)
                ]
                tok = self.token_proj(torch.stack(toks, dim=0))
                pad = torch.zeros(MAX_PARAM_TOKENS, 64, device=device)
                pad[: min(tok.size(0), MAX_PARAM_TOKENS)] = tok[:MAX_PARAM_TOKENS]
                feats.append(pad.reshape(-1))
            else:
                tok = self.token_proj(h_param)
                pad = torch.zeros(MAX_PARAM_TOKENS, 64, device=device)
                pad[: min(tok.size(0), MAX_PARAM_TOKENS)] = tok[:MAX_PARAM_TOKENS]
                feats.append(pad.reshape(-1))
        h = torch.stack(feats, dim=0)
        parts = [h, spec_t]
        if knobs is not None:
            parts.append(knobs)
        return self.head(torch.cat(parts, dim=-1))


def _train_eval(model, train_rows, test_rows, y_key, epochs, seed, batch_size=64):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    y_tr = torch.tensor(np.stack([r[y_key] for r in train_rows]), dtype=torch.float)
    y_mean, y_std = y_tr.mean(0), y_tr.std(0).clamp_min(1e-4)
    model.train()
    for _ in range(epochs):
        order = rng.permutation(len(train_rows))
        for start in range(0, len(train_rows), batch_size):
            batch = [train_rows[i] for i in order[start:start + batch_size]]
            y = torch.tensor(np.stack([r[y_key] for r in batch]), dtype=torch.float)
            pred = model.forward_batch(batch)
            loss = ((pred - (y - y_mean) / y_std) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        pred = model.forward_batch(test_rows)
        yhat = (pred * y_std + y_mean).numpy()
        y = np.stack([r[y_key] for r in test_rows])
    per_r2 = {METRIC_KEYS[i]: r2_np(y[:, i], yhat[:, i]) for i in range(len(METRIC_KEYS))}
    per_sign = {METRIC_KEYS[i]: sign_acc(y[:, i], yhat[:, i]) for i in range(len(METRIC_KEYS))}
    err = yhat - y
    return {
        "test_r2": per_r2,
        "test_sign_acc": per_sign,
        "err_mean": {METRIC_KEYS[i]: float(err[:, i].mean()) for i in range(len(METRIC_KEYS))},
        "err_std": {METRIC_KEYS[i]: float(err[:, i].std()) for i in range(len(METRIC_KEYS))},
    }


def run_fold(holdout, variant, abs_by_topo, sens_by_topo, epochs, seed):
    train_abs = [r for t, rows in abs_by_topo.items() if t != holdout for r in rows]
    test_abs = abs_by_topo[holdout]
    train_sens = [r for t, rows in sens_by_topo.items() if t != holdout for r in rows]
    test_sens = sens_by_topo[holdout]

    abs_model = SharedModel(variant, mode="abs")
    abs_rec = _train_eval(abs_model, train_abs, test_abs, "y", epochs, seed)
    sens_model = SharedModel(variant, mode="sens")
    sens_rec = _train_eval(sens_model, train_sens, test_sens, "dm", epochs, seed + 17)

    gate_keys = GATE_METRICS[holdout]
    return {
        "holdout": holdout,
        "variant": variant,
        "seed": seed,
        "n_train_abs": len(train_abs),
        "n_test_abs": len(test_abs),
        "absolute": abs_rec,
        "sensitivity": sens_rec,
        "abs_gate_ok": all(abs_rec["test_r2"][k] > 0.3 for k in gate_keys),
        "sens_gate_ok": all(sens_rec["test_sign_acc"][k] >= 0.55 for k in gate_keys),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200, help="samples per topology")
    ap.add_argument("--n-sens", type=int, default=150)
    ap.add_argument("--epochs", type=int, default=250)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--variants", default="gin,circuit-typed,d1a,r3")
    ap.add_argument("--holdouts", default=",".join(TOPOLOGY_PARAMS))
    ap.add_argument("--out", default=os.path.join(
        REPO_ROOT, "results", "graphs", "d5_supervised_loocv.json",
    ))
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    holdouts = [h.strip() for h in args.holdouts.split(",")]
    variants = [v.strip() for v in args.variants.split(",")]

    # Build once per topology (shared across folds/seeds of same n).
    abs_by_topo, sens_by_topo = {}, {}
    for topo in TOPOLOGY_PARAMS:
        rng = np.random.default_rng(1000 + (hash(topo) % 10000))
        print(f"building {topo}...", flush=True)
        abs_by_topo[topo] = build_abs_samples(topo, args.n, rng)
        sens_by_topo[topo] = build_sens_samples(topo, args.n_sens, rng)

    report = {
        "n": args.n, "n_sens": args.n_sens, "epochs": args.epochs,
        "seeds": seeds, "folds": [], "macro": {},
    }
    for holdout in holdouts:
        for variant in variants:
            for seed in seeds:
                print(f"\nfold holdout={holdout} variant={variant} seed={seed}", flush=True)
                t0 = time.time()
                rec = run_fold(
                    holdout, variant, abs_by_topo, sens_by_topo, args.epochs, seed,
                )
                rec["wall_seconds"] = time.time() - t0
                report["folds"].append(rec)
                print(
                    f"  abs_gate={'PASS' if rec['abs_gate_ok'] else 'FAIL'} "
                    f"sens_gate={'PASS' if rec['sens_gate_ok'] else 'FAIL'} "
                    f"IL_R2={rec['absolute']['test_r2']['il_db']:+.3f} "
                    f"sign_IL={rec['sensitivity']['test_sign_acc']['il_db']:.3f}",
                    flush=True,
                )

    for variant in variants:
        folds = [f for f in report["folds"] if f["variant"] == variant]
        report["macro"][variant] = {
            "abs_pass": int(sum(f["abs_gate_ok"] for f in folds)),
            "sens_pass": int(sum(f["sens_gate_ok"] for f in folds)),
            "n": len(folds),
            "mean_il_r2": float(np.mean([f["absolute"]["test_r2"]["il_db"] for f in folds])),
            "mean_sign_il": float(np.mean([
                f["sensitivity"]["test_sign_acc"]["il_db"] for f in folds
            ])),
        }

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(report, fh, indent=2)
    print("wrote", args.out, flush=True)
    print("macro", json.dumps(report["macro"], indent=2), flush=True)


if __name__ == "__main__":
    main()
