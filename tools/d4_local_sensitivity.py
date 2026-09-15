"""D4v2: local sensitivity with token-indexed readout.

Scientific question: does an explicit parameter token predict the effect of
perturbing *that* parameter better than a pooled graph + knob one-hot?

Variants:
  action       — action + knob one-hot (ceiling)
  gin          — pooled z + knob one-hot
  circuit-typed— pooled z + knob one-hot
  d1a          — indexed (h_dev ∥ context) token for the perturbed param
  r3           — indexed h_param token for the perturbed param

Loss: sign BCE (primary) + small MSE on Δm.
Gate: mean sign accuracy on gate metrics ≥ 0.65.

    python tools/d4_local_sensitivity.py --n 200 --epochs 250 \\
        --out results/graphs/d4_local_sensitivity.json
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
import torch.nn.functional as F

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from env.graph_utils import SLOT_ACTION_DIM, TOPOLOGY_PARAMS, get_topology_graph
from env.netlist_graph import (
    N_STATES, build_circuit_graph, device_names, params_numeric, sized_devices,
)
from env.param_semantics import PARAM_CONTEXT_DIM, param_context_matrix, param_role_onehot
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
SIGN_ACC_GATE = 0.65
DELTA = 0.08
N_ROLES = 10  # matches PARAM_ROLES length used as one-hot width


def _wrap(d):
    return ((d + 180.0) % 360.0) - 180.0


def _fixed_spec():
    with open(TRAIN_SPECSET_PATH) as fh:
        doc = json.load(fh)
    spec = dict(doc["specs"][0]["spec"])
    spec["fc_ghz"] = 28.0
    return spec


SPEC = _fixed_spec()


def _pad_action(action, dim=SLOT_ACTION_DIM):
    out = np.full(dim, 0.5, dtype=np.float32)
    a = np.asarray(action, dtype=np.float32).reshape(-1)
    out[: min(len(a), dim)] = a[:dim]
    return out


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


def _decode(action, topo):
    return action_to_params(
        action, topo, SPEC, sizing="log", bounds="electrical", switch_model="ideal",
    )


def build_sensitivity_samples(topo, n, rng, delta=DELTA):
    keys = TOPOLOGY_PARAMS[topo]
    n_act = len(keys)
    sized = sized_devices(topo)
    # Map action key → sized-device index (param token row).
    key_to_sized = {p: i for i, (_, p) in enumerate(sized)}
    rows = []
    attempts = 0
    while len(rows) < n and attempts < n * 20:
        attempts += 1
        base = rng.random(n_act).astype(np.float32)
        base = np.clip(base, delta + 0.02, 1.0 - delta - 0.02)
        params0 = _decode(base, topo)
        m0 = _metrics(topo, params0)
        if m0 is None:
            continue
        i = int(rng.integers(0, n_act))
        key = keys[i]
        if key not in key_to_sized:
            continue  # skip action dims without a sized token
        a_pos = base.copy(); a_pos[i] = float(np.clip(base[i] + delta, 0, 1))
        mp = _metrics(topo, _decode(a_pos, topo))
        if mp is None:
            continue
        dm = mp - m0
        numeric = params_numeric(params0)
        states = _states(topo)
        graphs = [
            build_circuit_graph(
                topo, SPEC, state=s, typed=True, params=numeric, param_nodes=True,
            )
            for s in states
        ]
        knob_onehot = np.zeros(SLOT_ACTION_DIM, dtype=np.float32)
        knob_onehot[i] = 1.0
        role = param_role_onehot(key)
        rows.append({
            "action": _pad_action(base),
            "knob": knob_onehot,
            "knob_idx": i,
            "token_idx": int(key_to_sized[key]),
            "role": role.astype(np.float32),
            "params": numeric,
            "graphs": graphs,
            "context": param_context_matrix(topo, SPEC, params=numeric),
            "dm": dm.astype(np.float32),
            "sign": np.sign(dm).astype(np.float32),
        })
    return rows


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


class SensModel(nn.Module):
    def __init__(self, variant, topo):
        super().__init__()
        self.variant = variant
        self.topo = topo
        self.sized = sized_devices(topo)
        self.dnames = device_names(topo)
        self.name_to_idx = {n: i for i, n in enumerate(self.dnames)}

        if variant == "action":
            self.enc = None
            in_dim = SPEC_DIM + SLOT_ACTION_DIM + SLOT_ACTION_DIM
        elif variant == "gin":
            self.enc = TopologyEncoder()
            self.gin_graph = get_topology_graph(topo)
            in_dim = 64 + SPEC_DIM + SLOT_ACTION_DIM
        elif variant == "circuit-typed":
            self.enc = make_encoder("circuit-typed")
            in_dim = 64 + SPEC_DIM + SLOT_ACTION_DIM
        elif variant == "d1a":
            self.enc = make_encoder("circuit-typed")
            self.token_proj = nn.Sequential(
                nn.Linear(64 + PARAM_CONTEXT_DIM, 64), nn.ReLU(),
            )
            # Indexed token only — no knob one-hot (identity is the index).
            in_dim = 64 + SPEC_DIM
        elif variant == "r3":
            self.enc = make_encoder("circuit-typed-param")
            self.token_proj = nn.Sequential(nn.Linear(64, 64), nn.ReLU())
            in_dim = 64 + SPEC_DIM
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
        B = len(rows)
        device = next(self.parameters()).device
        acts = torch.tensor(np.stack([r["action"] for r in rows]), dtype=torch.float, device=device)
        knobs = torch.tensor(np.stack([r["knob"] for r in rows]), dtype=torch.float, device=device)
        spec_t = torch.tensor(
            np.stack([normalize_spec(SPEC) for _ in rows]), dtype=torch.float, device=device,
        )

        if self.variant == "action":
            return self.head(torch.cat([spec_t, acts, knobs], dim=-1))

        if self.variant == "gin":
            g = self.gin_graph
            z = self.enc(g.x.to(device), g.edge_index.to(device))
            if z.dim() == 1:
                z = z.unsqueeze(0)
            z = z.expand(B, -1).contiguous()
            return self.head(torch.cat([z, spec_t, knobs], dim=-1))

        feats = []
        for r in rows:
            graphs = [g.to(device) for g in r["graphs"]]
            z, h_dev, h_param = self._encode_typed(graphs)
            if self.variant == "circuit-typed":
                feats.append(z)
            elif self.variant == "d1a":
                ctx = torch.tensor(r["context"], dtype=torch.float, device=device)
                ti = int(r["token_idx"])
                dname = self.sized[ti][0]
                tok = torch.cat([h_dev[self.name_to_idx[dname]], ctx[ti]], dim=-1)
                feats.append(self.token_proj(tok))
            else:  # r3 — index the perturbed param token
                ti = int(r["token_idx"])
                feats.append(self.token_proj(h_param[ti]))
        h = torch.stack(feats, dim=0)
        if self.variant == "circuit-typed":
            return self.head(torch.cat([h, spec_t, knobs], dim=-1))
        return self.head(torch.cat([h, spec_t], dim=-1))


def train_variant(variant, topo, rows, epochs, seed, batch_size=64):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    n_tr = int(0.8 * len(rows))
    idx = rng.permutation(len(rows))
    train_rows = [rows[i] for i in idx[:n_tr]]
    test_rows = [rows[i] for i in idx[n_tr:]]

    model = SensModel(variant, topo)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    y_tr = torch.tensor(np.stack([r["dm"] for r in train_rows]), dtype=torch.float)
    y_mean, y_std = y_tr.mean(0), y_tr.std(0).clamp_min(1e-4)

    t0 = time.time()
    model.train()
    for _ in range(epochs):
        order = rng.permutation(len(train_rows))
        for start in range(0, len(train_rows), batch_size):
            batch = [train_rows[i] for i in order[start:start + batch_size]]
            y = torch.tensor(np.stack([r["dm"] for r in batch]), dtype=torch.float)
            pred = model.forward_batch(batch)
            target = (y - y_mean) / y_std
            mse = ((pred - target) ** 2).mean()
            # Sign BCE on non-near-zero labels
            sign_t = torch.sign(y)
            mask = (y.abs() > 1e-6).float()
            bce = F.binary_cross_entropy_with_logits(
                pred, (sign_t > 0).float(), weight=mask, reduction="sum",
            ) / mask.sum().clamp_min(1.0)
            loss = bce + 0.25 * mse
            opt.zero_grad(); loss.backward(); opt.step()
    train_s = time.time() - t0

    model.eval()
    with torch.no_grad():
        pred = model.forward_batch(test_rows)
        yhat = (pred * y_std + y_mean).numpy()
        y = np.stack([r["dm"] for r in test_rows])
    per_r2 = {METRIC_KEYS[i]: r2_np(y[:, i], yhat[:, i]) for i in range(len(METRIC_KEYS))}
    per_sign = {METRIC_KEYS[i]: sign_acc(y[:, i], yhat[:, i]) for i in range(len(METRIC_KEYS))}
    gate_keys = GATE_METRICS[topo]
    return {
        "n_train": len(train_rows),
        "n_test": len(test_rows),
        "train_seconds": train_s,
        "delta_action": DELTA,
        "protocol": "token_indexed_d1a_r3_sign_bce",
        "test_r2_dm": per_r2,
        "test_sign_acc": per_sign,
        "gate_metrics": gate_keys,
        "mean_sign_acc_gate": float(np.mean([per_sign[k] for k in gate_keys])),
        "gate_ok": all(per_sign[k] >= SIGN_ACC_GATE for k in gate_keys),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--epochs", type=int, default=250)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--topologies", default=",".join(TOPOLOGY_PARAMS))
    ap.add_argument("--variants", default="action,gin,circuit-typed,d1a,r3")
    ap.add_argument("--out", default=os.path.join(
        REPO_ROOT, "results", "graphs", "d4_local_sensitivity.json",
    ))
    args = ap.parse_args()

    report = {
        "n": args.n, "epochs": args.epochs, "seed": args.seed,
        "delta_action": DELTA, "sign_acc_gate": SIGN_ACC_GATE,
        "protocol": "token_indexed_d1a_r3_sign_bce",
        "topologies": {},
    }
    for topo in args.topologies.split(","):
        topo = topo.strip()
        print(f"\n=== {topo} ===", flush=True)
        rng = np.random.default_rng(args.seed + (hash(topo) % 10000))
        t0 = time.time()
        rows = build_sensitivity_samples(topo, args.n, rng)
        print(f"  built {len(rows)} sensitivity samples in {time.time()-t0:.1f}s", flush=True)
        report["topologies"][topo] = {"variants": {}}
        for variant in args.variants.split(","):
            variant = variant.strip()
            print(f"  training {variant}...", flush=True)
            rec = train_variant(variant, topo, rows, args.epochs, args.seed)
            report["topologies"][topo]["variants"][variant] = rec
            print(
                f"    gate={'PASS' if rec['gate_ok'] else 'FAIL'} "
                f"sign={rec['mean_sign_acc_gate']:.3f}  "
                f"IL_R2={rec['test_r2_dm']['il_db']:+.3f} "
                f"RL_R2={rec['test_r2_dm']['rl_db']:+.3f}",
                flush=True,
            )

    macro = {}
    for v in args.variants.split(","):
        v = v.strip()
        oks = [report["topologies"][t]["variants"][v]["gate_ok"] for t in report["topologies"]]
        signs = [
            report["topologies"][t]["variants"][v]["mean_sign_acc_gate"]
            for t in report["topologies"]
        ]
        macro[v] = {
            "n_pass": int(sum(oks)), "n": len(oks), "all_pass": all(oks),
            "mean_sign_acc": float(np.mean(signs)),
        }
    report["macro_gate"] = macro
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    # Preserve v1 for comparison
    v1 = args.out.replace(".json", "_v1.json")
    if os.path.isfile(args.out) and not os.path.isfile(v1):
        os.rename(args.out, v1)
    with open(args.out, "w") as fh:
        json.dump(report, fh, indent=2)
    print("\nwrote", args.out, flush=True)
    print("macro_gate", json.dumps(macro, indent=2), flush=True)


if __name__ == "__main__":
    main()
