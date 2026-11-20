"""D5b: bounded role-transfer pilot (D4v2 protocol).

Fixes the D5 protocol confound (see results/graphs/D5_PROTOCOL_DIAGNOSIS.md):

- Exact D4v2 selected-token readout for d1a/r3 (no flattened pad, no topo-local knob).
- Sign BCE + small MSE (not magnitude-only).
- Seeds set *before* model init; topology RNG uses a fixed map (no PYTHONHASHSEED).
- Train-only y normalization.
- Sizing-informed baselines: gin/typed get (z ∥ action ∥ role_onehot), not knob index.
- One shared role, one holdout. Incremental JSON writes.
- Reports: balanced sign accuracy, majority baseline, MAE/RMSE, target variance.
- ABCD off. No architecture changes. RL gated.

Default: impedance (Z0) role; hold out Vector_Modulator; train on
Loaded_Line, Switched_Line, Reflection_Type.

    python tools/d5b_role_transfer.py --n 100 --epochs 80 --seed 0 \\
        --out results/graphs/d5b_role_transfer.json
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
from env.param_semantics import (
    N_PARAM_ROLES, PARAM_CONTEXT_DIM, PARAM_ROLE_IDX, param_context_matrix,
    param_role_name, param_role_onehot,
)
from models.circuit_encoder import make_encoder
from models.gnn_encoder import TopologyEncoder
from sim.mna_scorer import mna_evaluate, solve_sparams, sparams_to_metrics
from specset.schema import SPEC_DIM, TRAIN_SPECSET_PATH, normalize_spec
from train_diffusion import action_to_params

METRIC_KEYS = ["il_db", "rl_db", "rms_phase_err_deg", "gain_err_db", "raw_dphi"]
# Primary sensitivity metrics for the impedance-role pilot.
GATE_METRICS = ["il_db", "rl_db", "gain_err_db", "raw_dphi"]
DELTA = 0.08
# Stable per-topology seeds (no hash()).
TOPO_SEED = {
    "Loaded_Line": 101, "Switched_Line": 102, "Reflection_Type": 103,
    "Switched_Filter": 104, "Vector_Modulator": 105, "All_Pass": 106,
}
ROLE_TOPOS = {
    "impedance": ["Loaded_Line", "Switched_Line", "Reflection_Type", "Vector_Modulator"],
    "length": ["Loaded_Line", "Switched_Line", "Reflection_Type", "Vector_Modulator"],
}


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


def role_keys(topo, role):
    return [k for k in TOPOLOGY_PARAMS[topo] if param_role_name(k) == role]


def build_role_samples(topo, role, n, rng, delta=DELTA):
    """Sensitivity samples that only perturb parameters of ``role``."""
    keys = TOPOLOGY_PARAMS[topo]
    rkeys = role_keys(topo, role)
    if not rkeys:
        return []
    sized = sized_devices(topo)
    key_to_sized = {p: i for i, (_, p) in enumerate(sized)}
    rows = []
    attempts = 0
    while len(rows) < n and attempts < n * 25:
        attempts += 1
        base = rng.random(len(keys)).astype(np.float32)
        base = np.clip(base, delta + 0.02, 1.0 - delta - 0.02)
        params0 = _decode(base, topo)
        m0 = _metrics(topo, params0)
        if m0 is None:
            continue
        key = rkeys[int(rng.integers(0, len(rkeys)))]
        if key not in key_to_sized:
            continue
        i = keys.index(key)
        a_pos = base.copy()
        a_pos[i] = float(np.clip(base[i] + delta, 0, 1))
        mp = _metrics(topo, _decode(a_pos, topo))
        if mp is None:
            continue
        dm = (mp - m0).astype(np.float32)
        numeric = params_numeric(params0)
        graphs = [
            build_circuit_graph(
                topo, SPEC, state=s, typed=True, params=numeric, param_nodes=True,
            )
            for s in _states(topo)
        ]
        rows.append({
            "topo": topo,
            "action": _pad_action(base),
            "role": param_role_onehot(key).astype(np.float32),
            "param_key": key,
            "token_idx": int(key_to_sized[key]),
            "graphs": graphs,
            "context": param_context_matrix(topo, SPEC, params=numeric),
            "dm": dm,
        })
    return rows


def sign_acc(y, yhat, eps=1e-6):
    y = np.asarray(y); yhat = np.asarray(yhat)
    mask = np.abs(y) > eps
    if mask.sum() == 0:
        return 1.0
    return float((np.sign(y[mask]) == np.sign(yhat[mask])).mean())


def balanced_sign_acc(y, yhat, eps=1e-6):
    """Average of TPR on positives and TNR on negatives (ignores near-zeros).
    Returns None if either positive or negative class is absent (balanced accuracy undefined).
    """
    y = np.asarray(y); yhat = np.asarray(yhat)
    mask = np.abs(y) > eps
    if mask.sum() == 0:
        return None
    ys, ps = np.sign(y[mask]), np.sign(yhat[mask])
    pos_mask = ys == +1.0
    neg_mask = ys == -1.0
    if pos_mask.sum() == 0 or neg_mask.sum() == 0:
        return None
    tpr = float((ps[pos_mask] == +1.0).mean())
    tnr = float((ps[neg_mask] == -1.0).mean())
    return float(0.5 * (tpr + tnr))


def majority_sign_baseline(y_train, y_test, eps=1e-6):
    """Per-metric majority sign from train labels, scored on test."""
    out = {}
    for i, k in enumerate(METRIC_KEYS):
        yt = y_train[:, i]
        mask = np.abs(yt) > eps
        if mask.sum() == 0:
            maj = 1.0
        else:
            pos = (yt[mask] > 0).mean()
            maj = 1.0 if pos >= 0.5 else -1.0
        pred = np.full_like(y_test[:, i], maj)
        out[k] = {
            "majority_sign": float(maj),
            "sign_acc": sign_acc(y_test[:, i], pred),
            "balanced_sign_acc": balanced_sign_acc(y_test[:, i], pred),
            "train_pos_frac": float((yt[mask] > 0).mean()) if mask.sum() else 0.5,
        }
    return out


def uncertainty_report(y, pred, seed=0, repeats=1000):
    """Paired sample bootstrap; balanced accuracy requires both sign classes."""
    rng = np.random.default_rng(seed)
    report = {}
    for i, key in enumerate(METRIC_KEYS):
        truth, guess = y[:, i], pred[:, i]
        positive, negative = truth > 1e-6, truth < -1e-6
        counts = {"positive": int(positive.sum()), "negative": int(negative.sum()),
                  "near_zero": int((abs(truth) <= 1e-6).sum())}
        both = bool(positive.any() and negative.any())
        maes, bals = [], []
        for _ in range(repeats):
            idx = rng.integers(0, len(truth), len(truth))
            yt, yp = truth[idx], guess[idx]
            maes.append(float(np.abs(yt-yp).mean()))
            if (yt > 1e-6).any() and (yt < -1e-6).any():
                b = balanced_sign_acc(yt, yp)
                if b is not None:
                    bals.append(b)
        report[key] = {
            "counts": counts, "both_sign_classes": both,
            "balanced_sign_accuracy": balanced_sign_acc(truth, guess) if both else None,
            "balanced_sign_ci95": np.quantile(bals, [.025, .975]).tolist() if both and bals else None,
            "mae_ci95": np.quantile(maes, [.025, .975]).tolist(),
            "bootstrap_repeats": repeats,
        }
    return report


def mae_rmse(y, yhat):
    err = yhat - y
    maes = {METRIC_KEYS[i]: float(np.abs(err[:, i]).mean()) for i in range(len(METRIC_KEYS))}
    stds = {METRIC_KEYS[i]: float(y[:, i].std()) for i in range(len(METRIC_KEYS))}
    mae_norm = {
        METRIC_KEYS[i]: (maes[METRIC_KEYS[i]] / stds[METRIC_KEYS[i]]) if stds[METRIC_KEYS[i]] > 1e-12 else None
        for i in range(len(METRIC_KEYS))
    }
    return {
        "mae": maes,
        "mae_normalized": mae_norm,
        "rmse": {METRIC_KEYS[i]: float(np.sqrt((err[:, i] ** 2).mean())) for i in range(len(METRIC_KEYS))},
        "y_std": stds,
        "y_var": {METRIC_KEYS[i]: float(y[:, i].var()) for i in range(len(METRIC_KEYS))},
    }


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


class RoleTransferModel(nn.Module):
    """Shared across topologies. d1a/r3 index the perturbed token (D4v2)."""

    def __init__(self, variant: str):
        super().__init__()
        self.variant = variant
        if variant == "action":
            self.enc = None
            in_dim = SPEC_DIM + SLOT_ACTION_DIM + N_PARAM_ROLES
        elif variant == "gin":
            # Sizing-informed: topology z + current action + transferable role.
            self.enc = TopologyEncoder()
            self._gin = {}
            in_dim = 64 + SPEC_DIM + SLOT_ACTION_DIM + N_PARAM_ROLES
        elif variant == "circuit-typed":
            self.enc = make_encoder("circuit-typed")
            in_dim = 64 + SPEC_DIM + SLOT_ACTION_DIM + N_PARAM_ROLES
        elif variant == "d1a":
            self.enc = make_encoder("circuit-typed")
            self.token_proj = nn.Sequential(
                nn.Linear(64 + PARAM_CONTEXT_DIM, 64), nn.ReLU(),
            )
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
        device = next(self.parameters()).device
        acts = torch.tensor(np.stack([r["action"] for r in rows]), dtype=torch.float, device=device)
        roles = torch.tensor(np.stack([r["role"] for r in rows]), dtype=torch.float, device=device)
        spec_t = torch.tensor(
            np.stack([normalize_spec(SPEC) for _ in rows]), dtype=torch.float, device=device,
        )

        if self.variant == "action":
            return self.head(torch.cat([spec_t, acts, roles], dim=-1))

        if self.variant == "gin":
            zs = []
            for r in rows:
                topo = r["topo"]
                if topo not in self._gin:
                    self._gin[topo] = get_topology_graph(topo)
                g = self._gin[topo]
                z = self.enc(g.x.to(device), g.edge_index.to(device))
                if z.dim() > 1:
                    z = z.squeeze(0)
                zs.append(z)
            return self.head(torch.cat([torch.stack(zs, dim=0), spec_t, acts, roles], dim=-1))

        feats = []
        for r in rows:
            graphs = [g.to(device) for g in r["graphs"]]
            z, h_dev, h_param = self._encode_typed(graphs)
            topo = r["topo"]
            ti = int(r["token_idx"])
            if self.variant == "circuit-typed":
                feats.append(z)
            elif self.variant == "d1a":
                sized = sized_devices(topo)
                name_to_idx = {n: i for i, n in enumerate(device_names(topo))}
                ctx = torch.tensor(r["context"], dtype=torch.float, device=device)
                dname = sized[ti][0]
                tok = torch.cat([h_dev[name_to_idx[dname]], ctx[ti]], dim=-1)
                feats.append(self.token_proj(tok))
            else:
                feats.append(self.token_proj(h_param[ti]))
        h = torch.stack(feats, dim=0)
        if self.variant == "circuit-typed":
            return self.head(torch.cat([h, spec_t, acts, roles], dim=-1))
        return self.head(torch.cat([h, spec_t], dim=-1))


def train_eval(variant, train_rows, test_rows, epochs, seed, batch_size=64, ckpt_dir=None):
    # Seed *before* model construction (D5 bug: seed was after).
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = RoleTransferModel(variant)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    y_tr = np.stack([r["dm"] for r in train_rows])
    y_te = np.stack([r["dm"] for r in test_rows])
    y_mean = torch.tensor(y_tr.mean(0), dtype=torch.float)
    y_std = torch.tensor(y_tr.std(0), dtype=torch.float).clamp_min(1e-4)

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
            from env.sign_alignment import sign_logits_from_normalized_pred
            sign_t = torch.sign(y)
            mask = (y.abs() > 1e-6).float()
            sign_logits = sign_logits_from_normalized_pred(pred, y_mean, y_std)
            bce = F.binary_cross_entropy_with_logits(
                sign_logits, (sign_t > 0).float(), weight=mask, reduction="sum",
            ) / mask.sum().clamp_min(1.0)
            loss = bce + 0.25 * mse
            opt.zero_grad(); loss.backward(); opt.step()
    train_s = time.time() - t0

    model.eval()
    with torch.no_grad():
        pred = model.forward_batch(test_rows)
        yhat = (pred * y_std + y_mean).numpy()

    per_sign = {k: sign_acc(y_te[:, i], yhat[:, i]) for i, k in enumerate(METRIC_KEYS)}
    per_bal = {k: balanced_sign_acc(y_te[:, i], yhat[:, i]) for i, k in enumerate(METRIC_KEYS)}
    maj = majority_sign_baseline(y_tr, y_te)
    err = mae_rmse(y_te, yhat)

    if ckpt_dir:
        os.makedirs(ckpt_dir, exist_ok=True)
        torch.save({"model": model.state_dict(), "y_mean": y_mean, "y_std": y_std},
                   os.path.join(ckpt_dir, f"{variant}.pt"))

    # Per-role / per-key breakdown on holdout
    by_key = {}
    for key in sorted({r["param_key"] for r in test_rows}):
        idx = [i for i, r in enumerate(test_rows) if r["param_key"] == key]
        yk = y_te[idx]; pk = yhat[idx]
        by_key[key] = {
            "n": len(idx),
            "sign_acc": {METRIC_KEYS[i]: sign_acc(yk[:, i], pk[:, i]) for i in range(len(METRIC_KEYS))},
            "balanced_sign_acc": {
                METRIC_KEYS[i]: balanced_sign_acc(yk[:, i], pk[:, i]) for i in range(len(METRIC_KEYS))
            },
            "mae": {METRIC_KEYS[i]: float(np.abs(pk[:, i] - yk[:, i]).mean()) for i in range(len(METRIC_KEYS))},
            "y_std": {METRIC_KEYS[i]: float(yk[:, i].std()) for i in range(len(METRIC_KEYS))},
        }

    valid_gate_keys = [
        k for k in GATE_METRICS
        if per_bal[k] is not None and maj[k]["balanced_sign_acc"] is not None
    ]
    mean_bal = float(np.mean([per_bal[k] for k in valid_gate_keys])) if valid_gate_keys else None
    mean_maj = float(np.mean([maj[k]["balanced_sign_acc"] for k in valid_gate_keys])) if valid_gate_keys else None
    return {
        "variant": variant,
        "n_train": len(train_rows),
        "n_test": len(test_rows),
        "train_seconds": train_s,
        "seed": seed,
        "protocol": "d4v2_token_indexed_sign_bce_role_transfer",
        "test_sign_acc": per_sign,
        "uncertainty": uncertainty_report(y_te, yhat, seed),
        "predictions": {"target": y_te.tolist(), "prediction": yhat.tolist()},
        "test_balanced_sign_acc": per_bal,
        "majority_baseline": maj,
        "errors": err,
        "per_param_key": by_key,
        "valid_gate_metrics": valid_gate_keys,
        "mean_balanced_sign_gate": mean_bal,
        "majority_balanced_sign_gate": mean_maj,
        "beats_majority": (mean_bal > mean_maj + 0.02) if (mean_bal is not None and mean_maj is not None) else False,
        "gate_metrics": GATE_METRICS,
    }


def _write(path, report):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as fh:
        json.dump(report, fh, indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--role", default="impedance", choices=sorted(ROLE_TOPOS))
    ap.add_argument("--holdout", default="Vector_Modulator")
    ap.add_argument("--n", type=int, default=100, help="samples per topology")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--variants", default="action,gin,circuit-typed,d1a,r3")
    ap.add_argument("--out", default=os.path.join(
        REPO_ROOT, "results", "graphs", "d5b_role_transfer.json",
    ))
    ap.add_argument("--ckpt-dir", default=os.path.join(
        REPO_ROOT, "results", "graphs", "d5b_ckpts",
    ))
    args = ap.parse_args()

    topos = ROLE_TOPOS[args.role]
    if args.holdout not in topos:
        raise SystemExit(f"holdout {args.holdout} not in role set {topos}")
    train_topos = [t for t in topos if t != args.holdout]

    report = {
        "protocol": "d5b_d4v2_role_transfer",
        "role": args.role,
        "holdout": args.holdout,
        "train_topologies": train_topos,
        "n": args.n, "epochs": args.epochs, "seed": args.seed,
        "delta_action": DELTA,
        "notes": (
            "Matches D4v2 selected-token + sign BCE for d1a/r3. "
            "Baselines are sizing-informed (action+role), not topo-local knob index. "
            "Seeds set before init. Train-only normalization. Incremental writes."
        ),
        "variants": {},
        "status": "running",
    }
    _write(args.out, report)

    print(f"role={args.role} holdout={args.holdout} train={train_topos}", flush=True)
    by_topo = {}
    for topo in topos:
        rng = np.random.default_rng(args.seed * 1000 + TOPO_SEED[topo])
        t0 = time.time()
        rows = build_role_samples(topo, args.role, args.n, rng)
        print(f"  built {topo}: {len(rows)} in {time.time()-t0:.1f}s", flush=True)
        by_topo[topo] = rows
        report.setdefault("build", {})[topo] = {"n": len(rows)}
        _write(args.out, report)

    train_rows = [r for t in train_topos for r in by_topo[t]]
    test_rows = by_topo[args.holdout]
    print(f"train={len(train_rows)} test={len(test_rows)}", flush=True)

    for variant in args.variants.split(","):
        variant = variant.strip()
        print(f"\ntraining {variant}...", flush=True)
        ckpt = os.path.join(args.ckpt_dir, f"role_{args.role}", f"holdout_{args.holdout}", f"seed_{args.seed}")
        rec = train_eval(
            variant, train_rows, test_rows, args.epochs, args.seed, ckpt_dir=ckpt,
        )
        report["variants"][variant] = rec
        _write(args.out, report)
        print(
            f"  bal_sign={rec['mean_balanced_sign_gate']} "
            f"(majority={rec['majority_balanced_sign_gate']}) "
            f"beats_maj={rec['beats_majority']}  "
            f"IL_sign={rec['test_sign_acc']['il_db']:.3f} "
            f"IL_bal={rec['test_balanced_sign_acc']['il_db']} "
            f"IL_MAE={rec['errors']['mae']['il_db']:.4f} "
            f"IL_ystd={rec['errors']['y_std']['il_db']:.4f}  "
            f"gate_metrics={rec['valid_gate_metrics']}  "
            f"({rec['train_seconds']:.1f}s)",
            flush=True,
        )

    report["status"] = "done"
    # Compact ranking
    ranking = sorted(
        report["variants"].items(),
        key=lambda kv: (
            kv[1]["mean_balanced_sign_gate"]
            if kv[1]["mean_balanced_sign_gate"] is not None else float("-inf")
        ),
        reverse=True,
    )
    report["ranking_balanced_sign"] = [
        {"variant": v, "bal": r["mean_balanced_sign_gate"], "beats_majority": r["beats_majority"]}
        for v, r in ranking
    ]
    _write(args.out, report)
    print("\nwrote", args.out, flush=True)
    print("ranking", json.dumps(report["ranking_balanced_sign"], indent=2), flush=True)


if __name__ == "__main__":
    main()
