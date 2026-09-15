"""Phase 5: supervised graph+sizing → MNA-reward LOOCV.

Train encoder + MLP to predict simulated reward from (graph, sizing, spec).
Six outer folds, multiple seeds, per-holdout numbers — not one average.

    python tools/supervised_loocv.py --out results/graphs/phase5_supervised.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from env.graph_utils import SLOT_ACTION_DIM, TOPOLOGY_PARAMS, get_topology_graph
from env.netlist_graph import params_numeric
from models.circuit_encoder import is_circuit_encoder, make_encoder
from specset.schema import SPEC_DIM, TRAIN_SPECSET_PATH, normalize_spec
from sim.mna_scorer import mna_evaluate
from train_diffusion import action_to_params

TOPOS = list(TOPOLOGY_PARAMS.keys())


class RewardProbe(nn.Module):
    def __init__(self, z_dim=64, spec_dim=SPEC_DIM, act_dim=SLOT_ACTION_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim + spec_dim + act_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, z, spec, action):
        return self.net(torch.cat([z, spec, action], dim=-1)).squeeze(-1)


def _load_specs(path, n, rng):
    with open(path) as fh:
        doc = json.load(fh)
    specs = [e["spec"] for e in doc["specs"]]
    idx = rng.choice(len(specs), size=min(n, len(specs)), replace=False)
    return [specs[int(i)] for i in idx]


def _pad_action(action, dim=SLOT_ACTION_DIM):
    out = np.full(dim, 0.5, dtype=np.float32)
    a = np.asarray(action, dtype=np.float32).reshape(-1)
    out[: min(len(a), dim)] = a[:dim]
    return out


def build_dataset(n_per_topo, spec_pool, rng) -> dict[str, list[dict]]:
    data = {t: [] for t in TOPOS}
    for topo in TOPOS:
        keys = TOPOLOGY_PARAMS[topo]
        attempts = 0
        while len(data[topo]) < n_per_topo and attempts < n_per_topo * 8:
            attempts += 1
            spec = spec_pool[int(rng.integers(len(spec_pool)))]
            action = rng.random(len(keys))
            params = action_to_params(
                action, topo, spec, sizing="log", bounds="electrical",
                switch_model="ideal",
            )
            score, agg = mna_evaluate(topo, params, spec)
            if agg is None:
                continue
            data[topo].append({
                "spec": spec,
                "action": _pad_action(action).tolist(),
                "params": params_numeric(params),
                "reward": float(score),
                "il_db": float(agg.get("il_db", 0.0) or 0.0),
                "rms_phase_err_deg": float(
                    agg.get("rms_phase_err_deg", 0.0) or 0.0
                ),
            })
    return data


def _encode_one(encoder_name, gnn, row, device, gin_graphs):
    spec = row["spec"]
    spec_t = torch.tensor(normalize_spec(spec), dtype=torch.float, device=device)
    act_t = torch.tensor(row["action"], dtype=torch.float, device=device)
    if is_circuit_encoder(encoder_name):
        z = gnn(row["topo"], spec, params=row["params"]).squeeze(0)
    else:
        g = gin_graphs[row["topo"]]
        z = gnn(g.x, g.edge_index).squeeze(0)
    return z, spec_t, act_t


def r2_score(y, yhat):
    y = y.detach()
    yhat = yhat.detach()
    ss_res = float(((y - yhat) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    if ss_tot < 1e-12:
        return 0.0
    return 1.0 - ss_res / ss_tot


def run_fold(encoder_name, holdout, seed, dataset, n_train, n_test, epochs, device):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    train_rows = []
    for t in TOPOS:
        if t == holdout:
            continue
        rows = list(dataset[t])
        rng.shuffle(rows)
        for r in rows[:n_train]:
            train_rows.append({**r, "topo": t})
    test_rows = [{**r, "topo": holdout} for r in dataset[holdout][:n_test]]

    gnn = make_encoder(encoder_name).to(device)
    if hasattr(gnn, "_cache_enabled"):
        gnn._cache_enabled = False
    gnn.train()
    probe = RewardProbe().to(device)
    opt = torch.optim.Adam(list(gnn.parameters()) + list(probe.parameters()), lr=1e-3)
    gin_graphs = None
    if not is_circuit_encoder(encoder_name):
        gin_graphs = {}
        for t in TOPOS:
            g = get_topology_graph(t)
            g.x = g.x.to(device)
            g.edge_index = g.edge_index.to(device)
            gin_graphs[t] = g

    batch = 32
    for _ in range(epochs):
        rng.shuffle(train_rows)
        gnn.train()
        probe.train()
        for i in range(0, len(train_rows), batch):
            chunk = train_rows[i:i + batch]
            zs, specs, acts, y = [], [], [], []
            for row in chunk:
                z, s, a = _encode_one(encoder_name, gnn, row, device, gin_graphs)
                zs.append(z)
                specs.append(s)
                acts.append(a)
                y.append(row["reward"])
            z_b = torch.stack(zs, dim=0)
            s_b = torch.stack(specs, dim=0)
            a_b = torch.stack(acts, dim=0)
            y_b = torch.tensor(y, dtype=torch.float, device=device)
            pred = probe(z_b, s_b, a_b)
            loss = F.mse_loss(pred, y_b)
            opt.zero_grad()
            loss.backward()
            opt.step()

    gnn.eval()
    probe.eval()

    def _eval(rows):
        preds, ys = [], []
        with torch.no_grad():
            for row in rows:
                z, s, a = _encode_one(encoder_name, gnn, row, device, gin_graphs)
                preds.append(float(probe(z.unsqueeze(0), s.unsqueeze(0), a.unsqueeze(0))))
                ys.append(row["reward"])
        yt = torch.tensor(ys)
        yp = torch.tensor(preds)
        return {
            "mae": float((yt - yp).abs().mean()),
            "r2": r2_score(yt, yp),
            "y_mean": float(yt.mean()),
            "yhat_mean": float(yp.mean()),
            "n": len(rows),
        }

    train_m = _eval(train_rows[: min(len(train_rows), n_test * 5)])
    test_m = _eval(test_rows)
    return {
        "encoder": encoder_name,
        "holdout": holdout,
        "seed": seed,
        "train": train_m,
        "test": test_m,
        "gap_r2": train_m["r2"] - test_m["r2"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoders", default="gin,circuit,circuit-typed")
    ap.add_argument("--seeds", default="42,1337,2026")
    ap.add_argument("--n-per-topo", type=int, default=120)
    ap.add_argument("--n-train", type=int, default=80)
    ap.add_argument("--n-test", type=int, default=40)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--n-specs", type=int, default=400)
    ap.add_argument("--out", default=os.path.join(
        REPO_ROOT, "results", "graphs", "phase5_supervised.json",
    ))
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    rng = np.random.default_rng(0)
    spec_pool = _load_specs(TRAIN_SPECSET_PATH, args.n_specs, rng)
    print(f"Building MNA dataset ({args.n_per_topo}/topo)…", flush=True)
    dataset = build_dataset(args.n_per_topo, spec_pool, rng)
    sizes = {t: len(v) for t, v in dataset.items()}
    print(f"dataset sizes {sizes}", flush=True)

    encoders = [s.strip() for s in args.encoders.split(",") if s.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    results = []
    for enc in encoders:
        for hold in TOPOS:
            for seed in seeds:
                print(f"fold {enc} hold={hold} seed={seed}", flush=True)
                r = run_fold(
                    enc, hold, seed, dataset,
                    args.n_train, args.n_test, args.epochs, device,
                )
                results.append(r)
                print(
                    f"  train_r2={r['train']['r2']:.3f} "
                    f"test_r2={r['test']['r2']:.3f} "
                    f"test_mae={r['test']['mae']:.3f}",
                    flush=True,
                )

    by_enc_hold = {}
    for r in results:
        key = (r["encoder"], r["holdout"])
        by_enc_hold.setdefault(key, []).append(r["test"]["r2"])
    summary = []
    for enc in encoders:
        row = {"encoder": enc, "per_holdout": {}}
        r2s = []
        for hold in TOPOS:
            vals = by_enc_hold.get((enc, hold), [])
            row["per_holdout"][hold] = {
                "mean_r2": float(np.mean(vals)) if vals else None,
                "std_r2": float(np.std(vals)) if vals else None,
                "seeds": vals,
            }
            if vals:
                r2s.append(float(np.mean(vals)))
        row["macro_mean_r2"] = float(np.mean(r2s)) if r2s else None
        row["worst_holdout"] = (
            min(row["per_holdout"].items(), key=lambda kv: kv[1]["mean_r2"] or 0)[0]
            if r2s else None
        )
        summary.append(row)

    report = {
        "dataset_sizes": sizes,
        "n_train": args.n_train,
        "n_test": args.n_test,
        "epochs": args.epochs,
        "seeds": seeds,
        "folds": results,
        "summary": summary,
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(report, fh, indent=2)
    print(json.dumps({"summary": summary, "out": args.out}, indent=2))


if __name__ == "__main__":
    main()
