"""D3: end-to-end same-topology interpolation + bottleneck probes.

Trains representations to predict raw MNA metrics on an 80/20 sizing split.

Primary heads are **representation-only** (no raw action concat) so the gate
tests whether sizing identity survives the encoder — not whether an MLP can
read the action vector.

Variants:
  action        — MLP on action (ceiling / information upper bound)
  gin           — GIN z only (topology-static; should fail sizing)
  circuit-typed — pooled z from sized typed graph
  d1a           — (h_dev ∥ param_context) tokens
  r3            — h_param tokens (circuit-typed-param)

Bottleneck probes (raw device.x, context, z, param token) are trained jointly.

    python tools/d3_trained_interp.py --n 250 --epochs 300 \\
        --out results/graphs/d3_trained_interp.json
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
from env.param_semantics import PARAM_CONTEXT_DIM, PARAM_NODE_IN, param_context_matrix
from models.circuit_encoder import DEVICE_IN, make_encoder
from models.gnn_encoder import TopologyEncoder
from sim.mna_scorer import mna_evaluate, solve_sparams, sparams_to_metrics
from specset.schema import SPEC_DIM, TRAIN_SPECSET_PATH, normalize_spec
from train_diffusion import action_to_params

METRIC_KEYS = ["il_db", "rl_db", "rms_phase_err_deg", "gain_err_db", "raw_dphi"]
GATE_METRICS = {
    "Loaded_Line": ["il_db", "rl_db", "rms_phase_err_deg", "gain_err_db"],
    "Switched_Line": ["il_db", "rl_db", "rms_phase_err_deg", "gain_err_db"],
    # Reflection_Type phase_err is discontinuous; gate on IL/RL/gain/raw_dphi.
    "Reflection_Type": ["il_db", "rl_db", "gain_err_db", "raw_dphi"],
    "Switched_Filter": ["il_db", "rl_db", "rms_phase_err_deg", "gain_err_db"],
    "Vector_Modulator": ["il_db", "rl_db", "rms_phase_err_deg", "gain_err_db"],
    "All_Pass": ["il_db", "rl_db", "rms_phase_err_deg", "gain_err_db"],
}
GATE_THRESH = 0.5
MAX_PARAM_TOKENS = 8  # pad/flatten; avoid mean-pool collapse of sizing identity


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


def build_samples(topo, n, rng):
    keys = TOPOLOGY_PARAMS[topo]
    rows = []
    attempts = 0
    while len(rows) < n and attempts < n * 12:
        attempts += 1
        action = rng.random(len(keys))
        params = action_to_params(
            action, topo, SPEC, sizing="log", bounds="electrical",
            switch_model="ideal",
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
            "action": _pad_action(action),
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


class VariantModel(nn.Module):
    def __init__(self, variant: str, topo: str):
        super().__init__()
        self.variant = variant
        self.topo = topo
        self.sized = sized_devices(topo)
        self.dnames = device_names(topo)
        self.name_to_idx = {n: i for i, n in enumerate(self.dnames)}

        # Representation-only heads (except action ceiling). Raw action is
        # NOT concatenated into graph variants — that would mask pooling loss.
        if variant == "action":
            self.enc = None
            in_dim = SPEC_DIM + SLOT_ACTION_DIM
        elif variant == "gin":
            self.enc = TopologyEncoder()
            self.gin_graph = get_topology_graph(topo)
            in_dim = 64 + SPEC_DIM
        elif variant == "circuit-typed":
            self.enc = make_encoder("circuit-typed")
            in_dim = 64 + SPEC_DIM
        elif variant == "d1a":
            self.enc = make_encoder("circuit-typed")
            self.token_proj = nn.Sequential(
                nn.Linear(64 + PARAM_CONTEXT_DIM, 64), nn.ReLU(),
            )
            in_dim = MAX_PARAM_TOKENS * 64 + SPEC_DIM
        elif variant == "r3":
            self.enc = make_encoder("circuit-typed-param")
            self.token_proj = nn.Sequential(nn.Linear(64, 64), nn.ReLU())
            in_dim = MAX_PARAM_TOKENS * 64 + SPEC_DIM
        else:
            raise ValueError(variant)

        self.head = Head(in_dim)
        self.probe_raw = Head(16 * DEVICE_IN + SPEC_DIM)
        self.probe_z = Head(64 + SPEC_DIM)
        self.probe_param = Head(64 + SPEC_DIM) if variant in ("d1a", "r3") else None
        self.probe_context = Head(8 * PARAM_CONTEXT_DIM + SPEC_DIM)

    def _encode_typed_from_graphs(self, graphs):
        """Multi-state encode using prebuilt HeteroData list."""
        enc = self.enc
        h_list, p_list = [], []
        for g in graphs:
            if self.variant == "r3":
                # Param encoder encode_state returns z, h_d, h_p
                _, h_d, h_p = enc.encode_state(g)
                h_list.append(h_d)
                p_list.append(h_p)
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
        spec_t = torch.tensor(
            np.stack([normalize_spec(SPEC) for _ in rows]),
            dtype=torch.float, device=device,
        )
        aux = {}

        # raw device.x from state-0 graph
        raw_pad = torch.zeros(B, 16, DEVICE_IN, device=device)
        for i, r in enumerate(rows):
            x = r["graphs"][0]["device"].x.to(device)
            raw_pad[i, : x.size(0)] = x
        aux["raw"] = raw_pad.reshape(B, -1)

        ctx_pad = torch.zeros(B, 8, PARAM_CONTEXT_DIM, device=device)
        for i, r in enumerate(rows):
            c = torch.tensor(r["context"], dtype=torch.float, device=device)
            ctx_pad[i, : c.size(0)] = c
        aux["context"] = ctx_pad.reshape(B, -1)

        if self.variant == "action":
            x = torch.cat([spec_t, acts], dim=-1)
            return self.head(x), aux

        if self.variant == "gin":
            g = self.gin_graph
            z = self.enc(g.x.to(device), g.edge_index.to(device))
            if z.dim() == 1:
                z = z.unsqueeze(0)
            z = z.expand(B, -1).contiguous()
            aux["z"] = z
            return self.head(torch.cat([z, spec_t], dim=-1)), aux

        zs, params_h = [], []
        for r in rows:
            graphs = [g.to(device) for g in r["graphs"]]
            z, h_dev, h_param = self._encode_typed_from_graphs(graphs)
            zs.append(z)
            if self.variant == "d1a":
                ctx = torch.tensor(r["context"], dtype=torch.float, device=device)
                toks = []
                for j, (dname, _) in enumerate(self.sized):
                    toks.append(torch.cat([h_dev[self.name_to_idx[dname]], ctx[j]], dim=-1))
                tok = self.token_proj(torch.stack(toks, dim=0))  # [n, 64]
                pad = torch.zeros(MAX_PARAM_TOKENS, 64, device=device)
                pad[: min(tok.size(0), MAX_PARAM_TOKENS)] = tok[:MAX_PARAM_TOKENS]
                params_h.append(pad.reshape(-1))
            elif self.variant == "r3":
                tok = self.token_proj(h_param)
                pad = torch.zeros(MAX_PARAM_TOKENS, 64, device=device)
                pad[: min(tok.size(0), MAX_PARAM_TOKENS)] = tok[:MAX_PARAM_TOKENS]
                params_h.append(pad.reshape(-1))

        z = torch.stack(zs, dim=0)
        aux["z"] = z
        if self.variant == "circuit-typed":
            return self.head(torch.cat([z, spec_t], dim=-1)), aux

        h = torch.stack(params_h, dim=0)
        # 64-d probe: mean over padded token slots (zeros dilute lightly).
        aux["param"] = h.view(B, MAX_PARAM_TOKENS, 64).mean(dim=1)
        return self.head(torch.cat([h, spec_t], dim=-1)), aux


def train_variant(variant, topo, rows, epochs, seed, batch_size=64):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    n_tr = int(0.8 * len(rows))
    idx = rng.permutation(len(rows))
    train_rows = [rows[i] for i in idx[:n_tr]]
    test_rows = [rows[i] for i in idx[n_tr:]]

    model = VariantModel(variant, topo)
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
            pred, aux = model.forward_batch(batch)
            target = (y - y_mean) / y_std
            loss = ((pred - target) ** 2).mean()
            spec_b = torch.tensor(
                np.stack([normalize_spec(SPEC)] * len(batch)), dtype=torch.float,
            )
            loss = loss + 0.05 * ((model.probe_raw(
                torch.cat([aux["raw"], spec_b], dim=-1)) - target) ** 2).mean()
            loss = loss + 0.05 * ((model.probe_context(
                torch.cat([aux["context"], spec_b], dim=-1)) - target) ** 2).mean()
            if aux.get("z") is not None:
                loss = loss + 0.05 * ((model.probe_z(
                    torch.cat([aux["z"], spec_b], dim=-1)) - target) ** 2).mean()
            if model.probe_param is not None and aux.get("param") is not None:
                loss = loss + 0.05 * ((model.probe_param(
                    torch.cat([aux["param"], spec_b], dim=-1)) - target) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
    train_s = time.time() - t0

    model.eval()

    def eval_split(split):
        with torch.no_grad():
            pred, aux = model.forward_batch(split)
            yhat = pred * y_std + y_mean
            y = torch.tensor(np.stack([r["y"] for r in split]), dtype=torch.float)
            per = {METRIC_KEYS[i]: r2_np(y[:, i].numpy(), yhat[:, i].numpy())
                   for i in range(len(METRIC_KEYS))}
            spec_b = torch.tensor(
                np.stack([normalize_spec(SPEC)] * len(split)), dtype=torch.float,
            )
            probes = {"head": per}
            for name, tensor, probe in (
                ("raw_device_x", aux.get("raw"), model.probe_raw),
                ("param_context", aux.get("context"), model.probe_context),
                ("z_pooled", aux.get("z"), model.probe_z),
                ("param_token", aux.get("param"), model.probe_param),
            ):
                if tensor is None or probe is None:
                    continue
                ph = probe(torch.cat([tensor, spec_b], dim=-1)) * y_std + y_mean
                probes[name] = {
                    METRIC_KEYS[i]: r2_np(y[:, i].numpy(), ph[:, i].numpy())
                    for i in range(len(METRIC_KEYS))
                }
        return per, probes

    test_r2, probes = eval_split(test_rows)
    train_r2, _ = eval_split(train_rows)
    gate_keys = GATE_METRICS[topo]
    return {
        "n_train": len(train_rows),
        "n_test": len(test_rows),
        "train_seconds": train_s,
        "train_r2": train_r2,
        "test_r2": test_r2,
        "gate_metrics": gate_keys,
        "gate_ok": all(test_r2[k] > GATE_THRESH for k in gate_keys),
        "bottleneck_probes_test": probes,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=250)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--topologies", default=",".join(TOPOLOGY_PARAMS))
    ap.add_argument("--variants", default="action,gin,circuit-typed,d1a,r3")
    ap.add_argument("--out", default=os.path.join(
        REPO_ROOT, "results", "graphs", "d3_trained_interp.json",
    ))
    args = ap.parse_args()

    report = {
        "n": args.n, "epochs": args.epochs, "seed": args.seed,
        "gate_thresh": GATE_THRESH,
        "topologies": {},
        "notes": {
            "Reflection_Type": (
                "rms_phase_err excluded from gate (discontinuous vs ideal "
                "−22.5°); gate uses IL/RL/gain/raw_dphi."
            ),
            "All_Pass": "D1 centre/ratio/coupling semantics active in d1a/r3.",
        },
    }

    for topo in args.topologies.split(","):
        topo = topo.strip()
        print(f"\n=== {topo} ===", flush=True)
        rng = np.random.default_rng(args.seed + (hash(topo) % 10000))
        t0 = time.time()
        rows = build_samples(topo, args.n, rng)
        print(f"  built {len(rows)} samples in {time.time()-t0:.1f}s", flush=True)
        report["topologies"][topo] = {"variants": {}}
        for variant in args.variants.split(","):
            variant = variant.strip()
            print(f"  training {variant}...", flush=True)
            rec = train_variant(variant, topo, rows, args.epochs, args.seed)
            report["topologies"][topo]["variants"][variant] = rec
            tr = rec["test_r2"]
            print(
                f"    gate={'PASS' if rec['gate_ok'] else 'FAIL'} "
                f"({rec['train_seconds']:.1f}s)  "
                f"IL={tr['il_db']:+.3f} RL={tr['rl_db']:+.3f} "
                f"ph={tr['rms_phase_err_deg']:+.3f} dphi={tr['raw_dphi']:+.3f}",
                flush=True,
            )

    macro = {}
    for v in args.variants.split(","):
        v = v.strip()
        oks = [
            report["topologies"][t]["variants"][v]["gate_ok"]
            for t in report["topologies"]
        ]
        macro[v] = {"n_pass": int(sum(oks)), "n": len(oks), "all_pass": all(oks)}
    report["macro_gate"] = macro
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(report, fh, indent=2)
    print("\nwrote", args.out, flush=True)
    print("macro_gate", json.dumps(macro, indent=2), flush=True)


if __name__ == "__main__":
    main()
