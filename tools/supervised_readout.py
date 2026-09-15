"""D2: where sizing information is destroyed — pre-pool vs pooled z.

Per topology, 80/20 sizing split, raw MNA metrics (not scalar reward first).

    python tools/supervised_readout.py --out results/graphs/d2_readout.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from env.graph_utils import TOPOLOGY_PARAMS
from env.netlist_graph import build_circuit_graph
from env.param_semantics import param_context_matrix
from models.circuit_encoder import make_encoder
from tools.supervised_sanity import (
    METRIC_KEYS, SPEC, MetricProbe, build_samples, fit_probe, r2_np, _pad_action,
)

MAX_DEV = 16
MAX_PARAM = 8


def _pad_rows(x: torch.Tensor, max_n: int) -> torch.Tensor:
    pad = torch.zeros(max_n, x.size(-1))
    pad[: x.size(0)] = x
    return pad.reshape(-1)


def encode_conditions(topo, spec, rows):
    typed = make_encoder("circuit-typed")
    typed.eval()
    typed._cache_enabled = False
    param_enc = make_encoder("circuit-typed-param")
    param_enc.eval()
    param_enc._cache_enabled = False

    spec_t = torch.tensor(
        __import__("specset.schema", fromlist=["normalize_spec"]).normalize_spec(spec),
        dtype=torch.float,
    ).unsqueeze(0).expand(len(rows), -1)

    acts, raws, zs, hdevs, hparams, ctxs = [], [], [], [], [], []
    with torch.no_grad():
        for row in rows:
            acts.append(torch.tensor(row["action"], dtype=torch.float))
            g = build_circuit_graph(
                topo, spec, state=0, typed=True, params=row["params"],
            )
            raws.append(_pad_rows(g["device"].x, MAX_DEV))
            z, h_d = typed(topo, spec, params=row["params"], return_device=True)
            zs.append(z.squeeze(0))
            hdevs.append(_pad_rows(h_d, MAX_DEV))
            _, h_p = param_enc(topo, spec, params=row["params"], return_param=True)
            hparams.append(_pad_rows(h_p, MAX_PARAM))
            ctx = torch.tensor(
                param_context_matrix(topo, spec, params=row["params"]),
                dtype=torch.float,
            )
            ctxs.append(_pad_rows(ctx, MAX_PARAM))
    return {
        "action_only": torch.cat([spec_t, torch.stack(acts)], dim=-1),
        "raw_device_x": torch.cat([spec_t, torch.stack(raws)], dim=-1),
        "h_dev_prepool": torch.cat([spec_t, torch.stack(hdevs)], dim=-1),
        "z_pooled": torch.cat([spec_t, torch.stack(zs)], dim=-1),
        "z_plus_action": torch.cat(
            [spec_t, torch.stack(zs), torch.stack(acts)], dim=-1,
        ),
        "h_param_r3": torch.cat([spec_t, torch.stack(hparams)], dim=-1),
        "param_context_d1a": torch.cat([spec_t, torch.stack(ctxs)], dim=-1),
    }


def run_topo(topo, n, epochs, seed):
    rng = np.random.default_rng(seed)
    rows = build_samples(topo, n, rng)
    y = torch.stack([torch.tensor(r["y"]) for r in rows], dim=0)
    feats = encode_conditions(topo, SPEC, rows)
    n_tr = int(0.8 * len(rows))
    tr, te = slice(0, n_tr), slice(n_tr, None)
    out = {"n": len(rows), "n_train": n_tr, "conditions": {}}
    for name, x in feats.items():
        per, mse = fit_probe(x[tr], y[tr], x[te], y[te], epochs, 1e-3, seed)
        out["conditions"][name] = {"r2": per, "mse": mse}
        print(f"  {topo:18s} {name:20s} IL={per['il_db']:+.3f} RL={per['rl_db']:+.3f} "
              f"ph={per['rms_phase_err_deg']:+.3f} rew={per['reward']:+.3f}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--epochs", type=int, default=800)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--topologies", default=",".join(TOPOLOGY_PARAMS))
    ap.add_argument("--out", default=os.path.join(
        REPO_ROOT, "results", "graphs", "d2_readout.json",
    ))
    args = ap.parse_args()
    report = {
        "n": args.n, "epochs": args.epochs, "seed": args.seed, "topologies": {},
    }
    for topo in args.topologies.split(","):
        topo = topo.strip()
        report["topologies"][topo] = run_topo(topo, args.n, args.epochs, args.seed)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(report, fh, indent=2)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
