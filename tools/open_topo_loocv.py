#!/usr/bin/env python3
"""Leave-one-topology-out selection regret on the open topology set.

Supervised over precomputed r_star. No RL, no SPICE in the loop.

Selectors
---------
random        analytic expected regret over N topologies
maj_regime    constant within active/passive partition; cannot name held-out
spec_mlp_39   39-way head — structurally undefined on held-out (the result)
spec_mlp_40   40-way head with held-out logit receiving no gradient
graph_ranker  CircuitEncoder + RankHead — defined and informed
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from env.netlist_graph import VM_IQ, build_circuit_graph
from models.circuit_encoder import CircuitEncoder
from specset.schema import SPEC_DIM, normalize_spec
from topology.load_pool import all_topology_names, load_pool, register_all

G_SCALE_MIN = 0.7
VM_MIN_DRIVE_MW = 5.0 * (G_SCALE_MIN ** 2) * max(
    gi * gi + gq * gq for gi, gq in VM_IQ
)
FC_BANDS = (("sub10", 0.0, 10.0), ("10to20", 10.0, 20.0), ("20plus", 20.0, 1e9))


def regime(spec: dict) -> str:
    return ("active_allowed" if float(spec["pmax_mw"]) >= VM_MIN_DRIVE_MW
            else "passive_only")


def fc_band(fc: float) -> str:
    for name, lo, hi in FC_BANDS:
        if lo <= fc < hi:
            return name
    return FC_BANDS[-1][0]


class RankHead(nn.Module):
    def __init__(self, spec_dim: int = SPEC_DIM, z_dim: int = 64, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(spec_dim + z_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, s: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([s, z], dim=-1)).squeeze(-1)


def load_rstar(path: str, switch_model: str, topologies: list[str]) -> list[dict]:
    with open(path) as fh:
        doc = json.load(fh)
    out = []
    for e in doc["specs"]:
        rs = (e.get("r_star") or {}).get(switch_model)
        if not rs or not rs.get("per_topology"):
            continue
        pt = dict(rs["per_topology"])
        # Keep None for inadmissible; do not invent a sentinel that pollutes regret.
        adm = e.get("admissible") or {}
        for t in topologies:
            if t not in pt:
                pt[t] = None
            # If admissibility mask says False, force None
            if adm and not adm.get(t, True):
                pt[t] = None
        finite = {t: v for t, v in pt.items() if v is not None and t in topologies}
        if len(finite) < 2:
            continue
        fixed = dict(e)
        fixed["r_star"] = dict(e["r_star"])
        fixed["r_star"][switch_model] = dict(rs)
        fixed["r_star"][switch_model]["per_topology"] = {
            t: pt.get(t) for t in topologies
        }
        best_t = max(finite, key=lambda t: finite[t])
        fixed["r_star"][switch_model]["best"] = finite[best_t]
        fixed["r_star"][switch_model]["argmax"] = best_t
        out.append(fixed)
    return out


def admissible_vals(entry: dict, topologies: list[str], switch_model: str
                    ) -> tuple[list[str], list[float]]:
    pt = entry["r_star"][switch_model]["per_topology"]
    names, vals = [], []
    for t in topologies:
        v = pt.get(t)
        if v is not None:
            names.append(t)
            vals.append(float(v))
    return names, vals


def regret_of_name(sel: str, names: list[str], vals: list[float]) -> float:
    best = max(vals)
    if sel not in names:
        # Selected an inadmissible / held-out-undefined topology → full regret
        return best - min(vals)
    return best - vals[names.index(sel)]


def precompute_embeddings(encoder: CircuitEncoder, topologies: list[str],
                          device: torch.device) -> dict[str, torch.Tensor]:
    """Cache z_topo per topology at a reference fc (28 GHz)."""
    cache = {}
    spec = {"fc_ghz": 28.0, "tech": 0}
    encoder.eval()
    with torch.no_grad():
        for name in topologies:
            z = encoder(name, spec)  # [1, out_dim]
            cache[name] = z.squeeze(0).cpu()
    return cache


def train_graph_ranker(entries, topologies, emb_cache, switch_model,
                       epochs=30, T=0.5, seed=0, device="cpu"):
    torch.manual_seed(seed)
    ranker = RankHead().to(device)
    opt = torch.optim.Adam(ranker.parameters(), lr=1e-3)
    Z = torch.stack([emb_cache[t] for t in topologies]).to(device)  # [N, 64]

    X, Y, M = [], [], []
    for e in entries:
        s = normalize_spec(e["spec"])
        vals, mask = [], []
        for t in topologies:
            v = e["r_star"][switch_model]["per_topology"].get(t)
            if v is None:
                vals.append(-1e3)
                mask.append(0.0)
            else:
                vals.append(float(v))
                mask.append(1.0)
        if sum(mask) < 2:
            continue
        X.append(s)
        Y.append(vals)
        M.append(mask)
    if not X:
        return ranker
    X = torch.tensor(np.asarray(X), dtype=torch.float32, device=device)
    Y = torch.tensor(np.asarray(Y), dtype=torch.float32, device=device)
    M = torch.tensor(np.asarray(M), dtype=torch.float32, device=device)
    # Softmax only over admissible: set inadmissible logits to -inf via mask
    Y_masked = Y.masked_fill(M < 0.5, -1e9)
    p_star = F.softmax(Y_masked / T, dim=-1)

    n = len(X)
    bs = min(256, n)
    for _ in range(epochs):
        perm = torch.randperm(n)
        for k in range(0, n, bs):
            idx = perm[k:k + bs]
            s = X[idx]
            B = s.size(0)
            s_rep = s.unsqueeze(1).expand(B, len(topologies), -1).reshape(-1, SPEC_DIM)
            z_rep = Z.unsqueeze(0).expand(B, -1, -1).reshape(-1, Z.size(-1))
            u = ranker(s_rep, z_rep).view(B, len(topologies))
            u_masked = u.masked_fill(M[idx] < 0.5, -1e9)
            log_p = F.log_softmax(u_masked / T, dim=-1)
            loss = -(p_star[idx] * log_p).sum(dim=-1).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
    return ranker


def train_spec_mlp(entries, topologies, switch_model, n_out,
                   holdout_idx=None, epochs=30, seed=0, device="cpu"):
    """Train n_out-way MLP. If holdout_idx is set, that logit gets no gradient."""
    torch.manual_seed(seed)
    X, y = [], []
    for e in entries:
        vals = []
        for t in topologies:
            v = e["r_star"][switch_model]["per_topology"].get(t)
            vals.append(-1e9 if v is None else float(v))
        X.append(normalize_spec(e["spec"]))
        order = np.argsort(vals)[::-1]
        lab = int(order[0])
        if holdout_idx is not None and lab == holdout_idx:
            lab = int(order[1]) if len(order) > 1 else 0
        if n_out == len(topologies) - 1 and holdout_idx is not None:
            if lab > holdout_idx:
                lab = lab - 1
            elif lab == holdout_idx:
                lab = 0
        y.append(lab)
    X = torch.tensor(np.asarray(X), dtype=torch.float32, device=device)
    y = torch.tensor(np.asarray(y), dtype=torch.long, device=device)

    net = nn.Sequential(
        nn.Linear(SPEC_DIM, 64), nn.ReLU(),
        nn.Linear(64, 64), nn.ReLU(),
        nn.Linear(64, n_out),
    ).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    lossf = nn.CrossEntropyLoss()
    bs = 256
    for _ in range(epochs):
        perm = torch.randperm(len(X))
        for k in range(0, len(X), bs):
            b = perm[k:k + bs]
            opt.zero_grad()
            logits = net(X[b])
            if n_out == len(topologies) and holdout_idx is not None:
                mask = torch.ones_like(logits)
                mask[:, holdout_idx] = 0.0
                logits = logits * mask
            lossf(logits, y[b]).backward()
            opt.step()
    return net


def regret_of(sel_idx: int, vals: list[float]) -> float:
    best = max(vals)
    return best - vals[sel_idx]


def summarize(regrets: list[float]) -> dict:
    a = np.asarray(regrets, dtype=float)
    if len(a) == 0:
        return {"n": 0, "mean": None, "p90": None}
    return {"n": int(len(a)), "mean": float(a.mean()),
            "p90": float(np.percentile(a, 90))}


def run_fold(held_out: str, topologies: list[str], entries: list[dict],
             emb_cache: dict, switch_model: str, device: str,
             noise_floor: float) -> dict:
    hold_i = topologies.index(held_out)
    train_topos = [t for t in topologies if t != held_out]

    n = len(entries)
    rng = np.random.default_rng(abs(hash(held_out)) % (2**31))
    perm = rng.permutation(n)
    n_fit = max(1, int(0.8 * n))
    fit_e = [entries[i] for i in perm[:n_fit]]
    eval_e = [entries[i] for i in perm[n_fit:]]

    by_reg = {"active_allowed": np.zeros(len(train_topos)),
              "passive_only": np.zeros(len(train_topos))}
    for e in fit_e:
        names, vals = admissible_vals(e, train_topos, switch_model)
        if not vals:
            continue
        pick = names[int(np.argmax(vals))]
        by_reg[regime(e["spec"])][train_topos.index(pick)] += 1

    mlp39 = train_spec_mlp(fit_e, topologies, switch_model,
                           n_out=len(train_topos), holdout_idx=hold_i,
                           device=device)
    mlp40 = train_spec_mlp(fit_e, topologies, switch_model,
                           n_out=len(topologies), holdout_idx=hold_i,
                           device=device)
    ranker = train_graph_ranker(fit_e, train_topos,
                                {t: emb_cache[t] for t in train_topos},
                                switch_model, device=device)

    results = {k: [] for k in
               ("random", "maj_regime", "spec_mlp_39", "spec_mlp_40", "graph_ranker")}
    strata = defaultdict(lambda: defaultdict(list))

    Z_train = torch.stack([emb_cache[t] for t in train_topos]).to(device)

    for e in eval_e:
        names_all, vals_all = admissible_vals(e, topologies, switch_model)
        if len(vals_all) < 2:
            continue
        s = torch.tensor(normalize_spec(e["spec"]), dtype=torch.float32,
                         device=device).unsqueeze(0)
        sm = e["r_star"][switch_model]
        stratum = sm.get("stratum", "unknown")
        reg = regime(e["spec"])

        # random: expected regret over uniform among *admissible* topologies
        r_rand = float(np.mean([regret_of(i, vals_all) for i in range(len(vals_all))]))
        results["random"].append(r_rand)

        # maj_regime: pick among train_topos ∩ admissible — cannot name held-out
        maj_counts = by_reg[reg]
        # Restrict to admissible train topos
        cand = [(train_topos[i], maj_counts[i]) for i in range(len(train_topos))
                if train_topos[i] in names_all]
        if not cand:
            continue
        maj_pick = max(cand, key=lambda x: x[1])[0]
        r_maj = regret_of_name(maj_pick, names_all, vals_all)
        results["maj_regime"].append(r_maj)

        with torch.no_grad():
            logits39 = mlp39(s).cpu().numpy()[0]
            # Map 39-way index → train_topo, then only among admissible
            order39 = np.argsort(logits39)[::-1]
            pick39 = None
            for oi in order39:
                t = train_topos[int(oi)]
                if t in names_all:
                    pick39 = t
                    break
            if pick39 is None:
                continue
            r39 = regret_of_name(pick39, names_all, vals_all)
        results["spec_mlp_39"].append(r39)

        with torch.no_grad():
            logits40 = mlp40(s).cpu().numpy()[0]
            order40 = np.argsort(logits40)[::-1]
            pick40 = None
            for oi in order40:
                t = topologies[int(oi)]
                if t in names_all:
                    pick40 = t
                    break
            if pick40 is None:
                continue
            r40 = regret_of_name(pick40, names_all, vals_all)
        results["spec_mlp_40"].append(r40)

        with torch.no_grad():
            s_rep = s.expand(len(train_topos), -1)
            u = ranker(s_rep, Z_train).cpu().numpy()
            # Score held-out zero-shot
            z_h = emb_cache[held_out].unsqueeze(0).to(device)
            u_h = ranker(s, z_h).item()
            scores = {train_topos[i]: float(u[i]) for i in range(len(train_topos))}
            scores[held_out] = float(u_h)
            # Pick best among admissible
            adm_scores = [(t, scores[t]) for t in names_all if t in scores]
            pick_g = max(adm_scores, key=lambda x: x[1])[0]
            rg = regret_of_name(pick_g, names_all, vals_all)
        results["graph_ranker"].append(rg)

        for sel, r in (("random", r_rand), ("maj_regime", r_maj),
                       ("spec_mlp_39", r39), ("spec_mlp_40", r40),
                       ("graph_ranker", rg)):
            strata[sel][stratum].append(r)
            strata[sel][reg].append(r)

    summary = {sel: summarize(rs) for sel, rs in results.items()}
    stratified = {sel: {k: summarize(v) for k, v in buckets.items()}
                  for sel, buckets in strata.items()}
    return {
        "held_out": held_out,
        "n_eval": len(eval_e),
        "summary": summary,
        "stratified": stratified,
        "noise_floor_p95": noise_floor,
        "note_spec_mlp_39": (
            "spec_mlp_39 has no output head for the held-out topology; "
            "reported regret is under forced fallback to the 39-way argmax "
            "(structurally undefined as a selector for the held-out class)."
        ),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--r-star", default=os.path.join(
        REPO_ROOT, "results", "open_topo", "r_star_40.json"))
    ap.add_argument("--dist", default=os.path.join(
        REPO_ROOT, "results", "open_topo", "distribution_check.json"))
    ap.add_argument("--out", default=os.path.join(
        REPO_ROOT, "results", "open_topo", "loocv_selection.json"))
    ap.add_argument("--switch-model", default="ideal", choices=("ideal", "realistic"))
    ap.add_argument("--noise-floor", type=float, default=0.118)
    ap.add_argument("--max-folds", type=int, default=0,
                    help="0 = all topologies; else cap for smoke")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    register_all()
    topologies = all_topology_names()
    entries = load_rstar(args.r_star, args.switch_model, topologies)
    print(f"loaded {len(entries)} specs with full r_star over {len(topologies)} topos")

    functional = topologies
    if os.path.exists(args.dist):
        with open(args.dist) as fh:
            d = json.load(fh)
        functional = d.get("functional") or topologies
        print(f"functional subset: {len(functional)}")

    print("encoding graphs …")
    encoder = CircuitEncoder().to(args.device)
    emb_cache = precompute_embeddings(encoder, topologies, torch.device(args.device))

    fold_topos = topologies
    if args.max_folds > 0:
        fold_topos = topologies[:args.max_folds]

    folds = []
    for i, held in enumerate(fold_topos):
        print(f"fold {i+1}/{len(fold_topos)}: hold out {held}")
        folds.append(run_fold(held, topologies, entries, emb_cache,
                              args.switch_model, args.device, args.noise_floor))

    # Aggregate over folds
    agg = {}
    for sel in ("random", "maj_regime", "spec_mlp_39", "spec_mlp_40", "graph_ranker"):
        means = [f["summary"][sel]["mean"] for f in folds
                 if f["summary"][sel]["mean"] is not None]
        p90s = [f["summary"][sel]["p90"] for f in folds
                if f["summary"][sel]["p90"] is not None]
        agg[sel] = {
            "mean_of_means": float(np.mean(means)) if means else None,
            "mean_of_p90": float(np.mean(p90s)) if p90s else None,
            "noise_floor_p95": args.noise_floor,
        }

    # Functional-only folds
    func_folds = [f for f in folds if f["held_out"] in functional]
    agg_func = {}
    for sel in ("random", "maj_regime", "spec_mlp_39", "spec_mlp_40", "graph_ranker"):
        means = [f["summary"][sel]["mean"] for f in func_folds
                 if f["summary"][sel]["mean"] is not None]
        agg_func[sel] = {
            "mean_of_means": float(np.mean(means)) if means else None,
            "n_folds": len(func_folds),
            "noise_floor_p95": args.noise_floor,
        }

    out = {
        "switch_model": args.switch_model,
        "n_topologies": len(topologies),
        "n_folds": len(folds),
        "aggregate_all": agg,
        "aggregate_functional": agg_func,
        "folds": folds,
        "claim": (
            "On a held-out topology the spec-only MLP is structurally undefined "
            "(no output head). The graph ranker is defined. Magnitudes sit beside "
            "the archive-vs-DE noise floor; differences below that floor are "
            "unresolvable."
        ),
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)
    print(json.dumps(agg, indent=2))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
