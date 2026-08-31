#!/usr/bin/env python3
"""T5 - topology selection evaluated by regret against the achievable envelope.

Accuracy is the wrong headline for selection. Picking the second-best topology
when it trails by 0.004 is not the same error as picking a topology that cannot
meet the spec at all, and top-1 accuracy scores them identically. Regret against
`r_star` -- the certified-achievable reward per (spec, topology, switch_model)
produced by `tools/compute_envelope.py` -- separates them.

    regret(sel)       = r_star.best - r_star.per_topology[sel]
    normalized regret = regret / (r_star.best - min_tau r_star.per_topology)

Because `r_star` is precomputed and stored in the pool, this runs without MNA.

Selectors:

  random      expected regret, computed analytically over the topology set rather
              than sampled, so it carries no seed variance
  majority    constant "always pick the most frequent winner", fit on the train
              pool. Random is the wrong contrast when one class dominates: under
              `realistic` Vector_Modulator wins ~71% of specs outright, so a
              constant selector is already strong and is the baseline a learned
              selector actually has to beat.
  maj_regime  constant *within regime*. Vector_Modulator is active and gated on
              `pmax_mw`, which partitions the pool cleanly -- it wins ~91% of
              active-allowed specs and exactly 0% of the rest. A flat six-way
              average hides that these are two different decisions, so this
              baseline concedes the partition and asks what is left.
  heuristic   argmax of `score_topology`, read from the pool's stored
              `heuristic_scores` (retired as ground truth in S0, kept as a baseline)
  spec_mlp    spec-only MLP on s -- the ablation that asks whether the graph
              encoder earns its place, since this sees no topology structure
  rank_head   R_xi(s, z_tau) from T4; skipped with a notice until T4 lands

`--switch-model` is required with no default: `r_star` differs per switch model
and an unlabelled number is not interpretable.

Results are stratified by area_rank, frequency band, and r_star margin stratum.
An unstratified mean hides that area only binds below ~10 GHz and that the
boundary stratum is where selection is actually hard.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from env.netlist_graph import VM_IQ
from specset.phaseshifter_scoring import TOPOLOGY_LABELS
from specset.schema import SPEC_DIM, normalize_spec

SWITCH_MODELS = ("ideal", "realistic")
FC_BANDS = (("sub10", 0.0, 10.0), ("10to20", 10.0, 20.0), ("20plus", 20.0, 1e9))

# Vector_Modulator is the only active topology and `check_physics_priors` gates
# it on the VCVS drive proxy 5*((G_I*scale)^2 + (G_Q*scale)^2), worst-case over
# the 16-state I/Q table. `G_{I,Q}_scale` bottoms out at 0.7 in action_to_params,
# so the least drive any sizing can present is the constant below -- a spec with
# `pmax_mw` under it can never admit VM, whatever the sizing.
#
# Must track the 0.7 lower bound in action_to_params; asserted in main().
G_SCALE_MIN = 0.7
VM_MIN_DRIVE_MW = 5.0 * (G_SCALE_MIN ** 2) * max(gi * gi + gq * gq for gi, gq in VM_IQ)


def regime(spec: dict) -> str:
    """Active-allowed vs passive-only, the partition VM's power gate induces."""
    return ("active_allowed" if float(spec["pmax_mw"]) >= VM_MIN_DRIVE_MW
            else "passive_only")


# ---------------------------------------------------------------------------
# pool loading
# ---------------------------------------------------------------------------

def load_pool(path: str) -> tuple[list[dict], dict]:
    with open(path) as f:
        doc = json.load(f)
    if isinstance(doc, dict) and "specs" in doc:
        return doc["specs"], doc
    return doc, {}


def entries_with_rstar(entries: list[dict], switch_model: str) -> list[dict]:
    out = []
    for e in entries:
        rs = (e.get("r_star") or {}).get(switch_model)
        if not rs or not rs.get("per_topology"):
            continue
        if any(rs["per_topology"].get(t) is None for t in TOPOLOGY_LABELS):
            continue
        out.append(e)
    return out


def fc_band(fc_ghz: float) -> str:
    for name, lo, hi in FC_BANDS:
        if lo <= fc_ghz < hi:
            return name
    return FC_BANDS[-1][0]


# ---------------------------------------------------------------------------
# selectors
# ---------------------------------------------------------------------------

def heuristic_scores(entry: dict) -> np.ndarray | None:
    hs = entry.get("heuristic_scores")
    if not hs:
        return None
    return np.array([float(hs.get(t, -1e9)) for t in TOPOLOGY_LABELS], dtype=float)


def win_counts(entries, switch_model, subset=None) -> np.ndarray:
    """Train-pool win frequency per topology, as a constant score vector.

    Used as the ranking for the constant baselines, so they get a well-defined
    top-2 rather than only a top-1 pick.
    """
    counts = np.zeros(len(TOPOLOGY_LABELS), dtype=float)
    for e in entries:
        if subset is not None and regime(e["spec"]) != subset:
            continue
        rs = e["r_star"][switch_model]
        vals = [rs["per_topology"][t] for t in TOPOLOGY_LABELS]
        counts[int(np.argmax(vals))] += 1.0
    return counts


def majority_scores(train_entries, eval_entries, switch_model, per_regime: bool):
    """Constant score vector, globally or within each regime."""
    if not per_regime:
        c = win_counts(train_entries, switch_model)
        return [c] * len(eval_entries), {
            "pick": TOPOLOGY_LABELS[int(np.argmax(c))],
            "train_share": float(c.max() / max(c.sum(), 1.0)),
        }
    by_regime = {r: win_counts(train_entries, switch_model, subset=r)
                 for r in ("active_allowed", "passive_only")}
    scores = [by_regime[regime(e["spec"])] for e in eval_entries]
    info = {r: {"pick": TOPOLOGY_LABELS[int(np.argmax(c))],
                "train_share": float(c.max() / max(c.sum(), 1.0))}
            for r, c in by_regime.items()}
    return scores, info


def train_spec_mlp(train_entries, switch_model, seed=0, epochs=40, device="cpu"):
    """Spec-only MLP: s -> topology. Target is argmax_tau r_star."""
    import torch
    import torch.nn as nn

    torch.manual_seed(seed)
    X, y = [], []
    for e in train_entries:
        rs = e["r_star"][switch_model]
        vals = [rs["per_topology"][t] for t in TOPOLOGY_LABELS]
        X.append(normalize_spec(e["spec"]))
        y.append(int(np.argmax(vals)))
    X = torch.tensor(np.asarray(X), dtype=torch.float32, device=device)
    y = torch.tensor(np.asarray(y), dtype=torch.long, device=device)

    n_val = max(1, int(0.1 * len(X)))
    perm = torch.randperm(len(X), generator=torch.Generator().manual_seed(seed))
    vi, ti = perm[:n_val], perm[n_val:]

    net = nn.Sequential(
        nn.Linear(SPEC_DIM, 64), nn.ReLU(),
        nn.Linear(64, 64), nn.ReLU(),
        nn.Linear(64, len(TOPOLOGY_LABELS)),
    ).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    lossf = nn.CrossEntropyLoss()

    bs = 256
    best_val, best_state = -1.0, None
    for _ in range(epochs):
        net.train()
        idx = ti[torch.randperm(len(ti))]
        for k in range(0, len(idx), bs):
            b = idx[k:k + bs]
            opt.zero_grad()
            lossf(net(X[b]), y[b]).backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            acc = (net(X[vi]).argmax(1) == y[vi]).float().mean().item()
        if acc > best_val:
            best_val = acc
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
    if best_state is not None:
        net.load_state_dict(best_state)
    return net, best_val


def spec_mlp_scores(net, entries, device="cpu") -> np.ndarray:
    import torch
    X = torch.tensor(np.asarray([normalize_spec(e["spec"]) for e in entries]),
                     dtype=torch.float32, device=device)
    net.eval()
    with torch.no_grad():
        return net(X).cpu().numpy()


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------

def regret_rows(entries, switch_model, scores_by_selector):
    """One row per (selector, spec): regret, normalized regret, top1, top2."""
    rows = []
    for i, e in enumerate(entries):
        rs = e["r_star"][switch_model]
        vals = np.array([rs["per_topology"][t] for t in TOPOLOGY_LABELS], dtype=float)
        best, worst = float(vals.max()), float(vals.min())
        span = best - worst
        truth = int(np.argmax(vals))
        band = fc_band(float(e["spec"]["fc_ghz"]))
        rank = e.get("area_rank")
        stratum = rs.get("stratum")

        for sel, scores in scores_by_selector.items():
            if sel == "random":
                # analytic expectation over a uniform pick
                regret = best - float(vals.mean())
                top1, top2 = 1.0 / len(vals), 2.0 / len(vals)
            else:
                s = scores[i]
                if s is None:
                    continue
                order = np.argsort(-np.asarray(s, dtype=float))
                pick = int(order[0])
                regret = best - float(vals[pick])
                top1 = float(pick == truth)
                top2 = float(truth in order[:2])
            rows.append({
                "selector": sel,
                "regret": float(regret),
                "norm_regret": float(regret / span) if span > 1e-12 else 0.0,
                "top1": top1,
                "top2": top2,
                "fc_band": band,
                "area_rank": rank,
                "margin_stratum": stratum,
                "regime": regime(e["spec"]),
            })
    return rows


def summarize(rows, key=None):
    """Aggregate rows by selector, optionally within one stratum key."""
    out = {}
    sels = sorted({r["selector"] for r in rows})
    for sel in sels:
        sub = [r for r in rows if r["selector"] == sel]
        if not sub:
            continue
        reg = np.array([r["regret"] for r in sub])
        nrg = np.array([r["norm_regret"] for r in sub])
        out[sel] = {
            "n": len(sub),
            "regret_mean": float(reg.mean()),
            "regret_p90": float(np.percentile(reg, 90)),
            "norm_regret_mean": float(nrg.mean()),
            "top1": float(np.mean([r["top1"] for r in sub])),
            "top2": float(np.mean([r["top2"] for r in sub])),
        }
    return out


def stratified(rows, field):
    groups = {}
    for r in rows:
        v = r.get(field)
        if v is None:
            continue
        groups.setdefault(str(v), []).append(r)
    return {k: summarize(v) for k, v in sorted(groups.items())}


# ---------------------------------------------------------------------------

def print_table(title, table, order=None):
    print(f"\n{title}")
    print(f"  {'selector':<12} {'n':>6} {'regret':>9} {'p90':>9} "
          f"{'norm':>8} {'top1':>7} {'top2':>7}")
    keys = order or sorted(table.keys())
    for sel in keys:
        if sel not in table:
            continue
        d = table[sel]
        print(f"  {sel:<12} {d['n']:>6} {d['regret_mean']:>9.4f} "
              f"{d['regret_p90']:>9.4f} {d['norm_regret_mean']:>8.4f} "
              f"{100*d['top1']:>6.1f}% {100*d['top2']:>6.1f}%")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--switch-model", required=True, choices=SWITCH_MODELS,
                    help="required: r_star differs per switch model")
    ap.add_argument("--pool", default=os.path.join(REPO_ROOT, "specset", "specset_eval.json"))
    ap.add_argument("--train-pool", default=os.path.join(REPO_ROOT, "specset", "specset_train.json"))
    ap.add_argument("--rank-head", default=None,
                    help="T4 RankHead checkpoint; skipped if omitted or missing")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--pass-label", default="unlabelled",
                    help="which plan pass this is: after-T1, after-T1.5, after-T4")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    sm = args.switch_model
    eval_all, _ = load_pool(args.pool)
    eval_entries = entries_with_rstar(eval_all, sm)
    if not eval_entries:
        print(f"error: no eval entries carry r_star['{sm}'] in {args.pool}")
        return 1
    print(f"eval pool  {args.pool}: {len(eval_entries)}/{len(eval_all)} usable")

    scores = {"random": None}

    hs = [heuristic_scores(e) for e in eval_entries]
    if all(h is not None for h in hs):
        scores["heuristic"] = hs
    else:
        print("note: heuristic_scores missing from pool; skipping that selector")

    n_active = sum(1 for e in eval_entries if regime(e["spec"]) == "active_allowed")
    print(f"regime split: active_allowed {n_active}/{len(eval_entries)} "
          f"({100*n_active/len(eval_entries):.1f}%), "
          f"VM drive floor {VM_MIN_DRIVE_MW:.2f} mW")

    train_all, _ = load_pool(args.train_pool)
    train_entries = entries_with_rstar(train_all, sm)
    majority_info = {}
    if train_entries:
        print(f"train pool {args.train_pool}: {len(train_entries)} usable")

        s, info = majority_scores(train_entries, eval_entries, sm, per_regime=False)
        scores["majority"] = s
        majority_info["majority"] = info
        print(f"           majority   -> always {info['pick']} "
              f"({100*info['train_share']:.1f}% of train)")

        s, info = majority_scores(train_entries, eval_entries, sm, per_regime=True)
        scores["maj_regime"] = s
        majority_info["maj_regime"] = info
        for r, d in info.items():
            print(f"           maj_regime -> {r}: {d['pick']} "
                  f"({100*d['train_share']:.1f}% of that regime)")

        net, val_acc = train_spec_mlp(train_entries, sm, seed=args.seed,
                                      epochs=args.epochs)
        print(f"           spec_mlp val top-1 accuracy {100*val_acc:.1f}%")
        scores["spec_mlp"] = list(spec_mlp_scores(net, eval_entries))
    else:
        print("note: no usable train pool; skipping majority/spec_mlp")

    if args.rank_head and os.path.exists(args.rank_head):
        print(f"note: rank head at {args.rank_head} -- loader not implemented "
              f"until T4 lands; skipping")
    else:
        print("note: rank_head selector skipped (T4 not landed)")

    rows = regret_rows(eval_entries, sm, scores)
    order = ["random", "majority", "maj_regime", "heuristic", "spec_mlp", "rank_head"]

    overall = summarize(rows)
    print(f"\n{'='*66}\nT5 topology selection -- switch_model={sm}, "
          f"pass={args.pass_label}\n{'='*66}")
    print_table("OVERALL (lower regret is better; regret is the headline)",
                overall, order)

    by_regime = stratified(rows, "regime")
    for r in ("active_allowed", "passive_only"):
        if r in by_regime:
            print_table(f"regime = {r}", by_regime[r], order)

    by_band = stratified(rows, "fc_band")
    for band in ("sub10", "10to20", "20plus"):
        if band in by_band:
            print_table(f"fc band = {band}", by_band[band], order)

    by_rank = stratified(rows, "area_rank")
    for rank in sorted(by_rank, key=lambda x: int(x)):
        print_table(f"area_rank = {rank}", by_rank[rank], order)

    by_margin = stratified(rows, "margin_stratum")
    for st in ("boundary", "medium", "easy"):
        if st in by_margin:
            print_table(f"margin stratum = {st}", by_margin[st], order)

    result = {
        "switch_model": sm,
        "pass_label": args.pass_label,
        "pool": args.pool,
        "n_specs": len(eval_entries),
        "selectors": sorted(scores.keys()),
        "vm_min_drive_mw": VM_MIN_DRIVE_MW,
        "majority_info": majority_info,
        "overall": overall,
        "by_regime": by_regime,
        "by_fc_band": by_band,
        "by_area_rank": by_rank,
        "by_margin_stratum": by_margin,
    }
    out = args.out or os.path.join(
        REPO_ROOT, "results", "joint", f"topology_select_eval_{sm}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
