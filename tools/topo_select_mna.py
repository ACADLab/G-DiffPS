"""App C replacement: evaluate-all + MNA vs SPICE oracle (expert_bonus=0).

Reporting notes
---------------
Raw top-1 accuracy is not interpretable on this task. The spec prior induced by
SPEC_BOUNDS is strongly non-uniform over oracle-best topologies, so a predictor
that emits a single constant label scores at the majority-class rate. The
correct null is therefore the majority-class frequency, not 1/6, and the
headline number is balanced accuracy plus the within-spec rank correlation
against the SPICE oracle.

The oracle itself is noisy: it is an argmax over per-topology best-of-k SPICE
rewards, and adjacent topologies are frequently separated by less than the
sampling noise. Specs whose top-two oracle rewards differ by less than
``--margin`` are reported separately as undecidable rather than silently
counted as errors.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict

import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from env.phaseshifter_env import PhaseShifterEnv
from env.graph_utils import TOPOLOGY_PARAMS
from inference_joint import load_policy, evaluate_all
from sim.physics_priors import check_physics_priors
from train_diffusion import (
    action_to_params, device_action_to_params, make_spice_netlist,
    parallel_eval_worker,
)
from tools.loocv_eval import sample_action
from env.graph_utils import get_topology_graph
from specset.phaseshifter_scoring import score_topology, TOPOLOGY_LABELS
from specset.topo_aware_specs import sample_spec_for_topology

TOPOS = list(TOPOLOGY_PARAMS.keys())
COMPLIANCE = 0.5


def _rankdata(values: list[float]) -> np.ndarray:
    """Average ranks, ties shared (scipy.stats.rankdata equivalent)."""
    a = np.asarray(values, dtype=float)
    order = a.argsort()
    ranks = np.empty(len(a), dtype=float)
    ranks[order] = np.arange(1, len(a) + 1, dtype=float)
    # Average the ranks of tied groups
    for v in np.unique(a):
        mask = a == v
        if mask.sum() > 1:
            ranks[mask] = ranks[mask].mean()
    return ranks


def spearman(x: list[float], y: list[float]) -> float:
    """Spearman rho = Pearson correlation of ranks. NaN if either is constant."""
    rx, ry = _rankdata(x), _rankdata(y)
    if rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def spice_oracle(
    spec, gnn, actor, encoder, action_space, device, topo_graphs,
    k_per_topo=8, bounds="electrical", fc_mode="spec",
):
    """Empirical best topology by SPICE reward (expert_bonus=0).

    Returns (winner, per_topology_best, margin) where margin is the reward gap
    between the best and second-best topology. A small margin means the label
    is decided by sampling noise rather than by physics.
    """
    env = PhaseShifterEnv()
    env.current_spec = spec
    obs = env._normalize(spec)
    spec_norm = torch.tensor(obs, dtype=torch.float, device=device).unsqueeze(0)
    best = {}
    for topo in TOPOS:
        best_r = -1e9
        for _ in range(k_per_topo):
            a = sample_action(
                actor, gnn, topo, spec, spec_norm,
                encoder, action_space, device, topo_graphs,
            )
            if action_space == "device":
                params = device_action_to_params(a, topo, spec, bounds=bounds)
            else:
                params = action_to_params(a, topo, spec, bounds=bounds)
            if not check_physics_priors(topo, params, spec["fc_ghz"]):
                continue
            nl = make_spice_netlist(topo, params, spec_dict=spec, fc_mode=fc_mode)
            r, _, _ = parallel_eval_worker((nl, spec, topo, 0.0, [topo]))
            if r > best_r:
                best_r = float(r)
        best[topo] = best_r
    # Prefer feasible (reward > COMPLIANCE); else highest reward
    feasible = {t: r for t, r in best.items() if r > COMPLIANCE}
    pool = feasible if feasible else best
    winner = max(pool, key=pool.get)
    ordered = sorted(best.values(), reverse=True)
    margin = float(ordered[0] - ordered[1]) if len(ordered) > 1 else float("inf")
    return winner, best, margin


def heuristic_pick(spec):
    scores = {t: score_topology(t, spec) for t in TOPOLOGY_LABELS}
    return max(scores, key=scores.get)


def evaluate_spec(spec, gnn, actor, args, device, topo_graphs):
    """One spec: MNA ranking, heuristic pick, SPICE oracle, agreement stats."""
    ranked = evaluate_all(
        spec, gnn, actor,
        encoder=args.encoder, action_space=args.action_space,
        device=device, n_samples=args.n_samples,
    )
    pred = ranked[0]["topology"]
    top2 = {ranked[0]["topology"], ranked[1]["topology"]}
    oracle, spice_scores, margin = spice_oracle(
        spec, gnn, actor, args.encoder, args.action_space, device,
        topo_graphs, k_per_topo=args.k,
    )
    mna_by_topo = {r["topology"]: r["score"] for r in ranked}
    common = [t for t in TOPOS if t in mna_by_topo and t in spice_scores]
    rho = spearman(
        [mna_by_topo[t] for t in common],
        [spice_scores[t] for t in common],
    )
    return {
        "pred": pred,
        "oracle": oracle,
        "heuristic": heuristic_pick(spec),
        "top1": pred == oracle,
        "top2": oracle in top2,
        "spearman": rho,
        "oracle_margin": margin,
        "decidable": margin >= args.margin,
        "mna_ranking": [(r["topology"], r["score"]) for r in ranked],
        "spice_best": spice_scores,
        "ideal_models": sorted(
            r["topology"] for r in ranked
            if r["metrics"].get("any_active") or r["metrics"].get("all_rl_saturated")
        ),
        "fc_ghz": spec["fc_ghz"],
    }


def collect(gnn, actor, args, device, topo_graphs):
    """Draw specs and label them, optionally balancing on the oracle class.

    Balancing must stratify on the SPICE oracle rather than on the heuristic
    label, otherwise the eval set inherits the heuristic's own bias.

    With ``--topo-aware`` (requires ``--balance``), proposals come from
    ``sample_spec_for_topology`` instead of uniform ``env.reset`` draws.
    """
    env = PhaseShifterEnv()
    details = []
    if not args.balance:
        for i in range(args.n):
            env.reset()
            rec = evaluate_spec(dict(env.current_spec), gnn, actor, args, device, topo_graphs)
            rec["i"] = i
            details.append(rec)
            print(f"[{i+1}/{args.n}] fc={rec['fc_ghz']:.2f} pred={rec['pred']} "
                  f"oracle={rec['oracle']} margin={rec['oracle_margin']:.3f} "
                  f"top1={rec['top1']}", flush=True)
        return details

    per_class = max(1, args.n // len(TOPOS))
    buckets: dict[str, list] = defaultdict(list)
    drawn = 0
    rng = np.random.default_rng(args.seed)
    rr = 0
    while drawn < args.pool and sum(min(len(v), per_class) for v in buckets.values()) < args.n:
        if args.topo_aware:
            need = [t for t in TOPOS if len(buckets[t]) < per_class]
            if not need:
                break
            target = need[rr % len(need)]
            rr += 1
            spec = sample_spec_for_topology(target, rng)
        else:
            env.reset()
            spec = dict(env.current_spec)
        rec = evaluate_spec(spec, gnn, actor, args, device, topo_graphs)
        rec["i"] = drawn
        drawn += 1
        cls = rec["oracle"]
        kept = len(buckets[cls]) < per_class
        if kept:
            buckets[cls].append(rec)
        filled = {k: len(v) for k, v in sorted(buckets.items())}
        tag = f" target={target}" if args.topo_aware else ""
        print(f"[pool {drawn}/{args.pool}] oracle={cls}{tag} "
              f"{'kept' if kept else 'full'} filled={filled}", flush=True)
    for v in buckets.values():
        details.extend(v[:per_class])
    print(f"\nStratified set: {len(details)} specs from {drawn} draws; "
          f"per-class target {per_class}"
          f"{' (topo-aware proposals)' if args.topo_aware else ''}")
    return details


def summarize(details, args, warned):
    n = len(details)
    oracle_counts = Counter(d["oracle"] for d in details)
    pred_counts = Counter(d["pred"] for d in details)
    majority = max(oracle_counts.values()) / n if n else float("nan")

    per_class = {}
    for t in TOPOS:
        tot = oracle_counts.get(t, 0)
        if tot:
            per_class[t] = sum(d["top1"] for d in details if d["oracle"] == t) / tot
    balanced = float(np.mean(list(per_class.values()))) if per_class else float("nan")

    confusion = {
        o: dict(Counter(d["pred"] for d in details if d["oracle"] == o))
        for o in sorted(oracle_counts)
    }

    dec = [d for d in details if d["decidable"]]
    rhos = [d["spearman"] for d in details if not np.isnan(d["spearman"])]
    margins = [d["oracle_margin"] for d in details if np.isfinite(d["oracle_margin"])]
    ideal = Counter(t for d in details for t in d["ideal_models"])

    top1 = sum(d["top1"] for d in details) / n if n else float("nan")
    return {
        "n": n,
        "encoder": args.encoder,
        "action_space": args.action_space,
        "run": args.run,
        "k_per_topo": args.k,
        "balanced_sampling": bool(args.balance),
        "topo_aware_sampling": bool(getattr(args, "topo_aware", False)),
        "warned_random_weights": warned,

        # Headline
        "balanced_accuracy": balanced,
        "mean_within_spec_spearman": float(np.mean(rhos)) if rhos else float("nan"),

        # Raw accuracy alongside the only nulls that make it meaningful
        "top1_accuracy": top1,
        "top2_accuracy": sum(d["top2"] for d in details) / n if n else float("nan"),
        "heuristic_accuracy": (
            sum(d["heuristic"] == d["oracle"] for d in details) / n if n else float("nan")
        ),
        "majority_class_baseline": majority,
        "chance_uniform": 1.0 / len(TOPOS),
        "appC_valuenet_ref": 0.10,
        "beats_majority_baseline": bool(top1 > majority),
        "is_constant_predictor": len(pred_counts) == 1,

        # Label quality
        "decidable_n": len(dec),
        "decidable_fraction": len(dec) / n if n else float("nan"),
        "decidable_top1_accuracy": (
            sum(d["top1"] for d in dec) / len(dec) if dec else float("nan")
        ),
        "oracle_margin_mean": float(np.mean(margins)) if margins else float("nan"),
        "oracle_margin_median": float(np.median(margins)) if margins else float("nan"),
        "margin_threshold": args.margin,

        "per_class_recall": per_class,
        "confusion_oracle_to_pred": confusion,
        "pred_counts": dict(pred_counts),
        "oracle_counts": dict(oracle_counts),
        "nonphysical_model_counts": dict(ideal),
        "details": details,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--encoder", default="circuit")
    ap.add_argument("--action-space", default="device")
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--k", type=int, default=8,
                    help="SPICE samples per topology for the oracle (higher = less label noise)")
    ap.add_argument("--n-samples", type=int, default=1,
                    help="Policy samples per topology for the MNA prediction")
    ap.add_argument("--margin", type=float, default=0.10,
                    help="Min oracle top1-top2 reward gap for a spec to count as decidable")
    ap.add_argument("--balance", action="store_true",
                    help="Stratify the eval set to equal counts per oracle-best topology")
    ap.add_argument("--topo-aware", action="store_true",
                    help="With --balance, propose specs via topology-biased priors "
                         "instead of uniform env.reset draws")
    ap.add_argument("--pool", type=int, default=400,
                    help="Max specs to draw when --balance is set")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=os.path.join(REPO_ROOT, "results", "joint", "topo_select_mna.json"))
    args = ap.parse_args()

    if args.topo_aware and not args.balance:
        ap.error("--topo-aware requires --balance")

    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    gnn, actor, warned = load_policy(args.run, args.encoder, args.action_space, device)
    topo_graphs = {t: get_topology_graph(t) for t in TOPOS} if args.encoder == "gin" else None

    details = collect(gnn, actor, args, device, topo_graphs)
    summary = summarize(details, args, warned)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps({k: v for k, v in summary.items() if k != "details"}, indent=2))


if __name__ == "__main__":
    main()
