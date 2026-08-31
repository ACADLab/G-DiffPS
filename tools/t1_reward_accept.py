#!/usr/bin/env python3
"""T1 acceptance: template-default MNA scores over a spec pool.

Under switch_model=ideal, score each topology at nominal (mid-action) sizing
with the shared compute_sim_reward. Compare against the pre-T1 fixed
90°-denominator phase term.

Reports both:
  - full r_sim spread (includes +1.0 all_close bonus)
  - continuous-part spread (sum of w_m * g_m only)

The original gate (median full spread < 0.15) is unreachable at template
defaults whenever topologies differ in compliance — the discrete +1.0 step
dominates. See results/joint/s1_probe.json and the restated criterion in
results/joint/t1_reward_accept_v2.json.
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

from env.graph_utils import TOPOLOGY_PARAMS
from env.netlist_graph import nominal_params
from env.reward import WEIGHTS_AREA, WEIGHTS_T1, compute_sim_reward
from sim.mna_scorer import mna_evaluate, score_from_metrics
from specset.schema import SCHEMA_VERSION, load_specset


def _legacy_score(metrics, targets) -> float:
    """Pre-T1 reward: phase denom fixed at 90° (everything else matches T1)."""
    if metrics is None:
        return -5.0
    required = ["rms_phase_err_deg", "il_db", "rl_db", "gain_err_db"]
    if any(metrics.get(k) is None for k in required):
        return -3.0
    w = WEIGHTS_T1
    r = 0.0
    m_phase = abs(float(metrics["rms_phase_err_deg"]))
    r += w.phase * max(0.0, 1.0 - m_phase / 90.0)
    t_il = max(float(targets.get("max_il_db", 5.0)), 0.1)
    m_il = float(metrics["il_db"])
    r += w.il * (0.0 if m_il < 0.0 else max(0.0, 1.0 - m_il / t_il))
    t_rl = max(float(targets.get("min_rl_db", 10.0)), 1.0)
    m_rl = abs(float(metrics["rl_db"]))
    r += w.rl * max(0.0, min(1.0, m_rl / t_rl))
    t_gain = max(float(targets.get("rms_gain_err_db", 1.0)), 0.1)
    m_gain = abs(float(metrics["gain_err_db"]))
    r += w.gain * max(0.0, 1.0 - m_gain / t_gain)
    t_pwr = max(float(targets.get("pmax_mw", 15.0)), 0.1)
    m_pwr = metrics.get("pwr_mw")
    m_pwr = 0.0 if m_pwr is None else abs(float(m_pwr))
    r += w.power * max(0.0, 1.0 - m_pwr / t_pwr)
    all_close = (
        m_phase <= 1.2 * max(float(targets.get("rms_phase_err_deg", 5.0)), 0.1)
        and 0.0 <= m_il <= 1.2 * t_il
        and m_rl >= 0.8 * t_rl
        and m_gain <= 1.2 * t_gain
        and m_pwr <= 1.2 * t_pwr
    )
    if all_close:
        r += 1.0
    return float(np.clip(r, -1.0, 2.0))


def _continuous_reward(metrics, targets) -> float:
    """Weighted soft margins only — no all_close bonus."""
    if metrics is None:
        return -5.0
    r, parts = compute_sim_reward(
        metrics, targets, weights=WEIGHTS_AREA, return_parts=True,
    )
    if parts.get("sentinel") is not None:
        return float(r)
    # Reconstruct continuous sum from parts.
    cont = 0.0
    for key, w in (
        ("g_phase", WEIGHTS_AREA.phase),
        ("g_il", WEIGHTS_AREA.il),
        ("g_rl", WEIGHTS_AREA.rl),
        ("g_gain", WEIGHTS_AREA.gain),
        ("g_area", WEIGHTS_AREA.area),
    ):
        g = parts.get(key)
        if g is not None:
            cont += w * float(g)
    return float(cont)


def _summarize(spreads: list[float]) -> dict:
    arr = np.asarray(spreads, dtype=np.float64)
    return {
        "median": float(np.median(arr)),
        "mean": float(np.mean(arr)),
        "p90": float(np.percentile(arr, 90)),
        "frac_below_0_15": float(np.mean(arr < 0.15)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--specset",
        default=os.path.join(REPO_ROOT, "specset", "specset_eval.json"),
        help="Prefer the disjoint eval pool for acceptance; pin published "
             "reproductions to specset_v1_frozen.json with --expect-version 1.",
    )
    ap.add_argument("--expect-version", type=int, default=SCHEMA_VERSION)
    ap.add_argument("--out", default=os.path.join(
        REPO_ROOT, "results", "joint", "t1_reward_accept_v2.json"))
    ap.add_argument("--switch-model", default="ideal", choices=["ideal", "realistic"])
    ap.add_argument("--bounds", default="electrical")
    ap.add_argument("--limit", type=int, default=0, help="if >0, only first N specs")
    args = ap.parse_args()

    specs = load_specset(args.specset, expect_version=args.expect_version)
    if args.limit > 0:
        specs = specs[: args.limit]

    topos = list(TOPOLOGY_PARAMS.keys())
    print(f"T1 accept: {len(specs)} specs × {len(topos)} topos, switch={args.switch_model}")

    spreads_before, spreads_after, spreads_cont = [], [], []
    per_spec = []
    fail = 0

    for i, entry in enumerate(specs):
        spec = entry.get("spec", entry)
        sid = entry.get("id", f"idx_{i}")
        scores_b, scores_a, scores_c = {}, {}, {}
        for topo in topos:
            params = nominal_params(
                topo, spec, bounds=args.bounds, switch_model=args.switch_model,
            )
            score, agg = mna_evaluate(topo, params, spec)
            if agg is None:
                fail += 1
                scores_b[topo] = -5.0
                scores_a[topo] = -5.0
                scores_c[topo] = -5.0
            else:
                scores_a[topo] = float(score)
                scores_b[topo] = _legacy_score(agg, spec)
                scores_c[topo] = _continuous_reward(agg, spec)
                assert abs(
                    score_from_metrics(agg, spec)
                    - compute_sim_reward(agg, spec, weights=WEIGHTS_AREA)
                ) < 1e-9
        vb = list(scores_b.values())
        va = list(scores_a.values())
        vc = list(scores_c.values())
        sb = float(max(vb) - min(vb))
        sa = float(max(va) - min(va))
        sc = float(max(vc) - min(vc))
        spreads_before.append(sb)
        spreads_after.append(sa)
        spreads_cont.append(sc)
        per_spec.append({
            "id": sid,
            "spread_before": sb,
            "spread_after": sa,
            "spread_continuous": sc,
            "scores_before": scores_b,
            "scores_after": scores_a,
            "scores_continuous": scores_c,
        })
        if (i + 1) % 100 == 0 or i == 0:
            print(f"  [{i+1}/{len(specs)}] median full={np.median(spreads_after):.4f}  "
                  f"cont={np.median(spreads_cont):.4f}")

    full = _summarize(spreads_after)
    cont = _summarize(spreads_cont)
    out = {
        "n_specs": len(specs),
        "n_topos": len(topos),
        "topos": topos,
        "specset": args.specset,
        "switch_model": args.switch_model,
        "bounds": args.bounds,
        "mna_fail_count": fail,
        "spread_before": _summarize(spreads_before),
        "spread_after": full,
        "spread_continuous": cont,
        "accept_median_below_0_15": bool(full["median"] < 0.15),
        "accept_continuous_median_below_0_15": bool(cont["median"] < 0.15),
        "criterion_restatement": (
            "The original T1 gate (median cross-topology r_sim spread < 0.15 at "
            "template defaults) is unreachable: All_Pass is worst on ~95% of specs "
            "at nominal sizing while Switched_Filter often clears all_close (+1.0). "
            "S1 feasibility filtering does not close the gap (see s1_probe.json). "
            "Report continuous-part spread (no all_close) and prefer sized-envelope "
            "r* spreads (tools/t15_accept.py) for selection well-posedness."
        ),
        "per_spec": per_spec,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)

    print("\n=== T1 reward acceptance (v2) ===")
    print(f"spread FULL  (with all_close): median={full['median']:.4f}  "
          f"frac<0.15={full['frac_below_0_15']:.3f}")
    print(f"spread CONT  (no all_close):   median={cont['median']:.4f}  "
          f"frac<0.15={cont['frac_below_0_15']:.3f}")
    print(f"ACCEPT full<0.15: {out['accept_median_below_0_15']}  "
          f"cont<0.15: {out['accept_continuous_median_below_0_15']}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
