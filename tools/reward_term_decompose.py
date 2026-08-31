#!/usr/bin/env python3
"""D1: decompose r_sim into g_phase/il/rl/gain before (90°) vs after (T1)."""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from specset.schema import TRAIN_SPECSET_PATH, load_specset

from env.graph_utils import TOPOLOGY_PARAMS
from env.netlist_graph import nominal_params
from env.reward import WEIGHTS_T1, compute_sim_reward
from sim.mna_scorer import _states_for_topo, mna_evaluate


def _legacy_parts(metrics, targets):
    """Pre-T1 soft margins with phase denom fixed at 90°."""
    if metrics is None:
        return -5.0, {"sentinel": -5.0}
    required = ["rms_phase_err_deg", "il_db", "rl_db", "gain_err_db"]
    if any(metrics.get(k) is None for k in required):
        return -3.0, {"sentinel": -3.0}
    w = WEIGHTS_T1
    parts = {}
    m_phase = abs(float(metrics["rms_phase_err_deg"]))
    parts["g_phase"] = max(0.0, 1.0 - m_phase / 90.0)
    parts["phase_denom"] = 90.0
    t_il = max(float(targets.get("max_il_db", 5.0)), 0.1)
    m_il = float(metrics["il_db"])
    parts["g_il"] = 0.0 if m_il < 0.0 else max(0.0, 1.0 - m_il / t_il)
    t_rl = max(float(targets.get("min_rl_db", 10.0)), 1.0)
    m_rl = abs(float(metrics["rl_db"]))
    parts["g_rl"] = max(0.0, min(1.0, m_rl / t_rl))
    t_gain = max(float(targets.get("rms_gain_err_db", 1.0)), 0.1)
    m_gain = abs(float(metrics["gain_err_db"]))
    parts["g_gain"] = max(0.0, 1.0 - m_gain / t_gain)
    t_pwr = max(float(targets.get("pmax_mw", 15.0)), 0.1)
    m_pwr = metrics.get("pwr_mw")
    m_pwr = 0.0 if m_pwr is None else abs(float(m_pwr))
    parts["g_power"] = max(0.0, 1.0 - m_pwr / t_pwr)
    r = (
        w.phase * parts["g_phase"]
        + w.il * parts["g_il"]
        + w.rl * parts["g_rl"]
        + w.gain * parts["g_gain"]
        + w.power * parts["g_power"]
    )
    all_close = (
        m_phase <= 1.2 * max(float(targets.get("rms_phase_err_deg", 5.0)), 0.1)
        and 0.0 <= m_il <= 1.2 * t_il
        and m_rl >= 0.8 * t_rl
        and m_gain <= 1.2 * t_gain
        and m_pwr <= 1.2 * t_pwr
    )
    parts["all_close"] = all_close
    if all_close:
        r += 1.0
    parts["reward"] = float(np.clip(r, -1.0, 2.0))
    return parts["reward"], parts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--specset",
        default=TRAIN_SPECSET_PATH,
    )
    ap.add_argument("--bounds", default="electrical")
    ap.add_argument("--switch-model", default="ideal")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument(
        "--out",
        default=os.path.join(REPO_ROOT, "results", "joint", "reward_decompose.json"),
    )
    args = ap.parse_args()

    specs = load_specset(args.specset)
    if args.limit > 0:
        specs = specs[: args.limit]

    topos = list(TOPOLOGY_PARAMS.keys())
    n_states = {t: len(_states_for_topo(t)) for t in topos}
    print(f"D1: {len(specs)} specs × {len(topos)} topos; n_states={n_states}")

    sum_b = {t: defaultdict(float) for t in topos}
    sum_a = {t: defaultdict(float) for t in topos}
    counts = {t: 0 for t in topos}
    win_b = defaultdict(int)
    win_a = defaultdict(int)
    keys = ("g_phase", "g_il", "g_rl", "g_gain", "reward")

    for i, entry in enumerate(specs):
        spec = entry.get("spec", entry)
        scores_b, scores_a = {}, {}
        parts_b_row, parts_a_row = {}, {}
        for topo in topos:
            params = nominal_params(
                topo, spec, bounds=args.bounds, switch_model=args.switch_model,
            )
            _, agg = mna_evaluate(topo, params, spec)
            if agg is None:
                continue
            rb, pb = _legacy_parts(agg, spec)
            ra, pa = compute_sim_reward(agg, spec, return_parts=True)
            scores_b[topo] = rb
            scores_a[topo] = ra
            parts_b_row[topo] = pb
            parts_a_row[topo] = pa
            counts[topo] += 1
            for k in keys:
                if k in pb:
                    sum_b[topo][k] += float(pb[k])
                if k in pa:
                    sum_a[topo][k] += float(pa[k])
        if scores_b:
            win_b[max(scores_b, key=scores_b.get)] += 1
            win_a[max(scores_a, key=scores_a.get)] += 1
        if (i + 1) % 100 == 0 or i == 0:
            print(f"  [{i+1}/{len(specs)}]")

    mean_b = {
        t: {k: (sum_b[t][k] / counts[t] if counts[t] else None) for k in keys}
        for t in topos
    }
    mean_a = {
        t: {k: (sum_a[t][k] / counts[t] if counts[t] else None) for k in keys}
        for t in topos
    }

    # Which term explains the winner's lead after T1?
    sf_b = mean_b.get("Switched_Filter", {})
    sf_a = mean_a.get("Switched_Filter", {})

    out = {
        "n_specs": len(specs),
        "switch_model": args.switch_model,
        "bounds": args.bounds,
        "n_states_scored": n_states,
        "counts": counts,
        "mean_parts_before": mean_b,
        "mean_parts_after": mean_a,
        "winners_before": dict(win_b),
        "winners_after": dict(win_a),
        "switched_filter_g_phase_before": sf_b.get("g_phase"),
        "switched_filter_g_phase_after": sf_a.get("g_phase"),
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)

    print("\n=== D1 mean parts AFTER (T1) ===")
    for t in topos:
        m = mean_a[t]
        print(
            f"{t:28s} g_phase={m['g_phase']:.3f} g_il={m['g_il']:.3f} "
            f"g_rl={m['g_rl']:.3f} g_gain={m['g_gain']:.3f} r={m['reward']:.3f}"
        )
    print("winners_before", dict(win_b))
    print("winners_after", dict(win_a))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
