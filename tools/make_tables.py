"""Reconstruct Table 2 / Table 5-6 style summaries with binomial CIs.

Mean reward is averaged over SUCCESSFUL proposals only (paper protocol).
Tolerates malformed JSONL lines.

Published-number reproductions that need the original 600-spec pool must load
`specset/specset_v1_frozen.json` (schema v1), never the live v2 train file.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import defaultdict

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Pin for any future path that re-scores the published 600-spec set.
V1_FROZEN_SPECSET = os.path.join(REPO_ROOT, "specset", "specset_v1_frozen.json")
PAPER_TABLE2 = {
    "Switched_Line": {"yield": 100.0, "best": 2.26, "mean": 0.88},
    "Reflection_Type": {"yield": 100.0, "best": 2.23, "mean": 0.81},
    "Vector_Modulator": {"yield": 100.0, "best": 2.14, "mean": 0.69},
    "Loaded_Line": {"yield": 99.3, "best": 1.82, "mean": 0.65},
    "Switched_Filter": {"yield": 82.0, "best": 2.10, "mean": 0.73},
    "All_Pass": {"yield": 49.0, "best": 0.96, "mean": 0.57},
}


def load_jsonl(path):
    rows, bad = [], 0
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or '"step"' not in line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                bad += 1
    return rows, bad


def wilson_ci(k, n, z=1.96):
    """Wilson score interval for a binomial proportion."""
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = k / n
    denom = 1 + z ** 2 / n
    centre = (p + z ** 2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / denom
    return p, max(0.0, centre - half), min(1.0, centre + half)


def table2_from_log(log_path, final_n=4000):
    rows, bad = load_jsonl(log_path)
    if not rows:
        return {}, bad
    max_step = max(r["step"] for r in rows)
    lo = max(0, max_step - final_n + 1)
    sub = [r for r in rows if r["step"] >= lo]
    by = defaultdict(list)
    for r in sub:
        by[r["topology"]].append(r)

    out = {}
    for topo, rs in by.items():
        succ = [r for r in rs if r.get("success")]
        rewards_all = [r["reward"] for r in rs]
        rewards_ok = [r["reward"] for r in succ]
        y = 100.0 * len(succ) / len(rs) if rs else 0.0
        out[topo] = {
            "n": len(rs),
            "yield_pct": y,
            "best_reward": max(rewards_all) if rewards_all else None,
            "mean_reward_success": float(np.mean(rewards_ok)) if rewards_ok else None,
            "mean_reward_all": float(np.mean(rewards_all)) if rewards_all else None,
            "paper": PAPER_TABLE2.get(topo),
        }
    return out, bad


def loocv_from_csv(csv_path):
    """Read AppC-style or matrix comparison CSV and attach Wilson CIs."""
    rows = []
    with open(csv_path) as fh:
        reader = csv.DictReader(fh)
        for r in reader:
            n_seeds = int(float(r.get("n_seeds", 3)))
            # Approximate n_trials = 200 * n_seeds (paper protocol)
            n = 200 * n_seeds
            rate = float(r["success_rate_mean"]) / 100.0
            k = int(round(rate * n))
            p, lo, hi = wilson_ci(k, n)
            rows.append({
                **r,
                "compliance": p,
                "ci95_lo": lo,
                "ci95_hi": hi,
                "n_approx": n,
            })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default=os.path.join(
        REPO_ROOT, "checkpoints/run_20260530_031117/train.log"))
    ap.add_argument("--loocv-csv", default=os.path.join(
        REPO_ROOT, "results/paper_tables/AppC_gnn_fix.csv"))
    ap.add_argument("--out-dir", default=os.path.join(REPO_ROOT, "results", "tables"))
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    t2, bad = table2_from_log(args.log)
    t2_path = os.path.join(args.out_dir, "table2_replay.json")
    with open(t2_path, "w") as fh:
        json.dump({"unparseable_lines": bad, "topologies": t2}, fh, indent=2)
    print("=== Table 2 (success-only mean reward) ===")
    print(f"{'Topology':18s} {'Yield':>7s} {'Best':>6s} {'MeanOK':>7s}  paper")
    for topo, d in t2.items():
        p = d.get("paper") or {}
        print(f"{topo:18s} {d['yield_pct']:6.1f}% {d['best_reward']:6.2f} "
              f"{d['mean_reward_success']:7.2f}  "
              f"({p.get('yield')}, {p.get('best')}, {p.get('mean')})")
    print(f"Saved -> {t2_path}")

    if os.path.exists(args.loocv_csv):
        loocv = loocv_from_csv(args.loocv_csv)
        out_csv = os.path.join(args.out_dir, "loocv_with_ci.csv")
        with open(out_csv, "w", newline="") as fh:
            fields = list(loocv[0].keys()) if loocv else []
            w = csv.DictWriter(fh, fieldnames=fields)
            w.writeheader()
            w.writerows(loocv)
        print(f"\n=== LOOCV with Wilson 95% CI ===")
        for r in loocv:
            print(f"{r['topology']:18s} {r.get('config','?'):12s} "
                  f"{100*r['compliance']:5.1f}% "
                  f"[{100*r['ci95_lo']:5.1f}, {100*r['ci95_hi']:5.1f}]")
        print(f"Saved -> {out_csv}")


if __name__ == "__main__":
    main()
