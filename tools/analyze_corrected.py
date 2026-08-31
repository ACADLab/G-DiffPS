"""Compare matrix comparison.csv against AppC_gnn_fix.csv."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from tools.make_tables import loocv_from_csv, wilson_ci


def load_appc(path):
    rows = {}
    with open(path) as fh:
        for r in csv.DictReader(fh):
            rows[(r["topology"], r.get("config", "gin"))] = r
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corrected", default=os.path.join(
        REPO_ROOT, "results/matrix_corrected/comparison.csv"))
    ap.add_argument("--fixed28", default=os.path.join(
        REPO_ROOT, "results/matrix_fixed28_control/comparison.csv"))
    ap.add_argument("--appc", default=os.path.join(
        REPO_ROOT, "results/paper_tables/AppC_gnn_fix.csv"))
    ap.add_argument("--joint", default=os.path.join(
        REPO_ROOT, "results/joint/topo_select_mna.json"))
    ap.add_argument("--out", default=os.path.join(
        REPO_ROOT, "results/tables/corrected_vs_appc.json"))
    args = ap.parse_args()

    appc = load_appc(args.appc) if os.path.exists(args.appc) else {}
    report = {"corrected": [], "fixed28_control": [], "joint": None}

    if os.path.exists(args.corrected):
        for r in loocv_from_csv(args.corrected):
            key = (r["topology"], "gin")  # AppC is gin baseline
            base = appc.get(key) or appc.get((r["topology"], r.get("config", "")))
            # Prefer matching AppC row by topology only if single config
            if base is None:
                cands = [v for (t, c), v in appc.items() if t == r["topology"]]
                base = cands[0] if cands else None
            entry = {
                "topology": r["topology"],
                "config": r.get("config"),
                "success_rate": float(r["success_rate_mean"]),
                "ci95": [100 * r["ci95_lo"], 100 * r["ci95_hi"]],
                "best_reward_mean": float(r.get("best_reward_mean", 0) or 0),
                "fc_mode": r.get("fc_mode"),
                "bounds": r.get("bounds"),
            }
            if base:
                entry["appc_success_rate"] = float(base["success_rate_mean"])
                entry["delta_pp"] = entry["success_rate"] - entry["appc_success_rate"]
            report["corrected"].append(entry)

    if os.path.exists(args.fixed28):
        for r in loocv_from_csv(args.fixed28):
            cands = [v for (t, c), v in appc.items() if t == r["topology"]]
            base = cands[0] if cands else None
            entry = {
                "topology": r["topology"],
                "config": r.get("config"),
                "success_rate": float(r["success_rate_mean"]),
                "ci95": [100 * r["ci95_lo"], 100 * r["ci95_hi"]],
            }
            if base:
                entry["appc_success_rate"] = float(base["success_rate_mean"])
                entry["delta_pp"] = entry["success_rate"] - entry["appc_success_rate"]
            report["fixed28_control"].append(entry)

    if os.path.exists(args.joint):
        with open(args.joint) as fh:
            j = json.load(fh)
        report["joint"] = {
            k: j[k] for k in (
                "n", "top1_accuracy", "top2_accuracy", "heuristic_accuracy",
                "chance", "appC_valuenet_ref", "beats_chance", "beats_appC",
            ) if k in j
        }

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(report, fh, indent=2)

    print("=== Corrected harness vs AppC ===")
    for e in report["corrected"]:
        d = e.get("delta_pp")
        d_s = f"{d:+.1f}pp" if d is not None else "n/a"
        print(f"{e['topology']:18s} {e['config']:16s} "
              f"{e['success_rate']:5.1f}%  AppCΔ {d_s}")
    if report["fixed28_control"]:
        print("\n=== fixed28 control vs AppC ===")
        for e in report["fixed28_control"]:
            d = e.get("delta_pp")
            d_s = f"{d:+.1f}pp" if d is not None else "n/a"
            print(f"{e['topology']:18s} {e['success_rate']:5.1f}%  AppCΔ {d_s}")
    if report["joint"]:
        print("\n=== Joint evaluate-all ===")
        print(json.dumps(report["joint"], indent=2))
    print(f"\nSaved -> {args.out}")


if __name__ == "__main__":
    main()
