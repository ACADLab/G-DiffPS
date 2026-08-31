"""Correlate MNA scores vs logged SPICE rewards from a train.log."""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from sim.mna_scorer import mna_score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default=os.path.join(
        REPO_ROOT, "checkpoints/run_20260530_031117/train.log"))
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--out", default=os.path.join(
        REPO_ROOT, "results/harness/mna_spice_corr.json"))
    args = ap.parse_args()

    # train.log has no per-step fc; paper meter was 28 GHz
    spec = {
        "fc_ghz": 28.0, "max_il_db": 5.0, "min_rl_db": 10.0,
        "rms_phase_err_deg": 5.0, "rms_gain_err_db": 1.0,
    }
    by_topo = {}
    with open(args.log) as fh:
        for line in fh:
            if '"params"' not in line or '"passed_prior": true' not in line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not rec.get("passed_prior"):
                continue
            topo = rec["topology"]
            reward = rec["reward"]
            params = rec["params"]
            # Strip expert bonus approx: paper rewards include bonus in [-0.1,0.3]
            # Compare rank correlation of raw MNA vs sim-ish reward
            mna = mna_score(topo, params, spec)
            by_topo.setdefault(topo, {"spice": [], "mna": []})
            by_topo[topo]["spice"].append(float(reward))
            by_topo[topo]["mna"].append(float(mna))
            total = sum(len(v["spice"]) for v in by_topo.values())
            if total >= args.n:
                break

    report = {"n_total": 0, "per_topology": {}}
    for topo, d in by_topo.items():
        sp = np.asarray(d["spice"])
        mn = np.asarray(d["mna"])
        if len(sp) < 5:
            corr = float("nan")
        else:
            corr = float(np.corrcoef(sp, mn)[0, 1])
            # Spearman
            rs = sp.argsort().argsort().astype(float)
            rm = mn.argsort().argsort().astype(float)
            spear = float(np.corrcoef(rs, rm)[0, 1])
        report["per_topology"][topo] = {
            "n": len(sp),
            "pearson": corr,
            "spearman": spear if len(sp) >= 5 else float("nan"),
            "spice_mean": float(sp.mean()),
            "mna_mean": float(mn.mean()),
        }
        report["n_total"] += len(sp)

    # Gate on IL correlation (S-params), not reward (contaminated by expert_bonus).
    # Loaded_Line / Switched_Line must show IL pearson > 0.9 vs logged SPICE metrics.
    from sim.mna_scorer import mna_evaluate as _mna_eval
    metric_gate = {}
    with open(args.log) as fh:
        buf = {t: [] for t in ("Loaded_Line", "Switched_Line")}
        for line in fh:
            for t in buf:
                if f'"topology": "{t}"' not in line or "passed_prior\": true" not in line:
                    continue
                line2 = line.replace(": NaN", ": null").replace(": nan", ": null")
                try:
                    rec = json.loads(line2)
                except json.JSONDecodeError:
                    continue
                if not rec.get("metrics"):
                    continue
                _, agg = _mna_eval(t, rec["params"], spec)
                if agg is None:
                    continue
                buf[t].append((rec["metrics"]["il_db"], agg["il_db"]))
            if all(len(v) >= 20 for v in buf.values()):
                break
    for t, pairs in buf.items():
        if len(pairs) < 5:
            metric_gate[t] = False
            continue
        a = np.asarray([p[0] for p in pairs])
        b = np.asarray([p[1] for p in pairs])
        pear = float(np.corrcoef(a, b)[0, 1])
        metric_gate[t] = bool(pear == pear and pear > 0.9)
        report["per_topology"].setdefault(t, {})["il_pearson_vs_spice"] = pear
    report["gate_pass"] = all(metric_gate.values()) if metric_gate else False
    report["gate"] = metric_gate

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(report, fh, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
