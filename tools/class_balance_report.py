"""S3 gate: report ALL SIX heuristic class shares, not just the loudest one.

The +8 experiment showed Switched_Filter's share falling 35.3% -> 20.6% -> 9.1%
and that was read as "the imbalance flattened". It did not: uniform over six is
16.7%, so 9.1% is *below* chance and the mass moved somewhere else. This tool
reports every class under both scorers and applies the gate explicitly.

Gate: if any class share exceeds HI_FRAC or falls below LO_FRAC, the pool is
still degenerate and S3 (margin stratification) is required.

Two scorers:
  original    — score_topology as shipped (hand-tuned bonus magnitudes)
  normalized  — every bonus/penalty collapsed to +-1, so a topology can only win
                by satisfying *more* rules, not by owning one large bonus

Run: .venv/bin/python tools/class_balance_report.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from specset.phaseshifter_scoring import TOPOLOGY_LABELS
from specset.schema import load_specset

N_TOPO = len(TOPOLOGY_LABELS)
UNIFORM = 1.0 / N_TOPO
HI_FRAC = 0.30
LO_FRAC = 0.05


def score_normalized(topology: str, spec: dict) -> float:
    """score_topology with every bonus magnitude collapsed to +-1.

    Mirrors the rule *structure* of specset.phaseshifter_scoring exactly; only
    the magnitudes change. Keep in sync if the rules there change.
    """
    s = 0.0
    fc = spec["fc_ghz"]
    bw = spec["bw_pct"]
    bits = spec["phase_bits"]
    pwr = spec["pmax_mw"]
    cov = spec["phase_coverage_deg"]
    tech = spec["tech"]

    if topology == "Switched_Line":
        s += (bits >= 4) - (fc > 20) - (fc < 5) + (bw > 30)
    elif topology == "Loaded_Line":
        s += (bits == 0) - (bw > 25) + (fc < 10) - (cov > 180)
    elif topology == "Reflection_Type":
        s += (fc > 15) + (bw > 30) - (pwr < 5)
    elif topology == "Switched_Filter":
        s += (bits >= 4 and bw > 40) + (fc < 6) - (fc > 20)
    elif topology == "Vector_Modulator":
        s += (bits >= 5) + (cov >= 360) - (pwr < 10) + (bw > 30)
    elif topology == "All_Pass":
        s += (bw > 50) + (bits == 0) - (cov > 180)

    if tech == 2 and topology in ("Switched_Line", "Reflection_Type"):
        s += 1
    return float(s)


def shares(specs: list[dict], scorer) -> dict:
    counts = Counter()
    for e in specs:
        spec = e["spec"]
        scores = {t: scorer(t, spec) for t in TOPOLOGY_LABELS}
        counts[max(scores, key=scores.get)] += 1
    n = max(len(specs), 1)
    return {
        t: {"n": counts.get(t, 0), "frac": counts.get(t, 0) / n}
        for t in TOPOLOGY_LABELS
    }


def evaluate_gate(sh: dict) -> dict:
    over = {t: v["frac"] for t, v in sh.items() if v["frac"] > HI_FRAC}
    under = {t: v["frac"] for t, v in sh.items() if v["frac"] < LO_FRAC}
    fracs = [v["frac"] for v in sh.values()]
    return {
        "over_hi": over,
        "under_lo": under,
        "max_frac": max(fracs),
        "min_frac": min(fracs),
        "spread": max(fracs) - min(fracs),
        # Total variation distance from uniform: 0 = perfectly balanced.
        "tv_from_uniform": 0.5 * sum(abs(f - UNIFORM) for f in fracs),
        "degenerate": bool(over or under),
    }


def _print_block(title: str, sh: dict, gate: dict) -> None:
    print(f"\n--- {title} ---")
    print(f"  {'topology':20s} {'n':>7s} {'share':>8s}  vs uniform {UNIFORM:.1%}")
    for t, v in sorted(sh.items(), key=lambda kv: -kv[1]["frac"]):
        mark = ""
        if v["frac"] > HI_FRAC:
            mark = f"  <-- ABOVE {HI_FRAC:.0%}"
        elif v["frac"] < LO_FRAC:
            mark = f"  <-- BELOW {LO_FRAC:.0%}"
        print(f"  {t:20s} {v['n']:7d} {v['frac']:8.2%}{mark}")
    print(f"  spread={gate['spread']:.3f}  TV(uniform)={gate['tv_from_uniform']:.3f}"
          f"  degenerate={gate['degenerate']}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train", default=os.path.join(
        REPO_ROOT, "specset", "specset_train.json"))
    ap.add_argument("--eval", dest="eval_path", default=os.path.join(
        REPO_ROOT, "specset", "specset_eval.json"))
    ap.add_argument("--out", default=os.path.join(
        REPO_ROOT, "results", "joint", "class_balance.json"))
    args = ap.parse_args()

    from specset.phaseshifter_scoring import score_topology

    report = {
        "n_topologies": N_TOPO,
        "uniform_frac": UNIFORM,
        "gate": {"hi_frac": HI_FRAC, "lo_frac": LO_FRAC},
        "pools": {},
    }
    any_degenerate = False
    for label, path in (("train", args.train), ("eval", args.eval_path)):
        if not os.path.exists(path):
            print(f"[skip] {label}: {path} not found")
            continue
        specs = load_specset(path)
        block = {}
        for sname, scorer in (("original", score_topology),
                              ("normalized_pm1", score_normalized)):
            sh = shares(specs, scorer)
            gate = evaluate_gate(sh)
            block[sname] = {"shares": sh, "gate": gate}
            any_degenerate = any_degenerate or gate["degenerate"]
            _print_block(f"{label} / {sname} (n={len(specs)})", sh, gate)
        report["pools"][label] = block

    report["s3_required"] = any_degenerate
    report["s3_decision"] = (
        "S3 REQUIRED: at least one class is outside "
        f"[{LO_FRAC:.0%}, {HI_FRAC:.0%}] — margin stratification is needed so "
        "selector performance is reported per difficulty stratum rather than "
        "against a prior that a constant predictor can exploit."
        if any_degenerate else
        "S3 not required: every class share falls inside "
        f"[{LO_FRAC:.0%}, {HI_FRAC:.0%}]."
    )
    print(f"\n=== S3 GATE ===\n  {report['s3_decision']}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(report, fh, indent=2)
    print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
