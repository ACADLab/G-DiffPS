"""Class shares from argmax_tau r_star, split by active-allowed regime.

Supersedes the `score_topology` argmax distribution for S3 purposes. That
heuristic was retired as ground truth in S0, so binning strata on it measured
the wrong thing: the class that matters is which topology actually wins the
envelope.

Reported separately for specs whose power budget admits an active stage and
those it does not. Presenting six topologies as one flat choice when one of
them needs a bias supply is a category error -- a system designer decides
"active or passive" at a different level than "which passive topology" -- and
stratifying dissolves the confound instead of hiding it.

Run: .venv/bin/python tools/envelope_class_shares.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from env.graph_utils import TOPOLOGY_PARAMS
from specset.schema import EVAL_SPECSET_PATH, load_specset

HI_FRAC = 0.30
LO_FRAC = 0.05
TOPOS = sorted(TOPOLOGY_PARAMS)


def shares(entries: list[dict], switch_model: str) -> tuple[Counter, int]:
    c: Counter = Counter()
    n = 0
    for e in entries:
        rs = (e.get("r_star") or {}).get(switch_model)
        if not rs or not rs.get("argmax"):
            continue
        c[rs["argmax"]] += 1
        n += 1
    return c, n


def _report(label: str, c: Counter, n: int) -> dict:
    if n == 0:
        print(f"  {label}: no specs")
        return {}
    row = {t: c.get(t, 0) / n for t in TOPOS}
    tv = 0.5 * sum(abs(v - 1.0 / len(TOPOS)) for v in row.values())
    viol = [t for t, v in row.items() if v > HI_FRAC or v < LO_FRAC]
    print(f"  {label}  (n={n}, TV from uniform {tv:.3f})")
    for t in sorted(row, key=lambda k: -row[k]):
        flag = ""
        if row[t] > HI_FRAC:
            flag = f"  > {HI_FRAC:.0%} HI"
        elif row[t] < LO_FRAC:
            flag = f"  < {LO_FRAC:.0%} LO"
        print(f"     {t:>18s} {row[t]:7.2%}{flag}")
    print(f"     -> S3 stratification {'REQUIRED' if viol else 'not required'}"
          f"{': ' + ', '.join(sorted(viol)) if viol else ''}")
    return {"n": n, "shares": row, "tv_from_uniform": tv, "violations": viol}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--specset", default=EVAL_SPECSET_PATH)
    ap.add_argument("--out", default=os.path.join(
        REPO_ROOT, "results", "joint", "envelope_class_shares.json"))
    args = ap.parse_args()

    specs = load_specset(args.specset)
    active = [e for e in specs if e.get("active_allowed")]
    passive = [e for e in specs if not e.get("active_allowed")]
    print(f"{args.specset}: {len(specs)} specs "
          f"({len(active)} active-allowed, {len(passive)} passive-only)")

    out: dict = {"specset": args.specset, "n": len(specs),
                 "hi_frac": HI_FRAC, "lo_frac": LO_FRAC, "by_switch_model": {}}
    for sm in ("ideal", "realistic"):
        print(f"\n=== {sm} ===")
        blocks = {}
        for label, subset in (("all specs", specs),
                              ("active-allowed", active),
                              ("passive-only", passive)):
            c, n = shares(subset, sm)
            blocks[label] = _report(label, c, n)
        out["by_switch_model"][sm] = blocks

    # Margin reliability: the argmax is only as trustworthy as the gap to the
    # runner-up relative to the archive's own tightness.
    print("\n=== top-2 margin (argmax reliability) ===")
    for sm in ("ideal", "realistic"):
        m = [e["r_star"][sm]["margin"] for e in specs
             if (e.get("r_star") or {}).get(sm)
             and e["r_star"][sm].get("margin") is not None]
        if m:
            m = np.array(m)
            print(f"  {sm:>10s}  p10 {np.percentile(m, 10):.4f}  "
                  f"p50 {np.median(m):.4f}  p90 {np.percentile(m, 90):.4f}")
            out["by_switch_model"][sm]["margin"] = {
                "p10": float(np.percentile(m, 10)),
                "p50": float(np.median(m)),
                "p90": float(np.percentile(m, 90)),
            }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nwrote {args.out}")
    print("\nNOTE: strata remain PROVISIONAL. The archive's gap tail makes the "
          "argmax unreliable where the top-two are close; hold strata until "
          "the top-two DE refinement (FRAMEWORK §9.12) lands before building "
          "T4's sampler on them.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
