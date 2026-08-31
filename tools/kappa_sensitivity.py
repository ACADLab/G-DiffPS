"""Is `g_il` leading the variance shares a design choice or an accident?

Frontier anchoring sets each spec target as `anchor · kappa`, so whichever term
is anchored tightest becomes the dominant discriminator. At the shipped kappa
ranges `g_il` (0.40) edges out `g_phase` (0.38) on a *phase-shifter* benchmark,
which is a consequence of five independently chosen kappa ranges rather than of
physics.

This publishes the realized kappa distributions and sweeps a multiplier on each
term's kappa independently, reporting how the variance shares move. Scaling a
term's kappa by `f` scales its target by `f` (or by `1/f` for min_rl_db, where
larger is stricter), so the sweep needs no regeneration.

Run: .venv/bin/python tools/kappa_sensitivity.py
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
from env.reward import WEIGHTS_AREA
from specset.schema import (
    GAIN_KAPPA_RANGE, IL_KAPPA_RANGE, PHASE_KAPPA_RANGE, RL_KAPPA_RANGE,
    TRAIN_SPECSET_PATH, load_specset,
)
from tools.t1_envelope_gate import TERMS, envelope_parts

# Which spec field each kappa scales, and whether larger kappa means a slacker
# target (+1) or a stricter one (-1, i.e. the target divides by kappa).
KAPPA_FIELDS = {
    "phase_kappa": ("rms_phase_err_deg", +1),
    "il_kappa": ("max_il_db", +1),
    "rl_kappa": ("min_rl_db", -1),
    "gain_kappa": ("rms_gain_err_db", +1),
}
RANGES = {
    "phase_kappa": PHASE_KAPPA_RANGE, "il_kappa": IL_KAPPA_RANGE,
    "rl_kappa": RL_KAPPA_RANGE, "gain_kappa": GAIN_KAPPA_RANGE,
}


def term_shares(specs, archives, sm: str, topos) -> dict[str, float]:
    per_term = {t: [] for t in TERMS}
    for e in specs:
        cols = {t: [] for t in TERMS}
        ok = True
        for topo in topos:
            parts = envelope_parts(e["spec"], archives, topo, sm)
            if parts is None:
                ok = False
                break
            for t in TERMS:
                v = parts.get(t)
                cols[t].append(0.0 if v is None else float(v))
        if not ok:
            continue
        for t in TERMS:
            per_term[t].append(float(np.var(cols[t])))
    w = {"g_phase": WEIGHTS_AREA.phase, "g_il": WEIGHTS_AREA.il,
         "g_rl": WEIGHTS_AREA.rl, "g_gain": WEIGHTS_AREA.gain,
         "g_area": WEIGHTS_AREA.area}
    contrib = {t: (w[t] ** 2) * float(np.mean(per_term[t]))
               for t in TERMS if per_term[t]}
    tot = sum(contrib.values()) or 1.0
    return {t: v / tot for t, v in contrib.items()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--specset", default=TRAIN_SPECSET_PATH)
    ap.add_argument("--archive", default=os.path.join(
        REPO_ROOT, "results", "joint", "envelope_archive.json"))
    ap.add_argument("--n", type=int, default=250)
    ap.add_argument("--switch-model", default="ideal")
    ap.add_argument("--out", default=os.path.join(
        REPO_ROOT, "results", "joint", "kappa_sensitivity.json"))
    args = ap.parse_args()

    with open(args.archive) as fh:
        archives = json.load(fh)["archives"]
    specs = load_specset(args.specset)
    topos = sorted(TOPOLOGY_PARAMS)

    print("=== realized kappa distributions (declared range in brackets) ===")
    realized = {}
    for k, rng_ in RANGES.items():
        v = np.array([e[k] for e in specs if k in e])
        if not len(v):
            continue
        realized[k] = {"p10": float(np.percentile(v, 10)),
                       "p50": float(np.median(v)),
                       "p90": float(np.percentile(v, 90)),
                       "declared": list(rng_)}
        print(f"  {k:>12s}  p10 {realized[k]['p10']:6.2f}  "
              f"p50 {realized[k]['p50']:6.2f}  p90 {realized[k]['p90']:6.2f}"
              f"   [{rng_[0]}, {rng_[1]}]")

    sub = [e for e in specs if (e.get("r_star") or {}).get(args.switch_model)]
    sub = sub[: args.n]

    print(f"\n=== term variance shares, {args.switch_model}, n={len(sub)} ===")
    base = term_shares(sub, archives, args.switch_model, topos)
    order = sorted(base, key=base.get, reverse=True)
    print("  baseline   " + "  ".join(f"{t}={base[t]:.2f}" for t in order))

    out = {"realized_kappa": realized, "baseline_shares": base, "sweep": {}}
    print(f"\n=== sensitivity: multiply one kappa, hold the rest ===")
    print(f"{'knob':>12s} {'x':>5s}  " + "  ".join(f"{t:>8s}" for t in order))
    for knob, (field, sgn) in KAPPA_FIELDS.items():
        out["sweep"][knob] = {}
        for f in (0.5, 1.0, 2.0, 4.0):
            mod = []
            for e in sub:
                e2 = dict(e)
                s = dict(e["spec"])
                s[field] = s[field] * (f if sgn > 0 else 1.0 / f)
                e2["spec"] = s
                mod.append(e2)
            sh = term_shares(mod, archives, args.switch_model, topos)
            out["sweep"][knob][str(f)] = sh
            print(f"{knob:>12s} {f:5.1f}  " +
                  "  ".join(f"{sh.get(t, 0.0):8.2f}" for t in order))

    lead = {}
    for knob in KAPPA_FIELDS:
        for f, sh in out["sweep"][knob].items():
            lead[(knob, f)] = max(sh, key=sh.get)
    flips = {k: v for k, v in lead.items() if v != order[0]}
    print(f"\nbaseline leading term: {order[0]} ({base[order[0]]:.2f})")
    if flips:
        print("the lead changes under these perturbations, so it is a "
              "consequence of anchoring tightness, not of physics:")
        for (knob, f), t in sorted(flips.items()):
            print(f"  {knob} x{f} -> {t} leads")
    else:
        print("the lead is stable across the sweep.")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2, default=str)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
