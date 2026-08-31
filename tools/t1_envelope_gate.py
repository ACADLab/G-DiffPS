"""T1 acceptance gate, restated at the envelope.

History
-------
The original T1 gate asked for r_sim spread < 0.15 across topologies at template
defaults. That was unreachable and correctly retired: the +1.0 all_close bonus is
discrete, so any spec where one topology clears every threshold and another does
not produces a spread >= 1.0 no matter how the continuous terms are weighted. But
retiring it left nothing behind, so T1 was landed on the strength of the code
existing.

This is the replacement. The right question is not whether topologies score
similarly at an arbitrary nominal sizing -- they should not, that is the whole
premise -- but whether the reward *discriminates between topologies when each is
sized as well as it can be*. That is a statement about r_star, which is why the
gate was blocked on the envelope.

Gates
-----
G1  envelope spread: median over specs of (max_tau r_star - min_tau r_star)
    must exceed MIN_ENVELOPE_SPREAD. If every topology reaches the same reward
    when optimally sized, topology choice is not a decidable problem and no
    selector result on this pool is meaningful.

G2  discrimination: the fraction of specs where ALL topologies reach all_close
    must stay below MAX_ALL_SATURATED. Above that, the spec is satisfied by
    anything and contributes no signal -- the same under-determination the
    floor-relative sampling convention exists to prevent, but in the reward
    rather than in a spec field.

G3  term decomposition: at the envelope, no single reward term may account for
    more than MAX_TERM_SHARE of the between-topology variance. If one term
    dominates, the other four are decoration.

Run: .venv/bin/python tools/t1_envelope_gate.py
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
from env.reward import WEIGHTS_AREA, compute_sim_reward
from sim.area_model import estimate_area_mm2
from specset.schema import atomic_write_json, load_specset
from tools.compute_envelope import (
    SWITCH_MODELS, _cell_key, c_off_class, fc_bin,
)
from train_diffusion import action_to_params

MIN_ENVELOPE_SPREAD = 0.25
MAX_ALL_SATURATED = 0.20
MAX_TERM_SHARE = 0.60

TERMS = ("g_phase", "g_il", "g_rl", "g_gain", "g_area")


def envelope_parts(spec: dict, archives: dict, topo: str, sm: str) -> dict | None:
    """Reward parts at the archive-optimal sizing for one (spec, topo, model)."""
    fc, tech = float(spec["fc_ghz"]), int(spec["tech"])
    c = None if sm == "ideal" else c_off_class(tech)
    cell = archives.get(_cell_key(topo, sm, fc_bin(fc), c))
    if not cell or "front" not in cell:
        return None
    best_r, best_parts = -1e9, None
    for pt in cell["front"]:
        m = dict(pt["m"])
        try:
            params = action_to_params(
                np.asarray(pt["action"], float), topo, {"fc_ghz": fc, "tech": tech},
                bounds="electrical", switch_model=sm,
            )
            m["area_mm2"] = estimate_area_mm2(topo, params, fc_ghz=fc, tech=tech)
        except Exception:
            m["area_mm2"] = m.get("area_ref_mm2")
        r, parts = compute_sim_reward(
            m, spec, weights=WEIGHTS_AREA, return_parts=True)
        if r > best_r:
            best_r, best_parts = r, parts
    return best_parts


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--specset", default=os.path.join(
        REPO_ROOT, "specset", "specset_eval.json"))
    ap.add_argument("--archive", default=os.path.join(
        REPO_ROOT, "results", "joint", "envelope_archive.json"))
    ap.add_argument("--n", type=int, default=400,
                    help="specs to decompose (term decomposition is the slow part)")
    ap.add_argument("--out", default=os.path.join(
        REPO_ROOT, "results", "joint", "t1_envelope_gate.json"))
    args = ap.parse_args()

    with open(args.archive) as fh:
        archives = json.load(fh)["archives"]
    specs = load_specset(args.specset)
    topos = list(TOPOLOGY_PARAMS)

    have_rstar = [e for e in specs if e.get("r_star")]
    if not have_rstar:
        print("ERROR: specset has no r_star. Run "
              "`compute_envelope.py assign` first.")
        return 2

    report = {"specset": args.specset, "n_specs": len(have_rstar),
              "gates": {}, "per_switch_model": {}}
    passed_all = True

    for sm in SWITCH_MODELS:
        # Inadmissible (topology, spec) pairs carry None, not a sentinel value:
        # they are masked out of the candidate set rather than scored. Every
        # statistic below is therefore over admissible topologies only.
        vals = np.array([
            [(np.nan if e["r_star"][sm]["per_topology"][t] is None
              else e["r_star"][sm]["per_topology"][t]) for t in topos]
            for e in have_rstar
        ], dtype=float)
        n_adm = np.sum(~np.isnan(vals), axis=1)
        vals = vals[n_adm >= 2]
        spread = np.nanmax(vals, axis=1) - np.nanmin(vals, axis=1)
        srt = np.sort(np.where(np.isnan(vals), -np.inf, vals), axis=1)
        margin = srt[:, -1] - srt[:, -2]
        all_sat = np.mean(np.all((vals >= 1.0) | np.isnan(vals), axis=1))

        block = {
            "envelope_spread": {
                "mean": float(spread.mean()), "p50": float(np.median(spread)),
                "p10": float(np.percentile(spread, 10)),
                "p90": float(np.percentile(spread, 90)),
            },
            "top2_margin": {
                "mean": float(margin.mean()), "p50": float(np.median(margin)),
                "p90": float(np.percentile(margin, 90)),
            },
            "frac_all_topologies_saturated": float(all_sat),
            "best_r_star_p50": float(np.median(vals.max(axis=1))),
        }

        # ---- G3: which term carries the between-topology variance? ----
        sub = have_rstar[: args.n]
        per_term = {t: [] for t in TERMS}
        for e in sub:
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
        # Weighted contribution of each term to reward variance across topologies.
        contrib = {t: (w[t] ** 2) * float(np.mean(per_term[t]))
                   for t in TERMS if per_term[t]}
        tot = sum(contrib.values()) or 1.0
        share = {t: v / tot for t, v in contrib.items()}
        block["term_variance_share"] = share
        block["term_decomposition_n"] = len(per_term["g_phase"])

        g1 = block["envelope_spread"]["p50"] >= MIN_ENVELOPE_SPREAD
        g2 = all_sat <= MAX_ALL_SATURATED
        top_term = max(share, key=share.get) if share else None
        g3 = (share[top_term] <= MAX_TERM_SHARE) if top_term else False
        block["gates"] = {
            "G1_envelope_spread": {
                "pass": bool(g1), "value": block["envelope_spread"]["p50"],
                "threshold": MIN_ENVELOPE_SPREAD, "cmp": ">=",
            },
            "G2_not_all_saturated": {
                "pass": bool(g2), "value": float(all_sat),
                "threshold": MAX_ALL_SATURATED, "cmp": "<=",
            },
            "G3_no_dominant_term": {
                "pass": bool(g3),
                "value": None if not top_term else share[top_term],
                "dominant_term": top_term,
                "threshold": MAX_TERM_SHARE, "cmp": "<=",
            },
        }
        passed_all = passed_all and g1 and g2 and g3
        report["per_switch_model"][sm] = block

        print(f"\n=== {sm} ===")
        print(f"  envelope spread   p10/p50/p90 = "
              f"{block['envelope_spread']['p10']:.3f}/"
              f"{block['envelope_spread']['p50']:.3f}/"
              f"{block['envelope_spread']['p90']:.3f}")
        print(f"  top-2 margin      p50 = {block['top2_margin']['p50']:.4f}")
        print(f"  best r_star       p50 = {block['best_r_star_p50']:.3f}")
        print(f"  all six saturated       = {all_sat:.1%}")
        print(f"  term variance share: " + ", ".join(
            f"{t}={share[t]:.2f}" for t in sorted(share, key=share.get, reverse=True)))
        for gname, g in block["gates"].items():
            print(f"  {'PASS' if g['pass'] else 'FAIL'}  {gname}: "
                  f"value={g['value']} {g['cmp']} {g['threshold']}")

    report["gates"]["all_pass"] = bool(passed_all)
    report["verdict"] = (
        "T1 ACCEPTED at the envelope" if passed_all else
        "T1 NOT ACCEPTED at the envelope — see failing gates"
    )
    print(f"\n=== VERDICT: {report['verdict']} ===")

    atomic_write_json(args.out, report)
    print(f"wrote {args.out}")
    return 0 if passed_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
