"""Does All_Pass's difficulty actually depend on frequency?

Section 5.3 attributes All_Pass's 0% zero-shot LOOCV compliance to band
mismatch: "its 2.4 GHz target lies an order of magnitude below the 10-28 GHz
training band, so the actor's learned LC parameter prior is physically
mismatched to All Pass's resonance scale -- a distributional, not architectural,
limitation."

There is a competing explanation that involves no frequency at all: both
bridged-T sections were sized from a single window, so at the box midpoint they
are identical, the two switch states are the same circuit, and Delta-phi is 0.

The two make different predictions, and this separates them without training
anything. Under bounds='legacy' -- the configuration the published number came
from -- All_Pass's action->params map has no fc term whatsoever, so the set of
reachable circuits is *identical* at 2.4 GHz and at 14 GHz. This measures, as a
function of fc:

  * Delta-phi at the box midpoint, where a log-warped actor concentrates
  * the fraction of the box that yields a usable design ("compliant volume")
  * the best design reachable by random search

Band mismatch predicts 2.4 GHz is an outlier on these. The midpoint explanation
predicts the midpoint is degenerate at *every* fc, and that compliant volume is
roughly flat.

Run: .venv/bin/python tools/allpass_band_probe.py
"""
from __future__ import annotations

import json
import math
import os
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from env.graph_utils import TOPOLOGY_PARAMS
from sim.mna_scorer import solve_sparams, sparams_to_metrics

FCS = (2.4, 5.0, 10.0, 14.0, 20.0, 28.0, 38.0)
TRAIN_BAND = (10.0, 28.0)
N_SAMPLES = 512
DPHI_TARGET = 60.0
RL_TARGET = 12.0
IL_TARGET = 3.0


def _metrics(params: dict, fc: float) -> tuple[float, float, float] | None:
    try:
        phases, ils, rls = [], [], []
        for state in (0, 1):
            s11, s21 = solve_sparams("All_Pass", params, fc, state=state)
            m = sparams_to_metrics(s11, s21)
            phases.append(m["phase_deg"])
            ils.append(m["il_db"])
            rls.append(abs(m["rl_db"]))
    except Exception:
        return None
    dphi = abs(((phases[1] - phases[0] + 180.0) % 360.0) - 180.0)
    return dphi, float(np.mean(ils)), float(np.mean(rls))


def scan(bounds: str, split_env: str | None) -> list[dict]:
    """Sweep fc under one sizing configuration."""
    if split_env is not None:
        os.environ["G_DIFFPS_ALLPASS_SPLIT"] = split_env
    for mod in ("train_diffusion",):
        sys.modules.pop(mod, None)
    from train_diffusion import action_to_params  # re-import under new split

    keys = TOPOLOGY_PARAMS["All_Pass"]
    rng = np.random.default_rng(0)
    actions = rng.random((N_SAMPLES, len(keys)))
    mid = np.full(len(keys), 0.5)

    rows = []
    for fc in FCS:
        spec = {"fc_ghz": fc, "tech": 0}
        m = _metrics(
            action_to_params(mid, "All_Pass", spec, bounds=bounds,
                             switch_model="ideal"), fc)
        mid_dphi = m[0] if m else float("nan")

        ok, best_dphi = 0, 0.0
        for a in actions:
            r = _metrics(
                action_to_params(a, "All_Pass", spec, bounds=bounds,
                                 switch_model="ideal"), fc)
            if r is None:
                continue
            dphi, il, rl = r
            best_dphi = max(best_dphi, dphi)
            if dphi >= DPHI_TARGET and rl >= RL_TARGET and il <= IL_TARGET:
                ok += 1
        rows.append({
            "bounds": bounds, "split": split_env, "fc_ghz": fc,
            "midpoint_dphi_deg": mid_dphi,
            "compliant_volume": ok / N_SAMPLES,
            "best_dphi_deg": best_dphi,
            "in_train_band": TRAIN_BAND[0] <= fc <= TRAIN_BAND[1],
        })
    return rows


def main() -> int:
    configs = [
        ("legacy", "1.0", "legacy box (published config)"),
        ("electrical", "1.0", "electrical box, pre-fix split"),
        ("electrical", "8.0", "electrical box, re-centred split"),
    ]
    all_rows = []
    for bounds, split, label in configs:
        rows = scan(bounds, split)
        all_rows.extend(rows)
        print(f"\n=== {label} ===")
        print(f"{'fc GHz':>7s} {'band':>6s} {'mid dphi':>10s} "
              f"{'compliant vol':>14s} {'best dphi':>10s}")
        for r in rows:
            print(f"{r['fc_ghz']:7.1f} "
                  f"{'in' if r['in_train_band'] else 'out':>6s} "
                  f"{r['midpoint_dphi_deg']:10.2f} "
                  f"{r['compliant_volume']:14.3f} "
                  f"{r['best_dphi_deg']:10.2f}")

    print("\n--- verdict ---")
    for bounds, split, label in configs:
        rows = [r for r in all_rows
                if r["bounds"] == bounds and r["split"] == split]
        inb = [r["compliant_volume"] for r in rows if r["in_train_band"]]
        outb = [r["compliant_volume"] for r in rows if not r["in_train_band"]]
        lo = [r for r in rows if r["fc_ghz"] == 2.4][0]
        print(f"{label}:")
        print(f"   compliant volume  in-band {np.mean(inb):.3f}  "
              f"out-of-band {np.mean(outb):.3f}  at 2.4 GHz {lo['compliant_volume']:.3f}")
        print(f"   midpoint dphi     "
              f"{min(r['midpoint_dphi_deg'] for r in rows):.2f} .. "
              f"{max(r['midpoint_dphi_deg'] for r in rows):.2f} deg across fc")

    out = os.path.join(REPO_ROOT, "results", "joint", "allpass_band_probe.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as fh:
        json.dump({"rows": all_rows, "n_samples": N_SAMPLES,
                   "train_band_ghz": TRAIN_BAND,
                   "criteria": {"dphi_deg": DPHI_TARGET, "rl_db": RL_TARGET,
                                "il_db": IL_TARGET}}, fh, indent=2)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
