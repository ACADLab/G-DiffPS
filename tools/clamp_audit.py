"""How much of each topology's action box is unreachable because of clamping?

`netlist/llm_netlist_gen.clamp_spice_value` caps inductors at 10 nH, capacitors
at 10 pF, and TL lengths at 50 mm, silently. Under `bounds='electrical'` the
windows are multiples of the resonant value at fc, and the resonant inductance
grows as 1/fc, so at low carriers the upper part of the window is simply not
reachable:

    L0(fc) = Z0 / (2*pi*fc)     ->  7.96 nH at 1 GHz, 0.28 nH at 28 GHz

so with a 10x upper multiplier, clamping starts at 1.26*L0 at 1 GHz but not
until 35*L0 at 28 GHz. That shrinks the achievable set for every
inductor-bearing topology in the sub-6 GHz band, which is exactly where the
area term and the passive-only regime live.

This measures the fraction of the box that clamps, per topology, per fc bin,
per parameter -- so the choice between raising the caps and rejecting outright
is made against numbers.

Run: .venv/bin/python tools/clamp_audit.py
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import defaultdict

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from env.graph_utils import TOPOLOGY_PARAMS
from tools.compute_envelope import FC_REFS
from train_diffusion import action_to_params

# (suffix -> (lo, hi)) mirroring clamp_spice_value. Duplicated deliberately:
# if the two drift apart this audit's numbers stop matching reality, and the
# regression test in tests/ pins them together.
CLAMPS = {
    "_nh": (0.005, 10.0),
    "_pf": (0.005, 10.0),
    "_mm": (0.1, 50.0),
}
REL = 1e-6


def _clamped(key: str, value_str: str) -> str | None:
    """'lo'/'hi' if this parameter is sitting on a clamp boundary, else None."""
    for suf, (lo, hi) in CLAMPS.items():
        if key.lower().endswith(suf):
            try:
                v = float(value_str)
            except (TypeError, ValueError):
                return None
            if v >= hi * (1.0 - REL):
                return "hi"
            if v <= lo * (1.0 + REL):
                return "lo"
            return None
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2048)
    ap.add_argument("--out", default=os.path.join(
        REPO_ROOT, "results", "joint", "clamp_audit.json"))
    args = ap.parse_args()

    rng = np.random.default_rng(0)
    out: dict = {"n_samples": args.n, "clamps": CLAMPS,
                 "fc_refs_ghz": list(FC_REFS), "by_topology": {}}

    print(f"{'topology':>18s} " +
          "".join(f"{f:>7.2f}" for f in FC_REFS) + "    (% of draws with >=1 clamped param)")
    worst_params: dict[str, float] = defaultdict(float)

    for topo in sorted(TOPOLOGY_PARAMS):
        keys = TOPOLOGY_PARAMS[topo]
        row, per_bin = [], {}
        for bi, fc in enumerate(FC_REFS):
            hits = 0
            per_key: dict[str, int] = defaultdict(int)
            for a in rng.random((args.n, len(keys))):
                p = action_to_params(a, topo, {"fc_ghz": fc, "tech": 0},
                                     bounds="electrical", switch_model="ideal")
                any_hit = False
                for k in keys:
                    side = _clamped(k, p.get(k))
                    if side:
                        per_key[f"{k}:{side}"] += 1
                        any_hit = True
                hits += any_hit
            frac = hits / args.n
            row.append(frac)
            per_bin[str(bi)] = {
                "fc_ghz": fc, "any_clamped": frac,
                "by_param": {k: v / args.n for k, v in
                             sorted(per_key.items(), key=lambda kv: -kv[1])},
            }
            for k, v in per_key.items():
                worst_params[f"{topo}.{k}"] = max(
                    worst_params[f"{topo}.{k}"], v / args.n)
        out["by_topology"][topo] = per_bin
        print(f"{topo:>18s} " + "".join(f"{v:6.0%} " for v in row))

    print(f"\nworst offending parameters (max over fc bins):")
    for k, v in sorted(worst_params.items(), key=lambda kv: -kv[1])[:12]:
        print(f"  {k:<34s} {v:6.1%}")

    # Where does clamping begin, analytically, for a 10x upper multiplier?
    print(f"\ninductor headroom: 10 nH cap vs L0 = Z0/omega")
    print(f"{'fc GHz':>8s} {'L0 nH':>8s} {'clamp starts at':>18s}")
    for fc in (1.0, 2.0, 5.0, 8.0, 14.0, 28.0, 40.0):
        l0 = 50.0 / (2 * math.pi * fc * 1e9) * 1e9
        print(f"{fc:8.1f} {l0:8.2f} {10.0 / l0:16.2f}xL0")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
