"""Replay logged (topology, params) triples under local ngspice.

Gates whether local ngspice-47 numbers are comparable to the paper's
ngspice-46 A100 numbers. Specs were not logged in train.log, so the
primary comparison is on SPICE-derived metrics (il_db, rl_db,
rms_phase_err_deg). Reward is recomputed against a fixed reference
spec only as a secondary signal.

Usage:
  python tools/replay_checkpoint.py \
      --log checkpoints/run_20260530_031117/train.log \
      --n 300 --seed 42 \
      --out results/harness/replay_ngspice47.json
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import defaultdict

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from env.phaseshifter_env import PhaseShifterEnv
from train_diffusion import make_spice_netlist


REF_SPEC = {
    "fc_ghz": 28.0,
    "bw_pct": 30.0,
    "phase_coverage_deg": 180.0,
    "phase_bits": 1,
    "rms_phase_err_deg": 5.0,
    "rms_gain_err_db": 1.0,
    "max_il_db": 5.0,
    "min_rl_db": 10.0,
    "vdd": 1.8,
    "pmax_mw": 15.0,
    "tech": 0,
    "app": 2,
}


def load_entries(log_path: str) -> list[dict]:
    rows = []
    bad = 0
    with open(log_path) as fh:
        for line in fh:
            line = line.strip()
            if not line or '"step"' not in line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                bad += 1
                continue
            if "topology" in d and "params" in d and d.get("passed_prior"):
                rows.append(d)
    return rows, bad


def metric_delta(logged: dict | None, replayed: dict | None, key: str) -> float | None:
    if not logged or not replayed:
        return None
    a, b = logged.get(key), replayed.get(key)
    if a is None or b is None:
        return None
    return float(b) - float(a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default=os.path.join(
        REPO_ROOT, "checkpoints/run_20260530_031117/train.log"))
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=os.path.join(
        REPO_ROOT, "results/harness/replay_ngspice47.json"))
    args = ap.parse_args()

    rows, bad = load_entries(args.log)
    print(f"Loaded {len(rows)} prior-passing entries ({bad} unparseable lines)")

    rng = random.Random(args.seed)
    # Stratify by topology so every family is represented.
    by_topo = defaultdict(list)
    for r in rows:
        by_topo[r["topology"]].append(r)
    sample = []
    per = max(1, args.n // max(1, len(by_topo)))
    for topo, lst in by_topo.items():
        k = min(per, len(lst))
        sample.extend(rng.sample(lst, k))
    if len(sample) < args.n:
        remaining = [r for r in rows if r not in sample]
        sample.extend(rng.sample(remaining, min(args.n - len(sample), len(remaining))))
    sample = sample[: args.n]
    print(f"Replaying {len(sample)} designs "
          f"({ {t: sum(1 for s in sample if s['topology']==t) for t in by_topo} })")

    env = PhaseShifterEnv()
    env.current_spec = dict(REF_SPEC)

    records = []
    wall_times = []
    flips = []
    deltas = defaultdict(list)

    for i, entry in enumerate(sample):
        topo = entry["topology"]
        params = entry["params"]
        logged_metrics = entry.get("metrics")
        logged_success = bool(entry.get("success"))

        t0 = time.perf_counter()
        nl = make_spice_netlist(topo, params)
        try:
            agg, sim_reward, state_indices, bits, ideal_step = env._evaluate_netlist(nl)
        finally:
            try:
                os.remove(nl)
            except OSError:
                pass
            lis = nl + ".lis"
            if os.path.exists(lis):
                try:
                    os.remove(lis)
                except OSError:
                    pass
        elapsed = time.perf_counter() - t0
        wall_times.append(elapsed)

        replay_success = agg is not None
        if replay_success != logged_success:
            flips.append({
                "step": entry["step"],
                "topology": topo,
                "logged_success": logged_success,
                "replay_success": replay_success,
            })

        rec = {
            "step": entry["step"],
            "topology": topo,
            "wall_s": elapsed,
            "logged_success": logged_success,
            "replay_success": replay_success,
            "logged_reward": entry.get("reward"),
            "replay_sim_reward": float(sim_reward) if sim_reward is not None else None,
            "deltas": {},
        }
        for key in ("rms_phase_err_deg", "il_db", "rl_db", "gain_err_db"):
            d = metric_delta(logged_metrics, agg, key)
            rec["deltas"][key] = d
            if d is not None:
                deltas[key].append(d)
        records.append(rec)

        if (i + 1) % 25 == 0 or i == 0:
            print(f"  [{i+1}/{len(sample)}] {topo} wall={elapsed*1000:.0f}ms "
                  f"success={logged_success}->{replay_success}", flush=True)

    def summarize(vals):
        if not vals:
            return {"n": 0}
        a = np.asarray(vals, dtype=float)
        return {
            "n": int(a.size),
            "mean": float(a.mean()),
            "std": float(a.std()),
            "p50": float(np.percentile(a, 50)),
            "p95": float(np.percentile(a, 95)),
            "max_abs": float(np.max(np.abs(a))),
        }

    summary = {
        "n_replayed": len(records),
        "n_unparseable_log_lines": bad,
        "n_success_flips": len(flips),
        "wall_time_s": summarize(wall_times),
        "metric_deltas": {k: summarize(v) for k, v in deltas.items()},
        "flips": flips[:50],
        "comparable": (
            len(flips) == 0
            and all(summarize(v).get("max_abs", 1.0) < 0.05 for v in deltas.values() if v)
        ),
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump({"summary": summary, "records": records}, fh, indent=2)

    print("\n=== HARNESS VALIDATION ===")
    print(json.dumps(summary, indent=2))
    print(f"\nSaved -> {args.out}")
    if summary["comparable"]:
        print("VERDICT: local ngspice-47 matches logged metrics (max |delta| < 0.05).")
    else:
        print("VERDICT: drift detected — inspect flips / max_abs before comparing to paper.")


if __name__ == "__main__":
    main()
