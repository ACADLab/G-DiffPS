"""Fan out LOOCV training runs across CPU cores and emit comparison CSVs.

Usage examples:
  # PRIMARY corrected harness (spec fc + electrical bounds):
  python tools/run_matrix.py --track corrected --steps 5000 --seeds 42,1337,2026 \\
      --out results/matrix_corrected

  # Short fixed28 control (paper meter faithfulness):
  python tools/run_matrix.py --track fixed28_control --steps 5000 --seeds 42,1337 \\
      --out results/matrix_fixed28_control
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from env.graph_utils import TOPOLOGY_PARAMS

TOPOS = list(TOPOLOGY_PARAMS.keys())

# PRIMARY: corrected frequency + electrical bounds.
# Circuit encoder is the main candidate; GIN is the ablation.
TRACK_CORRECTED = [
    {"encoder": "circuit", "action_space": "device", "fc_mode": "spec", "bounds": "electrical",
     "tag": "circuit_device", "switch_model": "ideal", "coupled_actions": False},
    {"encoder": "circuit", "action_space": "device", "fc_mode": "spec", "bounds": "electrical",
     "tag": "circuit_device_coupled", "switch_model": "ideal", "coupled_actions": True},
    {"encoder": "circuit", "action_space": "slot", "fc_mode": "spec", "bounds": "electrical",
     "tag": "circuit_slot", "switch_model": "ideal", "coupled_actions": False},
    {"encoder": "gin", "action_space": "device", "fc_mode": "spec", "bounds": "electrical",
     "tag": "gin_device", "switch_model": "ideal", "coupled_actions": False},
    {"encoder": "gin", "action_space": "slot", "fc_mode": "spec", "bounds": "electrical",
     "tag": "gin_slot", "switch_model": "ideal", "coupled_actions": False},
]

# CONTROL only: paper-faithful fixed28 meter (gin_slot).
TRACK_FIXED28_CONTROL = [
    {"encoder": "gin", "action_space": "slot", "fc_mode": "fixed28", "bounds": "legacy",
     "tag": "gin_slot_fixed28", "switch_model": "ideal", "coupled_actions": False},
]

# Legacy aliases (kept for older scripts / smoke dirs).
TRACK_A = [
    {"encoder": "gin", "action_space": "slot", "fc_mode": "fixed28", "bounds": "legacy",
     "tag": "gin_slot", "switch_model": "ideal", "coupled_actions": False},
    {"encoder": "gin", "action_space": "device", "fc_mode": "fixed28", "bounds": "legacy",
     "tag": "gin_device", "switch_model": "ideal", "coupled_actions": False},
    {"encoder": "circuit", "action_space": "slot", "fc_mode": "fixed28", "bounds": "legacy",
     "tag": "circuit_slot", "switch_model": "ideal", "coupled_actions": False},
    {"encoder": "circuit", "action_space": "device", "fc_mode": "fixed28", "bounds": "legacy",
     "tag": "circuit_device", "switch_model": "ideal", "coupled_actions": False},
]

# Matched encoder ablation (Phase 6): same policy, MNA scorer, device actions.
TRACK_TYPED = [
    {"encoder": "gin", "action_space": "device", "fc_mode": "spec", "bounds": "electrical",
     "tag": "gin_device", "switch_model": "ideal", "coupled_actions": False, "sim": "mna"},
    {"encoder": "circuit", "action_space": "device", "fc_mode": "spec", "bounds": "electrical",
     "tag": "circuit_device", "switch_model": "ideal", "coupled_actions": False, "sim": "mna"},
    {"encoder": "circuit-typed", "action_space": "device", "fc_mode": "spec", "bounds": "electrical",
     "tag": "circuit_typed_device", "switch_model": "ideal", "coupled_actions": False, "sim": "mna"},
]

TRACK_B = TRACK_CORRECTED


def _one_run(job: dict) -> dict:
    """Train with one topology held out, then evaluate zero-shot compliance."""
    py = sys.executable
    held_out = job["held_out"]
    train_topos = [t for t in TOPOS if t != held_out]
    run_dir = job["run_dir"]
    os.makedirs(run_dir, exist_ok=True)

    cmd = [
        py, os.path.join(REPO_ROOT, "train_diffusion.py"),
        "--total-timesteps", str(job["steps"]),
        "--batch-size", str(job.get("batch_size", 16)),
        "--seed", str(job["seed"]),
        "--actor", "cfm",
        "--encoder", job["encoder"],
        "--action-space", job["action_space"],
        "--fc-mode", job["fc_mode"],
        "--bounds", job["bounds"],
        "--switch-model", job.get("switch_model", "ideal"),
        "--expert-bonus-scale", str(job.get("expert_bonus_scale", 0.0)),
        "--run-dir", run_dir,
        "--restrict-to", *train_topos,
    ]
    if job.get("sim"):
        cmd.extend(["--sim", job["sim"]])
    if job.get("skip_prior"):
        cmd.append("--skip-prior")
    if job.get("coupled_actions"):
        cmd.append("--coupled-actions")
    log_path = os.path.join(run_dir, "train_stdout.log")
    t0 = time.time()
    env = os.environ.copy()
    env["PYTHONPATH"] = REPO_ROOT
    with open(log_path, "w") as fh:
        proc = subprocess.run(
            cmd, cwd=REPO_ROOT, stdout=fh, stderr=subprocess.STDOUT, env=env,
        )
    wall = time.time() - t0
    ckpt_dir = run_dir

    if proc.returncode != 0:
        result = {
            **{k: job[k] for k in ("tag", "held_out", "seed", "encoder",
                                   "action_space", "fc_mode", "bounds")},
            "train_returncode": proc.returncode,
            "eval_returncode": None,
            "wall_s": wall,
            "ckpt_dir": ckpt_dir,
            "run_dir": run_dir,
            "error": "training_failed",
        }
        with open(os.path.join(run_dir, "job_result.json"), "w") as fh:
            json.dump(result, fh, indent=2)
        return result

    eval_spec = job.get(
        "eval_specset",
        os.path.join(REPO_ROOT, "specset", "specset_eval.json"),
    )
    # Zero-shot eval on held-out topology
    eval_cmd = [
        py, os.path.join(REPO_ROOT, "tools", "loocv_eval.py"),
        "--run", ckpt_dir or run_dir,
        "--held-out", held_out,
        "--encoder", job["encoder"],
        "--action-space", job["action_space"],
        "--fc-mode", job["fc_mode"],
        "--bounds", job["bounds"],
        "--switch-model", job.get("switch_model", "ideal"),
        "--eval-specset", eval_spec,
        "--n", str(job.get("n_eval", 200)),
        "--seed", str(job["seed"]),
        "--out", os.path.join(run_dir, "loocv.json"),
    ]
    if job.get("sim"):
        eval_cmd.extend(["--sim", job["sim"]])
    if job.get("coupled_actions"):
        eval_cmd.append("--coupled-actions")
    eval_log = os.path.join(run_dir, "eval_stdout.log")
    with open(eval_log, "w") as fh:
        eval_proc = subprocess.run(
            eval_cmd, cwd=REPO_ROOT, stdout=fh, stderr=subprocess.STDOUT, env=env,
        )

    result = {
        **{k: job[k] for k in ("tag", "held_out", "seed", "encoder",
                               "action_space", "fc_mode", "bounds")},
        "train_returncode": proc.returncode,
        "eval_returncode": eval_proc.returncode,
        "wall_s": wall,
        "ckpt_dir": ckpt_dir,
        "run_dir": run_dir,
    }
    loocv_path = os.path.join(run_dir, "loocv.json")
    if os.path.exists(loocv_path):
        with open(loocv_path) as fh:
            result["loocv"] = json.load(fh)
    with open(os.path.join(run_dir, "job_result.json"), "w") as fh:
        json.dump(result, fh, indent=2)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--track",
        choices=["corrected", "fixed28_control", "A", "B", "both", "typed"],
        default="corrected",
        help="corrected=primary; typed=gin vs circuit vs circuit-typed MNA ablation",
    )
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--seeds", default="42,1337,2026")
    ap.add_argument("--folds", default=",".join(TOPOS),
                    help="Comma-separated held-out topologies")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    ap.add_argument("--n-eval", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--out", default=os.path.join(REPO_ROOT, "results", "matrix_corrected"))
    ap.add_argument("--dry-run", action="store_true",
                    help="Print jobs without executing")
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    folds = [f.strip() for f in args.folds.split(",") if f.strip()]
    configs = []
    if args.track == "corrected":
        configs += TRACK_CORRECTED
    elif args.track == "typed":
        configs += TRACK_TYPED
    elif args.track == "fixed28_control":
        configs += TRACK_FIXED28_CONTROL
    elif args.track in ("A", "both"):
        configs += TRACK_A
    if args.track in ("B", "both"):
        configs += TRACK_B

    tag_filter = os.environ.get("G_DIFFPS_MATRIX_TAGS", "").strip()
    if tag_filter:
        allowed = {t.strip() for t in tag_filter.split(",") if t.strip()}
        configs = [c for c in configs if c["tag"] in allowed]

    jobs = []
    for cfg in configs:
        for held in folds:
            for seed in seeds:
                run_dir = os.path.join(
                    args.out, cfg["tag"], f"holdout_{held}", f"seed_{seed}",
                )
                jobs.append({
                    **cfg,
                    "held_out": held,
                    "seed": seed,
                    "steps": args.steps,
                    "n_eval": args.n_eval,
                    "batch_size": args.batch_size,
                    "run_dir": run_dir,
                })

    print(f"Scheduled {len(jobs)} jobs across {args.workers} workers")
    if args.dry_run:
        for j in jobs:
            print(f"  {j['tag']} hold={j['held_out']} seed={j['seed']}")
        return

    os.makedirs(args.out, exist_ok=True)
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(_one_run, j): j for j in jobs}
        for fut in as_completed(futs):
            j = futs[fut]
            try:
                r = fut.result()
            except Exception as e:
                r = {**j, "error": str(e)}
            results.append(r)
            print(f"DONE {r.get('tag')} hold={r.get('held_out')} "
                  f"seed={r.get('seed')} wall={r.get('wall_s', 0):.0f}s",
                  flush=True)

    # Emit CSV in AppC_gnn_fix schema + extras
    csv_path = os.path.join(args.out, "comparison.csv")
    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow([
            "topology", "config", "strict_compliance_mean", "best_physical_reward_mean",
            "n_seeds", "encoder", "action_space", "fc_mode", "bounds",
        ])
        # Aggregate by (held_out, tag)
        from collections import defaultdict
        buckets = defaultdict(list)
        for r in results:
            if "loocv" not in r:
                continue
            buckets[(r["held_out"], r["tag"])].append(r)
        for (topo, tag), rs in sorted(buckets.items()):
            rates = [r["loocv"].get("strict_compliance", r["loocv"].get("compliance", 0)) * 100
                     for r in rs]
            bests = [r["loocv"].get("best_physical_reward",
                                    r["loocv"].get("best_reward", -5.0)) for r in rs]
            cfg0 = rs[0]
            w.writerow([
                topo, tag,
                sum(rates) / len(rates),
                sum(bests) / len(bests),
                len(rs),
                cfg0["encoder"], cfg0["action_space"],
                cfg0["fc_mode"], cfg0["bounds"],
            ])
    print(f"Wrote {csv_path}")

    with open(os.path.join(args.out, "all_results.json"), "w") as fh:
        json.dump(results, fh, indent=2)


if __name__ == "__main__":
    main()
