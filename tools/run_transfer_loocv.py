"""Bounded LOOCV transfer pilot + full matrix runners.

Branch rule: no run >2k steps until C0 (SKY130 graph→SPICE→metrics) passes.
Pilot: hard holdouts, 1 seed, 1k–2k steps, circuit vs gin.
Full: six folds × 3 seeds after pilot.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _run(cmd: list[str], log: Path) -> int:
    log.parent.mkdir(parents=True, exist_ok=True)
    print("RUN:", " ".join(cmd), flush=True)
    with open(log, "w") as fh:
        proc = subprocess.run(
            cmd, cwd=str(REPO_ROOT), stdout=fh, stderr=subprocess.STDOUT,
            env={**os.environ, "PYTHONPATH": str(REPO_ROOT)},
        )
    return proc.returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["pilot", "full"], default="pilot")
    ap.add_argument("--steps", type=int, default=None,
                    help="Override steps (pilot default 1500, full default 2000)")
    ap.add_argument("--n-eval", type=int, default=50)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--python", default=sys.executable)
    args = ap.parse_args()

    py = args.python
    matrix = str(REPO_ROOT / "tools" / "run_matrix.py")

    if args.mode == "pilot":
        steps = args.steps or 1500
        out = str(REPO_ROOT / "results" / "transfer_pilot")
        folds = "All_Pass,Vector_Modulator,Reflection_Type"
        # Only primary matched pair for the pilot (circuit vs gin, device).
        # run_matrix TRACK_CORRECTED has more; filter via a temp approach:
        # invoke run_matrix with folds/seeds and post-filter, OR call jobs directly.
        cmd = [
            py, matrix,
            "--track", "corrected",
            "--steps", str(steps),
            "--seeds", "42",
            "--folds", folds,
            "--n-eval", str(args.n_eval),
            "--workers", str(args.workers),
            "--out", out,
        ]
        # Restrict configs by writing a thin wrapper: run two tags only via
        # environment variable consumed below if present.
        os.environ["G_DIFFPS_MATRIX_TAGS"] = "circuit_device,gin_device"
        rc = _run(cmd, Path(out) / "pilot_console.log")
        summary = {"mode": "pilot", "steps": steps, "out": out, "returncode": rc}
        Path(out).mkdir(parents=True, exist_ok=True)
        (Path(out) / "pilot_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        sys.exit(rc)

    # Full six-fold
    steps = args.steps or 2000
    out = str(REPO_ROOT / "results" / "transfer_loocv")
    os.environ["G_DIFFPS_MATRIX_TAGS"] = "circuit_device,gin_device"
    cmd = [
        py, matrix,
        "--track", "corrected",
        "--steps", str(steps),
        "--seeds", "42,1337,2026",
        "--folds", "Loaded_Line,Switched_Line,Reflection_Type,Switched_Filter,Vector_Modulator,All_Pass",
        "--n-eval", str(args.n_eval),
        "--workers", str(args.workers),
        "--out", out,
    ]
    rc = _run(cmd, Path(out) / "full_console.log")
    protocol = {
        "mode": "full",
        "steps": steps,
        "seeds": [42, 1337, 2026],
        "tags": ["circuit_device", "gin_device"],
        "primary_metric": "strict_compliance",
        "eval_specset": "specset/specset_eval.json",
        "expert_bonus_scale": 0.0,
        "checkpoint_selection": "final_training_checkpoint",
        "out": out,
        "returncode": rc,
    }
    Path(out).mkdir(parents=True, exist_ok=True)
    (Path(out) / "PROTOCOL.json").write_text(json.dumps(protocol, indent=2) + "\n")
    sys.exit(rc)


if __name__ == "__main__":
    main()
