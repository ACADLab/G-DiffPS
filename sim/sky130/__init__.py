"""SKY130 / ngspice simulation harness.

Resolves the pinned PDK, builds model includes, runs batch decks, and
separates simulator success from parsed measurement success.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PIN = REPO_ROOT / "pdk" / "sky130_pin.yaml"


@dataclass
class SimResult:
    ok: bool
    returncode: int
    stdout: str
    stderr: str
    metrics: dict[str, float] = field(default_factory=dict)
    raw_path: str | None = None
    error: str | None = None


def default_pdk_root() -> Path:
    env = os.environ.get("PDK_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    return (REPO_ROOT / "pdk").resolve()


def load_pin(path: Path | None = None) -> dict[str, Any]:
    pin_path = path or DEFAULT_PIN
    with open(pin_path) as fh:
        return yaml.safe_load(fh)


def resolve_ngspice_lib(pin: dict[str, Any] | None = None, pdk_root: Path | None = None) -> Path:
    pin = pin or load_pin()
    root = pdk_root or default_pdk_root()
    lib = root / pin["pdk"]["ngspice_lib"]
    if not lib.is_file():
        raise FileNotFoundError(
            f"SKY130 ngspice library not found at {lib}. "
            f"Set PDK_ROOT and run: volare enable --pdk sky130 {pin['pdk']['volare_hash']}"
        )
    return lib.resolve()


def ngspice_version() -> str | None:
    exe = shutil.which("ngspice")
    if not exe:
        return None
    try:
        out = subprocess.run([exe, "-v"], capture_output=True, text=True, timeout=10)
    except Exception:
        return None
    text = out.stdout + "\n" + out.stderr
    m = re.search(r"ngspice-(\d+(?:\.\d+)*)", text, re.IGNORECASE)
    return m.group(0) if m else text.strip().splitlines()[0] if text.strip() else None


def render_header(corner: str | None = None, pin: dict[str, Any] | None = None,
                  pdk_root: Path | None = None) -> str:
    pin = pin or load_pin()
    corner = corner or pin["pdk"]["default_corner"]
    lib = resolve_ngspice_lib(pin, pdk_root)
    # ngspice .lib paths with spaces break; keep absolute path quoted via include trick
    return "\n".join(
        [
            "* Auto-generated SKY130 header — do not edit by hand",
            f"* PDK variant={pin['pdk']['variant']} corner={corner} volare={pin['pdk']['volare_hash']}",
            ".options savecurrents",
            f".lib '{lib}' {corner}",
            "",
        ]
    )


def write_deck(body: str, *, corner: str | None = None, workdir: Path | None = None,
               name: str = "deck.spice") -> Path:
    """Write a complete deck = header + body into workdir (or a temp dir)."""
    header = render_header(corner=corner)
    if workdir is None:
        workdir = Path(tempfile.mkdtemp(prefix="gdiffps_sky130_"))
    else:
        workdir.mkdir(parents=True, exist_ok=True)
    path = workdir / name
    path.write_text(header + body)
    return path


def parse_key_equals_value(text: str, keys: list[str]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for key in keys:
        pattern = rf"{re.escape(key)}\s*=\s*([+\-0-9\.eE]+)"
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            try:
                metrics[key] = float(match.group(1))
            except ValueError:
                pass
    return metrics


def run_deck(
    deck_path: Path,
    *,
    metric_keys: list[str] | None = None,
    timeout_s: float | None = None,
    pin: dict[str, Any] | None = None,
) -> SimResult:
    pin = pin or load_pin()
    timeout = timeout_s if timeout_s is not None else float(pin["simulator"]["timeout_s"])
    exe = shutil.which("ngspice")
    if not exe:
        return SimResult(ok=False, returncode=-1, stdout="", stderr="", error="ngspice not on PATH")

    try:
        proc = subprocess.run(
            [exe, "-b", "-o", str(deck_path) + ".out", str(deck_path)],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(deck_path.parent),
        )
    except subprocess.TimeoutExpired:
        return SimResult(ok=False, returncode=-1, stdout="", stderr="", error=f"timeout after {timeout}s")
    except Exception as exc:
        return SimResult(ok=False, returncode=-1, stdout="", stderr="", error=str(exc))

    out_file = Path(str(deck_path) + ".out")
    file_text = out_file.read_text() if out_file.is_file() else ""
    combined = "\n".join([proc.stdout or "", proc.stderr or "", file_text])
    metrics = parse_key_equals_value(combined, metric_keys or [])
    ok = proc.returncode == 0 and "Error" not in combined[:5000]  # coarse; refined by callers
    # Prefer presence of requested metrics when keys were provided.
    if metric_keys:
        ok = proc.returncode == 0 and all(k in metrics for k in metric_keys)
    return SimResult(
        ok=ok,
        returncode=proc.returncode,
        stdout=proc.stdout or "",
        stderr=proc.stderr or "",
        metrics=metrics,
        raw_path=str(out_file) if out_file.is_file() else None,
        error=None if ok else "simulation failed or missing metrics",
    )


def environment_report(pin: dict[str, Any] | None = None) -> dict[str, Any]:
    pin = pin or load_pin()
    pdk_root = default_pdk_root()
    lib_ok = False
    lib_path = None
    try:
        lib_path = str(resolve_ngspice_lib(pin, pdk_root))
        lib_ok = True
    except FileNotFoundError as exc:
        lib_path = str(exc)

    return {
        "repo_root": str(REPO_ROOT),
        "pdk_root": str(pdk_root),
        "volare_hash": pin["pdk"]["volare_hash"],
        "variant": pin["pdk"]["variant"],
        "corner_default": pin["pdk"]["default_corner"],
        "ngspice_lib_ok": lib_ok,
        "ngspice_lib": lib_path,
        "ngspice_version": ngspice_version(),
        "ngspice_on_path": shutil.which("ngspice") is not None,
        "devices": pin["devices"],
        "operating_limits": pin["operating_limits"],
    }


def dump_environment_report(out_path: Path | None = None) -> dict[str, Any]:
    report = environment_report()
    out = out_path or (REPO_ROOT / "results" / "sky130" / "environment.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    return report
