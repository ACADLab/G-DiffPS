"""Ngspice subprocess wrapper.

Metric-agnostic: caller supplies the list of metric keys to extract from
ngspice's stdout. Each metric must be printed by the template's .control
block as `key = value` (case-insensitive).

Multi-state support (added Step 3 of multi-state arc):
    Caller may pass `param_overrides` as a list of dicts. Each dict maps
    parameter name -> value string (may be a `{ref}` symbol that ngspice
    resolves natively). For each override dict, the runner generates a
    derivative netlist with a trailing `.PARAM` block (last-wins semantics
    confirmed in ngspice 42) and simulates it. Returns a list of per-state
    metric dicts in the same order as the input overrides.

    When param_overrides is None, behavior is identical to before:
    single sim, returns a single dict (or None on failure). This keeps
    existing callers (PhaseShifterEnv.step, smoke tests) working unchanged.
"""

import subprocess
import re
import os
import tempfile


# Default metric set used when caller does not specify metric_keys.
# Reflects the phase-shifter framework's four core metrics; see env/
# phaseshifter_env.py for PS_METRIC_KEYS which is what every live caller
# passes explicitly.
DEFAULT_METRIC_KEYS = ["phase_deg", "il_db", "rl_db", "gain_err_db"]

# Metrics that should be defaulted (rather than failing the run) if missing.
# Nice-to-have metrics that have a sensible "bad" default value.
SOFT_DEFAULTS = {
    "pm_deg":      0.0,
    "pwr_w":       1.0,    # high power penalty if measurement missed
    "gain_err_db": 99.0,   # large gain error if missed
    "rl_db":       0.0,    # zero return loss = bad if missed
}

# Metrics that, if missing, must hard-fail the run (return None overall).
HARD_REQUIRED = {
    "phase_deg", "il_db",                     # phase-shifter essentials
}


def _run_single(netlist_path: str, metric_keys: list[str]) -> dict | None:
    """Single ngspice invocation. Returns metrics dict or None on failure.

    This is the original `run` logic, refactored out so that the new
    multi-state path can reuse it per-state.
    """
    try:
        result = subprocess.run(
            ["ngspice", netlist_path],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except subprocess.TimeoutExpired:
        print(f"[Timeout] Ngspice hung on {netlist_path}")
        return None
    except Exception as e:
        print(f"[Error] Ngspice failed: {e}")
        return None

    content = result.stdout + "\n" + result.stderr

    metrics = {}
    for key in metric_keys:
        pattern = rf"{re.escape(key)}\s*=\s*([+\-0-9\.eE]+)"
        match = re.search(pattern, content, re.IGNORECASE)
        if match:
            try:
                metrics[key] = float(match.group(1))
            except ValueError:
                metrics[key] = None
        else:
            metrics[key] = None

    # Hard-fail if any essential metric in the requested set is missing
    for key in metric_keys:
        if key in HARD_REQUIRED and metrics.get(key) is None:
            return None

    # Apply soft defaults for missing nice-to-have metrics
    for key in metric_keys:
        if metrics.get(key) is None and key in SOFT_DEFAULTS:
            metrics[key] = SOFT_DEFAULTS[key]

    # ngspice reports supply current as negative; flip the power sign so
    # downstream consumers always see a positive dissipation number.
    if "pwr_w" in metrics and metrics["pwr_w"] is not None:
        metrics["pwr_w"] = abs(metrics["pwr_w"])

    return metrics


def _inject_overrides(netlist_path: str, overrides: dict) -> str:
    """Write a derivative netlist with an appended .PARAM override block.

    The override line is inserted just before `.end` so ngspice parses it.
    If `.end` is absent (malformed template), the override is appended at
    the end and `.end` is added.

    Returns the path to the temp netlist. Caller is responsible for cleanup.
    """
    with open(netlist_path, "r") as f:
        original = f.read()

    # Build the override .PARAM line.
    # Values may be symbolic (e.g. '{R_off}') or literal ('10k', '0.04').
    # Either form is valid in a .PARAM line — ngspice resolves at parse time.
    if overrides:
        override_tokens = " ".join(f"{k}={v}" for k, v in overrides.items())
        override_line = f"\n* === STATE OVERRIDE (runner-injected) ===\n.PARAM {override_tokens}\n"
    else:
        override_line = ""

    # Find the last `.end` directive (case-insensitive, on its own line).
    # We insert before it so ngspice sees the override before terminating.
    lines = original.splitlines()
    end_idx = None
    for i in range(len(lines) - 1, -1, -1):
        if lines[i].strip().lower() == ".end":
            end_idx = i
            break

    if end_idx is not None:
        out_lines = lines[:end_idx] + [override_line.rstrip("\n")] + lines[end_idx:]
        out_text = "\n".join(out_lines) + "\n"
    else:
        # Defensive: malformed template. Append and add .end.
        out_text = original.rstrip("\n") + override_line + "\n.end\n"

    fd, tmp_path = tempfile.mkstemp(suffix=".sp", prefix="state_")
    with os.fdopen(fd, "w") as f:
        f.write(out_text)
    return tmp_path


def run(
    netlist_path: str,
    metric_keys: list[str] | None = None,
    param_overrides: list[dict] | None = None,
) -> dict | list | None:
    """Run ngspice on the netlist and parse requested metrics from stdout.

    Args:
        netlist_path: path to the .sp file. Template must include a .control
            block that prints each metric as `key = value`.
        metric_keys: list of metric names to extract. Defaults to the four
            phase-shifter core metrics if not provided.
        param_overrides: optional list of per-state parameter override dicts.
            When provided, the runner generates one derivative netlist per
            entry (with the overrides appended as a trailing .PARAM block,
            using ngspice's last-wins semantics), simulates each, and
            returns a list of per-state metric dicts in the same order.
            When None, behavior is unchanged: single sim, single dict result.

    Returns:
        - If param_overrides is None: dict mapping metric name -> float,
          or None if simulation failed or any HARD_REQUIRED metric is missing.
        - If param_overrides is provided: list with one entry per override
          dict, each entry being a metric dict or None on failure. The list
          itself is never None. (Caller must inspect entries individually.)
    """
    if metric_keys is None:
        metric_keys = DEFAULT_METRIC_KEYS

    # Single-state path (backward compatible)
    if param_overrides is None:
        return _run_single(netlist_path, metric_keys)

    # Multi-state path
    results: list[dict | None] = []
    for override_dict in param_overrides:
        tmp_path = _inject_overrides(netlist_path, override_dict)
        try:
            metrics = _run_single(tmp_path, metric_keys)
        finally:
            # Best-effort cleanup of temp netlist + ngspice .lis dump
            try:
                os.remove(tmp_path)
            except Exception:
                pass
            lis_path = tmp_path + ".lis"
            if os.path.exists(lis_path):
                try:
                    os.remove(lis_path)
                except Exception:
                    pass
        results.append(metrics)

    return results
