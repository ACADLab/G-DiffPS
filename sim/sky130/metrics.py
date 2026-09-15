"""Run SKY130-realizable phase-shifter decks and extract RF metrics."""
from __future__ import annotations

import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from sim.sky130 import SimResult, environment_report, run_deck, write_deck
from sim.sky130.realizable import DecodeResult, circuit_to_spice
from env.netlist_graph import Dev

PS_METRIC_KEYS = ["phase_deg", "il_db", "rl_db", "gain_err_db"]


@dataclass
class MetricsResult:
    ok: bool
    metrics: dict[str, float] = field(default_factory=dict)
    reason: Optional[str] = None
    spice_path: Optional[str] = None
    sim: Optional[SimResult] = None
    decode: Optional[DecodeResult] = None


def run_realizable_circuit(
    netlist: dict[str, Dev],
    *,
    topology_name: str = "custom",
    params: Optional[dict] = None,
    spec: Optional[dict] = None,
    state: int = 0,
    workdir: Optional[Path] = None,
    corner: Optional[str] = None,
) -> MetricsResult:
    """graph → SPICE → ngspice → metrics, with explicit failure reasons."""
    env = environment_report()
    if not env.get("ngspice_on_path"):
        return MetricsResult(ok=False, reason="ngspice_not_on_path")
    if not env.get("ngspice_lib_ok"):
        return MetricsResult(ok=False, reason="pdk_missing")

    decode = circuit_to_spice(
        netlist,
        topology_name=topology_name,
        params=params,
        spec=spec,
        state=state,
        include_pdk_header=False,  # write_deck injects header
        corner=corner,
    )
    if not decode.ok:
        return MetricsResult(
            ok=False,
            reason="decode_error:" + ";".join(decode.errors),
            decode=decode,
        )

    # Strip any accidental .lib — write_deck adds the pinned include.
    body = "\n".join(
        ln for ln in decode.spice.splitlines()
        if not ln.strip().lower().startswith(".lib")
    ) + "\n"

    if workdir is None:
        workdir = Path(tempfile.mkdtemp(prefix="gdiffps_sky130_ps_"))
    else:
        workdir = Path(workdir)
        workdir.mkdir(parents=True, exist_ok=True)

    deck = write_deck(body, corner=corner, workdir=workdir, name=f"{topology_name}_s{state}.spice")
    sim = run_deck(deck, metric_keys=PS_METRIC_KEYS)
    if not sim.ok:
        reason = "sim_failed"
        combined = (sim.stdout or "") + "\n" + (sim.stderr or "")
        if sim.raw_path:
            try:
                combined += "\n" + Path(sim.raw_path).read_text(errors="ignore")
            except Exception:
                pass
        # Re-parse from the .out file — batch mode writes prints there.
        from sim.sky130 import parse_key_equals_value
        file_metrics = parse_key_equals_value(combined, PS_METRIC_KEYS)
        if all(k in file_metrics for k in PS_METRIC_KEYS):
            return MetricsResult(
                ok=True, metrics=file_metrics, reason=None,
                spice_path=str(deck), sim=sim, decode=decode,
            )
        if "singular" in combined.lower() or "convergence" in combined.lower():
            reason = "dc_nonconvergence"
        elif any(k not in sim.metrics for k in PS_METRIC_KEYS):
            reason = "missing_metric"
        return MetricsResult(
            ok=False, reason=reason, metrics=dict(sim.metrics),
            spice_path=str(deck), sim=sim, decode=decode,
        )
    return MetricsResult(
        ok=True,
        metrics=dict(sim.metrics),
        reason=None,
        spice_path=str(deck),
        sim=sim,
        decode=decode,
    )


def aggregate_two_state(
    netlist: dict[str, Dev],
    *,
    topology_name: str,
    params: Optional[dict] = None,
    spec: Optional[dict] = None,
    ideal_step_deg: float = -22.5,
    workdir: Optional[Path] = None,
) -> MetricsResult:
    """Run states 0 and 1 and aggregate like PhaseShifterEnv."""
    from env.phaseshifter_env import aggregate_state_metrics

    per = []
    last = None
    for s in (0, 1):
        r = run_realizable_circuit(
            netlist, topology_name=topology_name, params=params,
            spec=spec, state=s,
            workdir=(Path(workdir) / f"s{s}") if workdir else None,
        )
        last = r
        per.append(r.metrics if r.ok else None)
        if not r.ok and r.reason in ("ngspice_not_on_path", "pdk_missing"):
            return r
    agg = aggregate_state_metrics([0, 1], per, ideal_step_deg)
    if agg is None:
        return MetricsResult(
            ok=False,
            reason="aggregate_failed",
            spice_path=last.spice_path if last else None,
            decode=last.decode if last else None,
        )
    return MetricsResult(ok=True, metrics=agg, reason=None)
