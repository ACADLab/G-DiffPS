"""Milestone C0: manual SKY130 CircuitGraph → SPICE → metrics."""
from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

try:
    import pytest
except ImportError:  # pragma: no cover
    class pytest:  # type: ignore
        @staticmethod
        def skip(msg):
            raise RuntimeError(f"SKIP: {msg}")
        @staticmethod
        def fixture(*a, **k):
            def deco(fn):
                return fn
            return deco

from sim.sky130 import environment_report
from sim.sky130.realizable import circuit_to_spice, loaded_line_sky130_graph
from sim.sky130.metrics import run_realizable_circuit, aggregate_two_state


def _env():
    os.environ.setdefault("PDK_ROOT", str(REPO_ROOT / "pdk"))
    return environment_report()


@pytest.fixture(scope="module")
def env():
    return _env()


def test_c0_decode_emits_nfet_instances():
    netlist, params = loaded_line_sky130_graph(w_um=5.0)
    dec = circuit_to_spice(
        netlist, topology_name="Loaded_Line", params=params,
        spec={"fc_ghz": 1.0, "bw_pct": 20.0}, state=1,
        include_pdk_header=False,
    )
    assert dec.ok, dec.errors
    assert "sky130_fd_pr__nfet_01v8" in dec.spice
    assert "XM_in_path" in dec.spice or "M_in_path" in dec.spice
    assert "T_main" in dec.spice
    assert "phase_deg" in dec.spice


def test_c0_simulate_or_skip(env=None):
    env = env or _env()
    if not env["ngspice_on_path"] or not env["ngspice_lib_ok"]:
        pytest.skip("SKY130 environment incomplete")

    netlist, params = loaded_line_sky130_graph(w_um=5.0, c_load_pf=0.5)
    spec = {"fc_ghz": 1.0, "bw_pct": 20.0, "tech": 0}
    work = REPO_ROOT / "results/sky130/runs/c0_loaded_line"
    r0 = run_realizable_circuit(
        netlist, topology_name="Loaded_Line", params=params,
        spec=spec, state=0, workdir=work / "s0",
    )
    assert r0.ok, f"state0 failed: {r0.reason} metrics={r0.metrics}"
    for k in ("phase_deg", "il_db", "rl_db"):
        assert k in r0.metrics
    assert abs(r0.metrics["il_db"]) < 40.0
    # Unloaded state can show very high return loss (>100 dB); that is success.
    assert float(r0.metrics["rl_db"]) > 0.0
    print("state0 metrics", r0.metrics)


def test_c0_two_state_aggregate_or_skip(env=None):
    env = env or _env()
    if not env["ngspice_on_path"] or not env["ngspice_lib_ok"]:
        pytest.skip("SKY130 environment incomplete")

    netlist, params = loaded_line_sky130_graph(w_um=5.0)
    spec = {"fc_ghz": 1.0, "bw_pct": 20.0}
    work = REPO_ROOT / "results/sky130/runs/c0_loaded_line_agg"
    agg = aggregate_two_state(
        netlist, topology_name="Loaded_Line", params=params,
        spec=spec, ideal_step_deg=-22.5, workdir=work,
    )
    assert agg.ok, agg.reason
    assert "rms_phase_err_deg" in agg.metrics
    assert "il_db" in agg.metrics


if __name__ == "__main__":
    env = _env()
    test_c0_decode_emits_nfet_instances()
    print("OK decode")
    try:
        test_c0_simulate_or_skip(env)
        print("OK simulate")
        test_c0_two_state_aggregate_or_skip(env)
        print("OK two-state aggregate")
    except RuntimeError as e:
        if str(e).startswith("SKIP:"):
            print(e)
        else:
            raise
    print("OK C0 smoke")
