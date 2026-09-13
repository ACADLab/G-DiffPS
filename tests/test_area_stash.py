"""Regression: area term must use the topology/params under evaluation."""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from env.phaseshifter_env import PhaseShifterEnv


def test_evaluate_netlist_uses_stashed_topology_and_params(tmp_path):
    env = PhaseShifterEnv()
    env.current_spec = {
        "fc_ghz": 2.4,
        "bw_pct": 20,
        "phase_coverage_deg": 360,
        "phase_bits": 1,
        "rms_phase_err_deg": 10,
        "rms_gain_err_db": 1,
        "max_il_db": 5,
        "min_rl_db": 10,
        "vdd": 1.8,
        "pmax_mw": 15,
        "tech": 0,
        "app": 2,
        "max_area_mm2": 100.0,
    }
    nl = tmp_path / "empty.sp"
    nl.write_text("* no STATE_TABLE\n.end\n")

    captured = {}

    def fake_estimate(topology, params=None, fc_ghz=28.0, **kwargs):
        captured["topology"] = topology
        captured["params"] = params
        return 12.34

    fake_agg = {
        "rms_phase_err_deg": 1.0,
        "il_db": 1.0,
        "rl_db": 20.0,
        "gain_err_db": 0.1,
    }

    with patch("sim.ngspice_runner.run", return_value=fake_agg), patch(
        "sim.area_model.estimate_area_mm2", side_effect=fake_estimate
    ):
        env._last_topology = "All_Pass"
        env._last_params = {"L_apA_nh": "1.0", "C_brA_pf": "0.5"}
        agg, reward, *_ = env._evaluate_netlist(str(nl))

    assert agg["area_mm2"] == 12.34
    assert captured["topology"] == "All_Pass"
    assert captured["params"]["L_apA_nh"] == "1.0"
    assert isinstance(reward, float)


def test_evaluate_netlist_defaults_are_loaded_line_nominal(tmp_path):
    """Without stash, area falls back to Loaded_Line + nominal params."""
    env = PhaseShifterEnv()
    env.current_spec = {"fc_ghz": 28.0, "max_area_mm2": 50.0, "rms_phase_err_deg": 10,
                        "rms_gain_err_db": 1, "max_il_db": 5, "min_rl_db": 10}
    for attr in ("_last_topology", "_last_params"):
        if hasattr(env, attr):
            delattr(env, attr)

    nl = tmp_path / "empty.sp"
    nl.write_text("* no STATE_TABLE\n.end\n")
    captured = {}

    def fake_estimate(topology, params=None, fc_ghz=28.0, **kwargs):
        captured["topology"] = topology
        captured["params"] = params
        return 1.0

    with patch("sim.ngspice_runner.run", return_value={
        "rms_phase_err_deg": 1.0, "il_db": 1.0, "rl_db": 20.0, "gain_err_db": 0.1,
    }), patch("sim.area_model.estimate_area_mm2", side_effect=fake_estimate):
        env._evaluate_netlist(str(nl))

    assert captured["topology"] == "Loaded_Line"
    assert captured["params"] is None  # empty stash → nominal path


if __name__ == "__main__":
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        test_evaluate_netlist_uses_stashed_topology_and_params(Path(d))
        test_evaluate_netlist_defaults_are_loaded_line_nominal(Path(d))
    print("OK area stash regression")
