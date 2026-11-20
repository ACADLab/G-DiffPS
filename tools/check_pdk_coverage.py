"""Check PDK-backed validity and six-family coverage across frequency.

Audits:
1. Which of the six families have SKY130 realizable netlists vs ideal templates.
2. Tests Loaded_Line_SKY130 simulation across carrier frequencies (1.0 GHz, 2.4 GHz, 3.0 GHz, 10.0 GHz, 28.0 GHz).
3. Reports metrics and limitations (switch off-isolation, insertion loss, parasitics at mmWave).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from env.graph_utils import TOPOLOGY_PARAMS
TOPOLOGY_NAMES = list(TOPOLOGY_PARAMS.keys())
from env.netlist_graph import TOPOLOGY_NETLIST
from sim.sky130 import environment_report
from sim.sky130.realizable import (
    circuit_to_spice, loaded_line_sky130_graph,
)
from sim.sky130.metrics import aggregate_two_state


def audit_six_family_coverage() -> dict:
    env = environment_report()
    coverage = {}
    for topo in TOPOLOGY_NAMES:
        netlist = TOPOLOGY_NETLIST.get(topo, {})
        has_pdk_models = any(getattr(dev, "pdk_model", None) for dev in netlist.values())
        has_switches = any(getattr(dev, "is_switch", False) for dev in netlist.values())
        dev_types = [dev.dtype for dev in netlist.values()]
        coverage[topo] = {
            "has_pdk_netlist": has_pdk_models,
            "has_switches": has_switches,
            "device_types": dev_types,
            "status": "PDK-backed" if has_pdk_models else "Ideal/abstract model only",
        }

    # Loaded_Line_SKY130 specialized graph
    coverage["Loaded_Line_SKY130"] = {
        "has_pdk_netlist": True,
        "has_switches": True,
        "device_types": ["TLine", "R_switch", "C", "R_switch", "C"],
        "pdk_switch_model": "sky130_fd_pr__nfet_01v8",
        "status": "PDK-backed (implemented in sim/sky130/realizable.py)",
    }
    return {
        "environment": env,
        "family_coverage": coverage,
    }


def test_frequencies_loaded_line(
    freqs: list[float] = [1.0, 2.4, 3.0, 10.0, 28.0],
) -> dict:
    results = {}
    workdir = REPO_ROOT / "results/sky130/runs/frequency_sweep"
    netlist, params = loaded_line_sky130_graph(w_um=5.0, c_load_pf=0.2)

    for fc in freqs:
        spec = {"fc_ghz": fc, "bw_pct": 20.0}
        lam4 = 47.43 / fc
        # Scale L_quarter to lambda/4 at fc
        p = dict(params)
        p["L_quarter_mm"] = lam4
        out_dir = workdir / f"fc_{fc}ghz"
        agg = aggregate_two_state(
            netlist, topology_name="Loaded_Line", params=p,
            spec=spec, ideal_step_deg=-22.5, workdir=out_dir,
        )
        results[f"{fc}_ghz"] = {
            "ok": agg.ok,
            "reason": agg.reason,
            "metrics": agg.metrics,
        }
    return results


def main():
    report = audit_six_family_coverage()
    print("\n--- PDK Family Coverage Audit ---")
    print(json.dumps(report, indent=2))

    print("\n--- Loaded_Line SKY130 Frequency Sweep (1 to 28 GHz) ---")
    sweep = test_frequencies_loaded_line()
    print(json.dumps(sweep, indent=2))

    out_file = REPO_ROOT / "results/graphs/pdk_coverage_audit.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps({"coverage": report, "frequency_sweep": sweep}, indent=2))
    print(f"\nSaved report to {out_file}")


if __name__ == "__main__":
    main()
