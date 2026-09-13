#!/usr/bin/env python3
"""Validate Bugbot area-reward findings are fixed in source + quantify impact.

Experiment A — source contract after the fix:
  * step() stashes topology/params before _evaluate_netlist
  * SAC / baselines pass a 7-tuple including params to parallel_eval_worker

Experiment B — reward impact of the old bug (no SPICE required):
  Compare WEIGHTS_AREA rewards when area uses:
    (1) Loaded_Line + nominal   ← old step() default
    (2) true topology + nominal ← old SAC path
    (3) true topology + sized   ← fixed path
  under identical RF metrics.

Usage:
  python3 tools/validate_area_stash_fix.py
"""
from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

# area_model → netlist_graph needs torch_geometric only for HeteroData typing;
# stub it so this validation can run without a full ML stack.
import types
if "torch_geometric" not in sys.modules:
    tg = types.ModuleType("torch_geometric")
    tg_data = types.ModuleType("torch_geometric.data")
    class HeteroData:  # noqa: N801
        pass
    tg_data.HeteroData = HeteroData
    sys.modules["torch_geometric"] = tg
    sys.modules["torch_geometric.data"] = tg_data

from env.reward import WEIGHTS_AREA, compute_sim_reward
from sim.area_model import estimate_area_mm2, nominal_params


RF_METRICS = {
    "rms_phase_err_deg": 4.0,
    "il_db": 2.5,
    "rl_db": 15.0,
    "gain_err_db": 0.4,
}

SPEC_2P4 = {
    "fc_ghz": 2.4,
    "rms_phase_err_deg": 10.0,
    "rms_gain_err_db": 1.0,
    "max_il_db": 5.0,
    "min_rl_db": 10.0,
    "max_area_mm2": 50.0,
    "tech": 0,
}


def check_source_contracts() -> dict:
    env_src = (REPO_ROOT / "env/phaseshifter_env.py").read_text()
    # After generate(), stash must precede _evaluate_netlist in step().
    gen_idx = env_src.find("llm_netlist_gen.generate(")
    stash_topo = env_src.find("self._last_topology = topology_name", gen_idx)
    stash_params = env_src.find("self._last_params = params_dict", gen_idx)
    eval_idx = env_src.find("self._evaluate_netlist(netlist_path)", gen_idx)
    step_ok = (
        gen_idx != -1
        and stash_topo != -1
        and stash_params != -1
        and eval_idx != -1
        and stash_topo < eval_idx
        and stash_params < eval_idx
    )

    sac_src = (REPO_ROOT / "baselines/train_sac.py").read_text()
    # Expect 7-tuple with params as last element in the SAC worker call.
    sac_ok = (
        "parallel_eval_worker(" in sac_src
        and "0.0, params)" in sac_src.replace(" ", "")
        or ("0.0, params)" in sac_src)
    )
    # More robust: parse AST for the call tuple length
    sac_tuple_len = None
    tree = ast.parse(sac_src)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "parallel_eval_worker":
            if node.args and isinstance(node.args[0], ast.Tuple):
                sac_tuple_len = len(node.args[0].elts)
                break
    sac_ok = sac_tuple_len == 7

    return {
        "step_stashes_before_evaluate": step_ok,
        "sac_worker_tuple_len": sac_tuple_len,
        "sac_passes_params": sac_ok,
        "pass": bool(step_ok and sac_ok),
    }


def reward_with_area(topology: str, params: dict | None, spec: dict) -> dict:
    area = estimate_area_mm2(topology, params, fc_ghz=float(spec["fc_ghz"]), tech=int(spec["tech"]))
    metrics = {**RF_METRICS, "area_mm2": area}
    r = float(compute_sim_reward(metrics, spec, weights=WEIGHTS_AREA))
    return {"topology": topology, "area_mm2": area, "reward": r, "params_mode": "sized" if params else "nominal"}


def impact_matrix() -> dict:
    # Sized All_Pass at 2.4 GHz — deliberately larger lengths than midpoint.
    sized = {
        **nominal_params("All_Pass", {"fc_ghz": 2.4}),
        "L_apA_nh": "8.0",
        "L_apB_nh": "8.0",
    }
    # Reflection_Type is the large footprint at low frequency.
    cases = {
        "bug_step_default_Loaded_Line_nominal": reward_with_area("Loaded_Line", None, SPEC_2P4),
        "bug_sac_true_topo_nominal": reward_with_area("All_Pass", None, SPEC_2P4),
        "fixed_true_topo_sized": reward_with_area("All_Pass", sized, SPEC_2P4),
        "reflection_type_nominal": reward_with_area("Reflection_Type", None, SPEC_2P4),
    }
    fixed = cases["fixed_true_topo_sized"]["reward"]
    deltas = {
        name: cases[name]["reward"] - fixed
        for name in cases
        if name != "fixed_true_topo_sized"
    }
    return {"spec": SPEC_2P4, "rf_metrics": RF_METRICS, "cases": cases, "reward_delta_vs_fixed": deltas}


def main():
    contracts = check_source_contracts()
    impact = impact_matrix()
    out = {
        "experiment": "validate_area_stash_fix",
        "bugbot_findings": [
            "step() missing _last_topology/_last_params",
            "SAC 5-tuple omitting params",
        ],
        "source_contracts": contracts,
        "reward_impact": impact,
    }
    path = REPO_ROOT / "results" / "area_fix" / "validate_area_stash_fix.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2) + "\n")

    print("Source contracts:", json.dumps(contracts, indent=2))
    print("\nReward impact @ 2.4 GHz (identical RF metrics):")
    for name, case in impact["cases"].items():
        print(f"  {name:<42} area={case['area_mm2']:8.2f} mm2  reward={case['reward']:+.4f}")
    print("\nDeltas vs fixed path:")
    for name, d in impact["reward_delta_vs_fixed"].items():
        print(f"  {name:<42} Δreward={d:+.4f}")
    print(f"\nWrote {path}")
    print("PASS" if contracts["pass"] else "FAIL")
    sys.exit(0 if contracts["pass"] else 1)


if __name__ == "__main__":
    main()
