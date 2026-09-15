"""D0 / Test A / Test B regressions for design-variable semantics."""
from __future__ import annotations

import os
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from env.graph_utils import TOPOLOGY_PARAMS
from env.netlist_graph import params_numeric, sized_devices
from train_diffusion import action_to_params, param_role_name
from tools.design_variable_audit import (
    action_alignment,
    sizing_provenance,
    test_a_param_moves_input as run_test_a,
)
from tools.env_consistency import SPEC, evaluate_topology


def test_action_to_params_are_strings_and_params_numeric_recovers():
    raw = action_to_params(
        np.full(3, 0.5), "Loaded_Line", {"fc_ghz": 28.0, "tech": 0},
        sizing="log", bounds="electrical", switch_model="ideal",
    )
    assert all(isinstance(raw[k], str) for k in TOPOLOGY_PARAMS["Loaded_Line"])
    numeric = params_numeric(raw)
    for k in TOPOLOGY_PARAMS["Loaded_Line"]:
        assert k in numeric
        assert np.isfinite(numeric[k])


def test_legacy_isinstance_filter_drops_sized_keys():
    """The Phase 5 dataset builder kept only numeric values; those are all strings."""
    prov = sizing_provenance()
    p5 = prov["phase5_supervised"]
    assert p5["action_to_params_values_are_strings"]
    assert p5["legacy_filter_kept_sized_keys"] == []
    recovered = p5["params_numeric_recovers"]
    assert all(recovered[k] is not None for k in TOPOLOGY_PARAMS["Loaded_Line"])


def test_every_action_index_maps_to_one_param():
    aa = action_alignment()
    assert aa["bijective_index_to_param"]
    for row in aa["rows"]:
        assert row["in_TOPOLOGY_PARAMS"]
        assert row["role"] == param_role_name(row["param_key"])
    # Distinct roles that currently share a device embedding.
    shared = {(s["topology"], s["device"]): s["params"] for s in aa["shared_device_rows"]}
    assert ("Loaded_Line", "T_main") in shared
    assert set(shared[("Loaded_Line", "T_main")]) == {"L_quarter_mm", "Z0_line"}


def test_sized_devices_cover_topology_params():
    for topo, keys in TOPOLOGY_PARAMS.items():
        sized = [p for _, p in sized_devices(topo)]
        assert set(sized) == set(keys), (topo, sized, keys)


def test_A_every_param_moves_typed_encoder_input():
    ta = run_test_a()
    assert ta["all_pass"], [
        f"{r['topology']}.{r['param']} x={r['rel_l2_x']:.3g} z={r['rel_l2_z']:.3g}"
        for r in ta["failures"]
    ]


def test_nominal_good_consistent_all_topologies():
    failures = []
    spec_24 = dict(SPEC)
    spec_24["fc_ghz"] = 2.4
    for topo in TOPOLOGY_PARAMS:
        rec = evaluate_topology(topo, SPEC)
        if not rec["nominal_consistent"]:
            g, b = rec["good"], rec["bad"]
            failures.append({
                "topology": topo,
                "good_prior": g["prior_pass"],
                "good_sim": g["sim_ok"],
                "good_failed": g["prior_explain"]["failed"],
                "good_reward": g["reward"],
                "bad_reward": b["reward"],
            })
    rec24 = evaluate_topology("All_Pass", spec_24)
    if not rec24["nominal_consistent"]:
        g = rec24["good"]
        failures.append({
            "topology": "All_Pass@2.4",
            "good_prior": g["prior_pass"],
            "good_sim": g["sim_ok"],
            "good_failed": g["prior_explain"]["failed"],
            "good_reward": g["reward"],
            "bad_reward": rec24["bad"]["reward"],
        })
    assert not failures, failures
