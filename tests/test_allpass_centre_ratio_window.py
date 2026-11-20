"""All_Pass centre value_01 must use the drawn section ratio, not a fixed nominal."""
import os
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from env.graph_utils import TOPOLOGY_PARAMS
from env.netlist_graph import sized_devices
from env.param_semantics import N_PARAM_ROLES, param_context_matrix
from train_diffusion import action_to_params

SPEC = {"fc_ghz": 28.0, "tech": 0}


def _params(action):
    return action_to_params(
        action, "All_Pass", SPEC, sizing="log",
        bounds="electrical", switch_model="ideal",
    )


def test_centre_value01_tracks_action_at_extreme_ratio():
    """At max ratio + mid centre action, value_01 must stay near 0.5."""
    keys = TOPOLOGY_PARAMS["All_Pass"]
    sized = sized_devices("All_Pass")
    i_centre = [i for i, (_, p) in enumerate(sized) if p == "L_apA_nh"][0]
    a = np.full(len(keys), 0.5)
    a[keys.index("L_apB_nh")] = 1.0  # max ratio
    ctx = param_context_matrix("All_Pass", SPEC, params=_params(a))
    v01 = float(ctx[i_centre, N_PARAM_ROLES])
    assert abs(v01 - 0.5) < 0.05, f"centre value_01={v01} drifted from action 0.5"


def test_centre_value01_at_max_action_with_max_ratio():
    """Max centre action should map near 1.0 even when ratio is extreme."""
    keys = TOPOLOGY_PARAMS["All_Pass"]
    sized = sized_devices("All_Pass")
    i_centre = [i for i, (_, p) in enumerate(sized) if p == "L_apA_nh"][0]
    a = np.full(len(keys), 0.5)
    a[keys.index("L_apA_nh")] = 1.0
    a[keys.index("L_apB_nh")] = 1.0
    ctx = param_context_matrix("All_Pass", SPEC, params=_params(a))
    v01 = float(ctx[i_centre, N_PARAM_ROLES])
    assert v01 > 0.85, f"expected near-1.0 centre value_01, got {v01}"
