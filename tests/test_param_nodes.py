"""D1 gate: parameter-role action tokens and R3 parameter nodes."""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from env.action_tokens import encode_action_tokens
from env.graph_utils import TOPOLOGY_PARAMS
from env.netlist_graph import build_circuit_graph, params_numeric, sized_devices
from env.param_semantics import (
    PARAM_CONTEXT_DIM, N_PARAM_ROLES, invert_allpass_controls,
    param_context_matrix, param_role_name, param_window,
)
from models.circuit_encoder import CircuitParamEncoder, CircuitTypedEncoder, make_encoder
from train_diffusion import action_to_params

SPEC = {"fc_ghz": 28.0, "tech": 0}


def _mid_params(topo):
    keys = TOPOLOGY_PARAMS[topo]
    return action_to_params(
        np.full(len(keys), 0.5), topo, SPEC,
        sizing="log", bounds="electrical", switch_model="ideal",
    )


def _action_params(topo, action):
    return action_to_params(
        action, topo, SPEC, sizing="log", bounds="electrical", switch_model="ideal",
    )


def test_every_action_maps_to_one_param():
    for topo, keys in TOPOLOGY_PARAMS.items():
        sized = sized_devices(topo)
        assert len(sized) == len(keys)
        assert {p for _, p in sized} == set(keys)


def test_context_has_role_and_bounds():
    ctx = param_context_matrix("Loaded_Line", SPEC, params=_mid_params("Loaded_Line"))
    assert ctx.shape[1] == PARAM_CONTEXT_DIM
    assert np.allclose(ctx[:, :N_PARAM_ROLES].sum(axis=1), 1.0)
    roles = [param_role_name(p) for _, p in sized_devices("Loaded_Line")]
    assert "length" in roles and "impedance" in roles


def test_same_device_params_have_different_d1a_contexts():
    topo = "Loaded_Line"
    sized = sized_devices(topo)
    owners = [d for d, _ in sized]
    assert owners.count("T_main") == 2
    ctx = param_context_matrix(topo, SPEC, params=_mid_params(topo))
    i_len = [i for i, (_, p) in enumerate(sized) if p == "L_quarter_mm"][0]
    i_z0 = [i for i, (_, p) in enumerate(sized) if p == "Z0_line"][0]
    assert not np.allclose(ctx[i_len], ctx[i_z0])
    assert not np.allclose(ctx[i_len, :N_PARAM_ROLES], ctx[i_z0, :N_PARAM_ROLES])


def test_allpass_roles_are_centre_ratio_coupling():
    assert param_role_name("L_apA_nh") == "centre"
    assert param_role_name("L_apB_nh") == "ratio"
    assert param_role_name("C_brA_pf") == "centre"
    assert param_role_name("C_brB_pf") == "ratio"
    assert param_role_name("C_cA_pf") == "coupling"
    assert param_role_name("C_cB_pf") == "coupling"


def test_allpass_windows_are_semantic_not_independent_physical():
    """Ratio window is the section-ratio range, not a physical L_B sweep."""
    lo_r, hi_r, is_log = param_window("All_Pass", "L_apB_nh", SPEC)
    assert is_log
    assert lo_r < 2.0 < hi_r
    lo_k, hi_k, is_log_k = param_window("All_Pass", "C_cA_pf", SPEC)
    assert not is_log_k
    assert abs(lo_k - 1.2) < 1e-6
    assert hi_k >= 4.0 - 1e-6


def test_allpass_ratio_action_moves_ratio_token_only():
    keys = TOPOLOGY_PARAMS["All_Pass"]
    sized = sized_devices("All_Pass")
    i_centre = [i for i, (_, p) in enumerate(sized) if p == "L_apA_nh"][0]
    i_ratio = [i for i, (_, p) in enumerate(sized) if p == "L_apB_nh"][0]
    mid = np.full(len(keys), 0.5)
    high = mid.copy()
    high[keys.index("L_apB_nh")] = 1.0
    ctx0 = param_context_matrix("All_Pass", SPEC, params=_action_params("All_Pass", mid))
    ctx1 = param_context_matrix("All_Pass", SPEC, params=_action_params("All_Pass", high))
    v0_r, v1_r = ctx0[i_ratio, N_PARAM_ROLES], ctx1[i_ratio, N_PARAM_ROLES]
    v0_c, v1_c = ctx0[i_centre, N_PARAM_ROLES], ctx1[i_centre, N_PARAM_ROLES]
    assert v1_r > v0_r + 0.2
    assert abs(v1_c - v0_c) < 0.15


def test_allpass_centre_action_moves_centre_token():
    keys = TOPOLOGY_PARAMS["All_Pass"]
    sized = sized_devices("All_Pass")
    i_centre = [i for i, (_, p) in enumerate(sized) if p == "L_apA_nh"][0]
    i_ratio = [i for i, (_, p) in enumerate(sized) if p == "L_apB_nh"][0]
    mid = np.full(len(keys), 0.5)
    high = mid.copy()
    high[keys.index("L_apA_nh")] = 1.0
    ctx0 = param_context_matrix("All_Pass", SPEC, params=_action_params("All_Pass", mid))
    ctx1 = param_context_matrix("All_Pass", SPEC, params=_action_params("All_Pass", high))
    assert ctx1[i_centre, N_PARAM_ROLES] > ctx0[i_centre, N_PARAM_ROLES] + 0.2
    assert abs(ctx1[i_ratio, N_PARAM_ROLES] - ctx0[i_ratio, N_PARAM_ROLES]) < 0.15


def test_invert_allpass_controls_roundtrip_midpoint():
    p = params_numeric(_mid_params("All_Pass"))
    sem = invert_allpass_controls(p)
    assert abs(sem["L_apB_nh"] - 5.0) < 0.05
    assert abs(sem["C_cA_pf"] - (p["C_cA_pf"] / p["C_brA_pf"])) < 1e-6


def test_param_nodes_attached_and_aligned():
    topo = "Loaded_Line"
    params = _mid_params(topo)
    g = build_circuit_graph(
        topo, SPEC, state=0, typed=True, params=params, param_nodes=True,
    )
    sized = sized_devices(topo)
    assert g["param"].x.size(0) == len(sized)
    assert list(g["param"].names) == [p for _, p in sized]
    assert list(g["param"].owners) == [d for d, _ in sized]


def test_r3_same_device_params_have_different_embeddings():
    enc = CircuitParamEncoder(use_pe=True)
    enc.eval()
    enc._cache_enabled = False
    topo = "Loaded_Line"
    params = params_numeric(_mid_params(topo))
    with torch.no_grad():
        z, h_param = enc(topo, SPEC, params=params, return_param=True)
    sized = sized_devices(topo)
    i_len = [i for i, (_, p) in enumerate(sized) if p == "L_quarter_mm"][0]
    i_z0 = [i for i, (_, p) in enumerate(sized) if p == "Z0_line"][0]
    assert h_param.size(0) == len(sized)
    assert not torch.allclose(h_param[i_len], h_param[i_z0])


def test_changing_z0_moves_z0_token_not_just_length():
    enc = CircuitParamEncoder(use_pe=True)
    enc.eval()
    enc._cache_enabled = False
    topo = "Loaded_Line"
    base = params_numeric(_mid_params(topo))
    alt = dict(base)
    alt["Z0_line"] = float(base["Z0_line"]) * 1.2
    with torch.no_grad():
        _, h0 = enc(topo, SPEC, params=base, return_param=True)
        _, h1 = enc(topo, SPEC, params=alt, return_param=True)
    sized = sized_devices(topo)
    i_len = [i for i, (_, p) in enumerate(sized) if p == "L_quarter_mm"][0]
    i_z0 = [i for i, (_, p) in enumerate(sized) if p == "Z0_line"][0]
    d_z0 = float((h0[i_z0] - h1[i_z0]).norm())
    d_len = float((h0[i_len] - h1[i_len]).norm())
    assert d_z0 > 1e-6
    assert d_z0 > d_len


def test_d1a_action_tokens_differ_for_shared_device():
    enc = CircuitTypedEncoder(use_pe=True)
    enc.eval()
    enc._cache_enabled = False
    topo = "Loaded_Line"
    params = params_numeric(_mid_params(topo))
    with torch.no_grad():
        z, h_act, ctx = encode_action_tokens(
            enc, topo, SPEC, encoder_name="circuit-typed", params=params,
        )
    sized = sized_devices(topo)
    i_len = [i for i, (_, p) in enumerate(sized) if p == "L_quarter_mm"][0]
    i_z0 = [i for i, (_, p) in enumerate(sized) if p == "Z0_line"][0]
    tok0 = torch.cat([h_act[i_len], ctx[i_len]], dim=-1)
    tok1 = torch.cat([h_act[i_z0], ctx[i_z0]], dim=-1)
    assert not torch.allclose(tok0, tok1)


def test_make_encoder_param_variant():
    enc = make_encoder("circuit-typed-param")
    assert isinstance(enc, CircuitParamEncoder)
    with torch.no_grad():
        z, h_p = enc("Switched_Line", SPEC, return_param=True)
    assert h_p.size(0) == len(sized_devices("Switched_Line"))
    assert z.shape[-1] == 64
