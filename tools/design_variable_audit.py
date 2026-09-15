"""D0 / Test A / Test B: what the model actually sees.

Answers:
  1. Which sizing vector enters circuit-typed at rollout vs supervised vs replay?
  2. Does every action index map to one physical parameter role? Shared device rows?
  A. Does every tunable parameter change the graph/encoder input?

    python tools/design_variable_audit.py --out results/graphs/d0_design_variable.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from env.graph_utils import TOPOLOGY_PARAMS
from env.netlist_graph import (
    build_circuit_graph,
    nominal_params,
    params_numeric,
    sized_devices,
)
from models.circuit_encoder import CircuitTypedEncoder
from train_diffusion import PARAM_ROLES, action_to_params, param_role_name

SPEC = {"fc_ghz": 28.0, "tech": 0}
TOPOS = list(TOPOLOGY_PARAMS.keys())


def _feat_l2(a, b) -> float:
    return float((a.detach().reshape(-1) - b.detach().reshape(-1)).float().norm())


def sizing_provenance() -> dict:
    """Where params= is (not) passed into the encoder, plus the Phase-5 drop bug."""
    keys = TOPOLOGY_PARAMS["Loaded_Line"]
    raw = action_to_params(
        [0.3, 0.7, 0.2], "Loaded_Line", SPEC,
        sizing="log", bounds="electrical", switch_model="ideal",
    )
    n_numeric_raw = sum(isinstance(v, (int, float)) for v in raw.values())
    dropped = {k: float(raw[k]) for k in raw if isinstance(raw[k], (int, float))}
    recovered = params_numeric(raw)
    return {
        "encoder_default": (
            "CircuitEncoder.forward params=None → nominal_params() "
            "(bounds midpoint, not the chosen design)"
        ),
        "rl_rollout": {
            "file": "train_diffusion.py train()",
            "passes_chosen_params": False,
            "meaning": (
                "Encoder runs BEFORE the actor samples. Graph is the midpoint "
                "stand-in. Reward is computed on the sampled action. State and "
                "target are different circuits."
            ),
        },
        "rl_replay_reencode": {
            "passes_chosen_params": False,
            "meaning": (
                "Replay re-encode uses replay_spec only; still nominal_params. "
                "Critic/actor updates never see the design that produced the reward."
            ),
        },
        "phase5_supervised": {
            "file": "tools/supervised_loocv.py",
            "calls_params_kw": True,
            "action_to_params_values_are_strings": n_numeric_raw == 0,
            "legacy_isinstance_filter_kept": dropped,
            "legacy_filter_kept_sized_keys": [k for k in keys if k in dropped],
            "params_numeric_recovers": {k: recovered.get(k) for k in keys},
            "meaning": (
                "Phase 5 called gnn(..., params=row['params']) but stored only "
                "isinstance(v, (int, float)). action_to_params writes SPICE "
                "strings, so the stored dict was empty and the encoder saw "
                "resonant type+fc defaults — not the simulated design. Train "
                "R² ~0.2 was measured under that defect. Fixed by params_numeric."
            ),
        },
        "loocv_eval": {
            "passes_chosen_params": False,
            "meaning": "Eval encoder also runs before sampling; midpoint graph.",
        },
    }


def action_alignment() -> dict:
    rows = []
    shared = []
    for topo in TOPOS:
        sized = sized_devices(topo)
        keys = TOPOLOGY_PARAMS[topo]
        seen_dev = {}
        for i, (dname, pkey) in enumerate(sized):
            row = {
                "topology": topo,
                "action_index": i,
                "device": dname,
                "param_key": pkey,
                "role": param_role_name(pkey),
                "in_TOPOLOGY_PARAMS": pkey in keys,
            }
            rows.append(row)
            seen_dev.setdefault(dname, []).append(pkey)
        for dname, pkeys in seen_dev.items():
            if len(pkeys) > 1:
                shared.append({
                    "topology": topo,
                    "device": dname,
                    "params": pkeys,
                    "issue": (
                        "NodeFlowMatchingPolicy (default) emits one scalar per "
                        "row with a SHARED device embedding and NO role unless "
                        "--coupled-actions. These parameters are not distinguished."
                    ),
                })
    unique_map = len({(r["topology"], r["action_index"]) for r in rows}) == len(rows)
    return {
        "n_action_rows": len(rows),
        "bijective_index_to_param": unique_map,
        "shared_device_rows": shared,
        "default_actor_sees_role": False,
        "role_path": "--coupled-actions / CoupledNodeFlowMatchingPolicy",
        "roles_vocab": PARAM_ROLES,
        "rows": rows,
    }


def test_a_param_moves_input() -> dict:
    enc = CircuitTypedEncoder(use_pe=True)
    enc.eval()
    enc._cache_enabled = False
    out = []
    with torch.no_grad():
        for topo in TOPOS:
            base = dict(nominal_params(topo, SPEC))
            states = enc._states_to_encode(topo)
            z0 = enc(topo, SPEC, params=base).squeeze(0)
            g0_by_s = {
                s: build_circuit_graph(topo, SPEC, state=s, typed=True, params=base)
                for s in states
            }
            for key in TOPOLOGY_PARAMS[topo]:
                if key not in base:
                    out.append({
                        "topology": topo, "param": key,
                        "in_nominal": False, "graph_changed": False,
                        "z_changed": False, "rel_l2_x": 0.0, "rel_l2_z": 0.0,
                        "pass": False, "note": "key missing from nominal_params",
                    })
                    continue
                p = dict(base)
                v0 = float(params_numeric(p)[key])
                delta = 0.15 * v0 if abs(v0) > 1e-12 else 0.15
                p[key] = v0 + delta
                rels = []
                for s in states:
                    g1 = build_circuit_graph(topo, SPEC, state=s, typed=True, params=p)
                    dx = _feat_l2(g0_by_s[s]["device"].x, g1["device"].x)
                    nx = float(g0_by_s[s]["device"].x.reshape(-1).norm()) + 1e-12
                    rels.append(dx / nx)
                z1 = enc(topo, SPEC, params=p).squeeze(0)
                dz = _feat_l2(z0, z1)
                nz = float(z0.reshape(-1).norm()) + 1e-12
                graph_changed = max(rels) > 1e-6
                out.append({
                    "topology": topo,
                    "param": key,
                    "role": param_role_name(key),
                    "in_nominal": True,
                    "graph_changed": graph_changed,
                    "z_changed": dz / nz > 1e-6,
                    "rel_l2_x": max(rels),
                    "rel_l2_z": dz / nz,
                    "pass": graph_changed,
                })
            s0, s1 = states[0], states[min(1, len(states) - 1)]
            dsx = _feat_l2(g0_by_s[s0]["device"].x, g0_by_s[s1]["device"].x)
            nx = float(g0_by_s[s0]["device"].x.reshape(-1).norm()) + 1e-12
            out.append({
                "topology": topo,
                "param": "switch_state_0_to_1",
                "role": "state",
                "in_nominal": True,
                "graph_changed": dsx / nx > 1e-6,
                "z_changed": True,
                "rel_l2_x": dsx / nx,
                "rel_l2_z": 0.0,
                "pass": dsx / nx > 1e-6 or s0 == s1,
            })
    n = len(out)
    n_pass = sum(1 for r in out if r["pass"])
    return {
        "n": n,
        "n_pass": n_pass,
        "all_pass": n_pass == n,
        "failures": [r for r in out if not r["pass"]],
        "rows": out,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(
        REPO_ROOT, "results", "graphs", "d0_design_variable.json",
    ))
    args = ap.parse_args()
    report = {
        "sizing_provenance": sizing_provenance(),
        "action_alignment": action_alignment(),
        "test_A": test_a_param_moves_input(),
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(report, fh, indent=2)
    aa = report["action_alignment"]
    ta = report["test_A"]
    p5 = report["sizing_provenance"]["phase5_supervised"]
    print(json.dumps({
        "sizing": {
            "rl_passes_chosen_params": False,
            "phase5_calls_params_kw": p5["calls_params_kw"],
            "action_to_params_values_are_strings": p5["action_to_params_values_are_strings"],
            "legacy_filter_kept_sized_keys": p5["legacy_filter_kept_sized_keys"],
        },
        "shared_device_rows": aa["shared_device_rows"],
        "test_A_all_pass": ta["all_pass"],
        "test_A_failures": [
            f"{r['topology']}.{r['param']}" for r in ta["failures"]
        ],
        "out": args.out,
    }, indent=2))


if __name__ == "__main__":
    main()
