"""Zero-shot LOOCV evaluation for a trained checkpoint.

Reports eligibility, prior pass, simulator success, strict compliance,
tolerant all_close, and physical reward separately. Fails closed on missing
checkpoints or missing eval pools (unless --legacy-seen-specs is set).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from env.phaseshifter_env import PhaseShifterEnv
from env.graph_utils import get_topology_graph, TOPOLOGY_PARAMS, SLOT_ACTION_DIM
from env.action_tokens import encode_action_tokens
from env.reward import (
    WEIGHTS_AREA, classify_attempt, compute_sim_reward,
)
from models.gnn_encoder import TopologyEncoder
from models.circuit_encoder import is_circuit_encoder, make_encoder, uses_param_nodes
from env.param_semantics import PARAM_CONTEXT_DIM
from models.diffusion_policy import (
    FlowMatchingPolicy, NodeFlowMatchingPolicy, CoupledNodeFlowMatchingPolicy,
)
from sim.physics_priors import check_physics_priors
from sim.mna_scorer import topology_admits_spec
from specset.schema import normalize_spec
from train_diffusion import (
    action_to_params, device_action_to_params, make_spice_netlist,
    parallel_eval_worker,
)

_DEFAULT_EVAL_SPECSET = os.path.join(REPO_ROOT, "specset", "specset_eval.json")


def _file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_state(module, path, device):
    state = torch.load(path, map_location=device, weights_only=True)
    if any(k.startswith("module.") for k in state):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}
    module.load_state_dict(state)


def load_models(run_dir, encoder, action_space, device, coupled_actions=False):
    gnn_path = os.path.join(run_dir, "gnn_encoder.pt")
    actor_path = os.path.join(run_dir, "actor.pt")
    if not os.path.isfile(gnn_path) or not os.path.isfile(actor_path):
        raise FileNotFoundError(
            f"Missing checkpoint(s) in {run_dir!r}: "
            f"gnn={os.path.isfile(gnn_path)} actor={os.path.isfile(actor_path)}. "
            "Refusing to evaluate an untrained / incomplete model."
        )

    if is_circuit_encoder(encoder):
        gnn = make_encoder(encoder).to(device)
    else:
        gnn = TopologyEncoder().to(device)

    coupled = bool(coupled_actions) or uses_param_nodes(encoder)
    if action_space == "device":
        if coupled:
            actor = CoupledNodeFlowMatchingPolicy(
                num_steps=10, role_dim=PARAM_CONTEXT_DIM,
            ).to(device)
        else:
            actor = NodeFlowMatchingPolicy(num_steps=10).to(device)
    else:
        actor = FlowMatchingPolicy(
            action_dim=SLOT_ACTION_DIM, graph_dim=64, num_steps=10
        ).to(device)

    _load_state(gnn, gnn_path, device)
    _load_state(actor, actor_path, device)
    gnn.eval()
    actor.eval()
    return gnn, actor, {
        "gnn_encoder_sha256": _file_sha256(gnn_path),
        "actor_sha256": _file_sha256(actor_path),
    }


def sample_action(actor, gnn, topo, spec, spec_norm, encoder, action_space, device,
                  topo_graphs=None, bounds="electrical", switch_model="ideal",
                  coupled_actions=False):
    with torch.no_grad():
        if action_space == "device":
            z, h_act, ctx = encode_action_tokens(
                gnn, topo, spec, encoder_name=encoder,
                bounds=bounds, switch_model=switch_model,
                gin_graph=None if topo_graphs is None else topo_graphs[topo],
                device=device,
            )
            coupled = bool(coupled_actions) or uses_param_nodes(encoder)
            if coupled:
                a = actor.sample(spec_norm, h_act, role=ctx).cpu().numpy()
            else:
                a = actor.sample(spec_norm, h_act).cpu().numpy()
            return a
        if is_circuit_encoder(encoder):
            z = gnn(
                topo, spec, return_device=False,
                bounds=bounds, switch_model=switch_model,
            )
            a = actor.sample(spec_norm, z).squeeze(0).cpu().numpy()
        else:
            g = topo_graphs[topo]
            z = gnn(g.x.to(device), g.edge_index.to(device))
            a = actor.sample(spec_norm, z).squeeze(0).cpu().numpy()
    return a


def _module_fingerprint(module) -> str:
    h = hashlib.sha256()
    with torch.no_grad():
        for p in module.parameters():
            h.update(p.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--held-out", required=True)
    ap.add_argument("--encoder", default="circuit",
                    choices=["gin", "circuit", "circuit-typed", "circuit-typed-param"])
    ap.add_argument("--action-space", default="device", choices=["slot", "device"])
    ap.add_argument("--fc-mode", default="spec")
    ap.add_argument("--bounds", default="electrical",
                    choices=["legacy", "electrical", "sky130"])
    ap.add_argument(
        "--switch-model", required=True, choices=["ideal", "realistic"],
        help="Required: ideal or realistic (no default)",
    )
    ap.add_argument("--coupled-actions", action="store_true")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--force-fc", type=float, default=None,
        help="Override every sampled spec's fc_ghz.",
    )
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument(
        "--eval-specset",
        default=_DEFAULT_EVAL_SPECSET if os.path.exists(_DEFAULT_EVAL_SPECSET) else None,
        help="Held-out eval pool JSON (default: specset/specset_eval.json if present)",
    )
    ap.add_argument(
        "--legacy-seen-specs", action="store_true",
        help="Explicitly allow falling back to the train pool (labeled legacy track).",
    )
    ap.add_argument("--expert-bonus-scale", type=float, default=0.0)
    ap.add_argument("--sim", default="spice", choices=["spice", "mna"])
    args = ap.parse_args()

    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    gnn, actor, ckpt_hashes = load_models(
        args.run, args.encoder, args.action_space, device,
        coupled_actions=args.coupled_actions,
    )
    fp_before = {
        "gnn": _module_fingerprint(gnn),
        "actor": _module_fingerprint(actor),
    }

    eval_path = args.eval_specset
    if eval_path and os.path.exists(eval_path):
        env = PhaseShifterEnv(
            restrict_to=[args.held_out],
            eval_specset_path=eval_path,
            pool="eval",
            expert_bonus_scale=args.expert_bonus_scale,
        )
        pool_label = "eval"
    elif args.legacy_seen_specs:
        print(
            "[Warning] --legacy-seen-specs: sampling from train pool "
            f"(eval path={eval_path!r})."
        )
        env = PhaseShifterEnv(
            restrict_to=[args.held_out],
            expert_bonus_scale=args.expert_bonus_scale,
        )
        pool_label = "train_legacy"
    else:
        raise FileNotFoundError(
            f"Primary evaluation requires --eval-specset "
            f"({eval_path!r} missing). Pass --legacy-seen-specs to opt into "
            "the labeled seen-spec reproduction track."
        )

    topo_graphs = None
    if args.encoder == "gin":
        topo_graphs = {args.held_out: get_topology_graph(args.held_out)}

    attempts = []
    n_eligible = n_prior = n_sim = n_strict = n_tolerant = 0
    physical_rewards = []

    for i in range(args.n):
        obs, _ = env.reset()
        spec = dict(env.current_spec)
        if args.force_fc is not None:
            spec["fc_ghz"] = float(args.force_fc)
            obs = normalize_spec(spec)
        eligible = topology_admits_spec(args.held_out, spec)
        if eligible:
            n_eligible += 1

        spec_norm = torch.tensor(obs, dtype=torch.float, device=device).unsqueeze(0)
        a = sample_action(
            actor, gnn, args.held_out, spec, spec_norm,
            args.encoder, args.action_space, device, topo_graphs,
            bounds=args.bounds, switch_model=args.switch_model,
            coupled_actions=args.coupled_actions,
        )
        if args.action_space == "device":
            params = device_action_to_params(
                a, args.held_out, spec, bounds=args.bounds,
                switch_model=args.switch_model,
            )
        else:
            params = action_to_params(
                a, args.held_out, spec, bounds=args.bounds,
                switch_model=args.switch_model,
            )

        prior_ok = check_physics_priors(
            args.held_out, params, spec["fc_ghz"],
            pmax_mw=float(spec.get("pmax_mw", 1e9)),
        )
        metrics = None
        physical_reward = -5.0
        if prior_ok:
            n_prior += 1
            if getattr(args, "sim", "spice") == "mna":
                from sim.mna_scorer import mna_evaluate
                r, agg = mna_evaluate(args.held_out, params, spec)
                metrics = agg
                if metrics is not None:
                    n_sim += 1
                    physical_reward = float(compute_sim_reward(
                        metrics, spec, weights=WEIGHTS_AREA, warmup_deg=0.0,
                    ))
                else:
                    physical_reward = float(r)
            else:
                nl = make_spice_netlist(
                    args.held_out, params, spec_dict=spec, fc_mode=args.fc_mode,
                )
                r, agg, _ = parallel_eval_worker(
                    (nl, spec, args.held_out, 0.0, [args.held_out], 0.0, params)
                )
                metrics = agg
                if metrics is not None:
                    n_sim += 1
                    physical_reward = float(compute_sim_reward(
                        metrics, spec, weights=WEIGHTS_AREA, warmup_deg=0.0,
                    ))
                else:
                    physical_reward = float(r)

        outcome = classify_attempt(
            topology=args.held_out,
            spec=spec,
            prior_pass=prior_ok,
            metrics=metrics,
            physical_reward=physical_reward,
            expert_bonus=0.0,
        )
        if outcome["strict_compliance"]:
            n_strict += 1
        if outcome["tolerant_all_close"]:
            n_tolerant += 1
        physical_rewards.append(physical_reward)
        attempts.append({
            "i": i,
            "fc_ghz": spec.get("fc_ghz"),
            "pmax_mw": spec.get("pmax_mw"),
            **outcome,
        })

    fp_after = {
        "gnn": _module_fingerprint(gnn),
        "actor": _module_fingerprint(actor),
    }
    if fp_before != fp_after:
        raise RuntimeError("Model weights changed during evaluation — freeze violated.")

    n = args.n
    n_elig_denom = max(n_eligible, 1)
    summary = {
        "held_out": args.held_out,
        "n": n,
        "pool": pool_label,
        "eval_specset": eval_path,
        "encoder": args.encoder,
        "action_space": args.action_space,
        "fc_mode": args.fc_mode,
        "bounds": args.bounds,
        "switch_model": args.switch_model,
        "coupled_actions": bool(args.coupled_actions),
        "force_fc_ghz": args.force_fc,
        "checkpoint_hashes": ckpt_hashes,
        "frozen_ok": True,
        "eligible_rate": n_eligible / n,
        "prior_pass_rate": n_prior / n,
        "prior_pass_rate_eligible": n_prior / n_elig_denom,
        "sim_success_rate": n_sim / n,
        "strict_compliance": n_strict / n,
        "strict_compliance_eligible": n_strict / n_elig_denom,
        "tolerant_all_close": n_tolerant / n,
        # Legacy alias kept for older matrix CSV parsers — means strict now.
        "compliance": n_strict / n,
        "best_physical_reward": float(max(physical_rewards)) if physical_rewards else -5.0,
        "mean_physical_reward": float(np.mean(physical_rewards)) if physical_rewards else -5.0,
        "attempts": attempts,
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(summary, fh, indent=2)
    # Compact stdout (omit per-attempt dump)
    print(json.dumps({k: v for k, v in summary.items() if k != "attempts"}, indent=2))


if __name__ == "__main__":
    main()
