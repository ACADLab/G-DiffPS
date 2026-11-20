"""Evaluate-all joint topology + sizing via MNA (no ValueNet ranker).

For a spec, size all 6 topologies with the trained CFM policy, score each
with the differentiable MNA scorer at true fc, filter by physics priors
(including VM ``pmax_mw``), and return argmax among prior-feasible designs
(falling back to the full ranking if none pass).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from env.graph_utils import get_topology_graph, TOPOLOGY_PARAMS, SLOT_ACTION_DIM
from env.action_tokens import encode_action_tokens
from models.gnn_encoder import TopologyEncoder
from models.circuit_encoder import is_circuit_encoder, make_encoder, uses_param_nodes
from models.diffusion_policy import (
    FlowMatchingPolicy, NodeFlowMatchingPolicy, CoupledNodeFlowMatchingPolicy,
)
from env.param_semantics import PARAM_CONTEXT_DIM
from sim.mna_scorer import mna_evaluate
from sim.physics_priors import check_physics_priors
from train_diffusion import action_to_params, device_action_to_params
from inference_topology_select import normalize_spec  # wraps specset.schema.normalize_spec → torch
from specset.schema import SPEC_DIM

TOPOLOGY_NAMES = list(TOPOLOGY_PARAMS.keys())


def _run_config(run_dir):
    path = os.path.join(run_dir, "run_config.json")
    if not os.path.isfile(path):
        return {}
    with open(path) as fh:
        return json.load(fh)


def _resolve_flag(cli_value, run_cfg, key, default):
    """Prefer an explicit CLI flag; otherwise use the training run_config."""
    from_run = run_cfg.get(key)
    if cli_value is not None:
        if from_run is not None and cli_value != from_run:
            print(f"[warn] --{key.replace('_', '-')}={cli_value} differs from "
                  f"run_config {key}={from_run}", flush=True)
        return cli_value
    if from_run is not None:
        return from_run
    return default


def _load_state(module, path, device):
    try:
        state = torch.load(path, map_location=device, weights_only=True)
    except Exception:
        state = torch.load(path, map_location=device, weights_only=False)
    if any(k.startswith("module.") for k in state):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}
    # Strict load so a renamed layer cannot silently become random weights.
    missing, unexpected = module.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"Checkpoint mismatch for {path}: missing={list(missing)} "
            f"unexpected={list(unexpected)}"
        )


def load_policy(run_dir: str, encoder: str, action_space: str, device,
                coupled_actions: bool = False):
    if is_circuit_encoder(encoder):
        gnn = make_encoder(encoder).to(device)
    else:
        gnn = TopologyEncoder().to(device)
    coupled = bool(coupled_actions) or uses_param_nodes(encoder)
    if action_space == "device":
        if coupled:
            actor = CoupledNodeFlowMatchingPolicy(
                spec_dim=SPEC_DIM, num_steps=10, role_dim=PARAM_CONTEXT_DIM,
            ).to(device)
        else:
            actor = NodeFlowMatchingPolicy(spec_dim=SPEC_DIM, num_steps=10).to(device)
    else:
        actor = FlowMatchingPolicy(
            action_dim=SLOT_ACTION_DIM, spec_dim=SPEC_DIM, graph_dim=64, num_steps=10
        ).to(device)

    gnn_path = os.path.join(run_dir, "gnn_encoder.pt")
    actor_path = os.path.join(run_dir, "actor.pt")
    warned = False
    if os.path.exists(gnn_path):
        _load_state(gnn, gnn_path, device)
    else:
        print(f"[warn] missing {gnn_path}; using random encoder weights")
        warned = True
    if os.path.exists(actor_path):
        _load_state(actor, actor_path, device)
    else:
        print(f"[warn] missing {actor_path}; using random actor weights")
        warned = True
    gnn.eval()
    actor.eval()
    return gnn, actor, warned


def _sample_one(actor, gnn, topo, spec, spec_norm, encoder, action_space, device,
                topo_graphs, bounds="electrical", switch_model="ideal",
                coupled_actions=False):
    with torch.no_grad():
        if action_space == "device":
            z, h_act, ctx = encode_action_tokens(
                gnn, topo, spec, encoder_name=encoder,
                bounds=bounds, switch_model=switch_model,
                gin_graph=None if topo_graphs is None else topo_graphs.get(topo),
                device=device,
            )
            coupled = bool(coupled_actions) or uses_param_nodes(encoder)
            if coupled:
                a = actor.sample(spec_norm, h_act, role=ctx).cpu().numpy()
            else:
                a = actor.sample(spec_norm, h_act).cpu().numpy()
            return device_action_to_params(
                a, topo, spec, bounds=bounds, switch_model=switch_model,
            )
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
    return action_to_params(
        a, topo, spec, bounds=bounds, switch_model=switch_model,
    )


def evaluate_all(
    spec: dict,
    gnn,
    actor,
    encoder: str = "circuit",
    action_space: str = "device",
    device: torch.device | None = None,
    n_samples: int = 1,
    bounds: str = "electrical",
    switch_model: str = "ideal",
    coupled_actions: bool = False,
) -> list[dict]:
    """Size each topology (optionally multi-sample), MNA-score, sort best-first."""
    device = device or torch.device("cpu")
    spec_norm = normalize_spec(spec).to(device)
    topo_graphs = None
    if encoder == "gin":
        topo_graphs = {t: get_topology_graph(t) for t in TOPOLOGY_NAMES}
    # Match training: coupled_actions from run_config OR param-node encoders.
    coupled = bool(coupled_actions) or uses_param_nodes(encoder)

    pmax_mw = float(spec.get("pmax_mw", 1e9))
    results = []
    for topo in TOPOLOGY_NAMES:
        best = None
        for _ in range(n_samples):
            params = _sample_one(
                actor, gnn, topo, spec, spec_norm,
                encoder, action_space, device, topo_graphs,
                bounds=bounds, switch_model=switch_model,
                coupled_actions=coupled,
            )
            score, metrics = mna_evaluate(topo, params, spec)
            prior_ok = bool(check_physics_priors(
                topo, params, float(spec["fc_ghz"]), pmax_mw=pmax_mw,
            ))
            cand = {
                "topology": topo,
                "score": float(score),
                "prior_ok": prior_ok,
                "params": params,
                "metrics": {
                    k: (float(v) if isinstance(v, (int, float, np.floating)) else None)
                    for k, v in (metrics or {}).items()
                    if k != "per_state"
                },
                "switch_model": switch_model,
                "bounds": bounds,
            }
            # Prefer prior-feasible candidates; among equals, higher MNA score.
            if best is None:
                best = cand
            elif (cand["prior_ok"], cand["score"]) > (best["prior_ok"], best["score"]):
                best = cand
        results.append(best)
    # Rank: prior/pmax passers first, then by MNA score. If none pass, still
    # return a full ranking (honest fallback; winner will have prior_ok=False).
    results.sort(key=lambda r: (r["prior_ok"], r["score"]), reverse=True)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="Checkpoint directory")
    ap.add_argument("--encoder", default="circuit",
                    choices=["gin", "circuit", "circuit-typed", "circuit-typed-param"])
    ap.add_argument("--action-space", default="device", choices=["slot", "device"])
    ap.add_argument(
        "--switch-model", default=None, choices=["ideal", "realistic"],
        help="Must match training --switch-model (default: run_config)",
    )
    ap.add_argument("--bounds", default=None, choices=["legacy", "electrical"],
                    help="Must match training --bounds (default: run_config or electrical)")
    ap.add_argument("--spec", type=str, default=None, help="JSON spec string")
    ap.add_argument("--spec-idx", type=int, default=0)
    ap.add_argument("--n-samples", type=int, default=1)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    run_cfg = _run_config(args.run)
    args.bounds = _resolve_flag(args.bounds, run_cfg, "bounds", "electrical")
    args.switch_model = _resolve_flag(args.switch_model, run_cfg, "switch_model", None)
    if args.switch_model not in ("ideal", "realistic"):
        ap.error("--switch-model is required when run_config.json has no switch_model")

    device = torch.device(args.device)
    coupled = bool(run_cfg.get("coupled_actions", False)) or uses_param_nodes(args.encoder)
    gnn, actor, _ = load_policy(
        args.run, args.encoder, args.action_space, device,
        coupled_actions=coupled,
    )
    print(f"decode bounds={args.bounds} switch_model={args.switch_model}", flush=True)

    if args.spec:
        spec = json.loads(args.spec)
    else:
        from specset.schema import TRAIN_SPECSET_PATH, load_specset
        path = TRAIN_SPECSET_PATH
        dataset = load_specset(path)
        spec = dataset[args.spec_idx]["spec"]

    ranked = evaluate_all(
        spec, gnn, actor,
        encoder=args.encoder, action_space=args.action_space,
        device=device, n_samples=args.n_samples,
        bounds=args.bounds, switch_model=args.switch_model,
        coupled_actions=coupled,
    )
    print("\nSpec:", json.dumps(spec, indent=2))
    print(f"\nEvaluate-all ranking (MNA+prior/pmax, bounds={args.bounds}, "
          f"switch_model={args.switch_model}):")
    for i, r in enumerate(ranked):
        flag = "ok" if r["prior_ok"] else "FAIL_PRIOR"
        print(f"  {i+1}. {r['topology']:20s}  score={r['score']:+.4f}  prior={flag}")
    out = {
        "switch_model": args.switch_model,
        "bounds": args.bounds,
        "ranked": ranked,
        "spec": spec,
    }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
