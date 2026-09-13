"""Zero-shot LOOCV evaluation for a trained checkpoint."""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from env.phaseshifter_env import PhaseShifterEnv
from env.graph_utils import get_topology_graph, TOPOLOGY_PARAMS, SLOT_ACTION_DIM, gin_device_rows
from env.netlist_graph import sized_devices, device_names
from models.gnn_encoder import TopologyEncoder
from models.circuit_encoder import CircuitEncoder
from models.diffusion_policy import (
    FlowMatchingPolicy, NodeFlowMatchingPolicy,
)
from sim.physics_priors import check_physics_priors
from specset.schema import normalize_spec
from train_diffusion import (
    action_to_params, device_action_to_params, make_spice_netlist,
    parallel_eval_worker,
)

COMPLIANCE_THRESHOLD = 0.5


def _load_state(module, path, device):
    state = torch.load(path, map_location=device, weights_only=True)
    if any(k.startswith("module.") for k in state):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}
    module.load_state_dict(state)


def load_models(run_dir, encoder, action_space, device):
    if encoder == "circuit":
        gnn = CircuitEncoder().to(device)
    else:
        gnn = TopologyEncoder().to(device)

    if action_space == "device":
        actor = NodeFlowMatchingPolicy(num_steps=10).to(device)
    else:
        actor = FlowMatchingPolicy(
            action_dim=SLOT_ACTION_DIM, graph_dim=64, num_steps=10
        ).to(device)

    gnn_path = os.path.join(run_dir, "gnn_encoder.pt")
    actor_path = os.path.join(run_dir, "actor.pt")
    if os.path.exists(gnn_path):
        _load_state(gnn, gnn_path, device)
    if os.path.exists(actor_path):
        _load_state(actor, actor_path, device)
    gnn.eval()
    actor.eval()
    return gnn, actor


def sample_action(actor, gnn, topo, spec, spec_norm, encoder, action_space, device,
                  topo_graphs=None, bounds="electrical", switch_model="ideal"):
    with torch.no_grad():
        if encoder == "circuit":
            if action_space == "device":
                z, h = gnn(
                    topo, spec, return_device=True,
                    bounds=bounds, switch_model=switch_model,
                )
                sized = sized_devices(topo)
                dnames = device_names(topo)
                name_to_idx = {n: i for i, n in enumerate(dnames)}
                h_rows = [h[name_to_idx[d]] for d, _ in sized]
                h_act = torch.stack(h_rows, dim=0)
                a = actor.sample(spec_norm, h_act).cpu().numpy()
            else:
                z = gnn(
                    topo, spec, return_device=False,
                    bounds=bounds, switch_model=switch_model,
                )
                a = actor.sample(spec_norm, z).squeeze(0).cpu().numpy()
        else:
            g = topo_graphs[topo]
            if action_space == "device":
                z, h = gnn(g.x.to(device), g.edge_index.to(device), return_nodes=True)
                sized = sized_devices(topo)
                h_rows = gin_device_rows(h, topo, sized)
                h_act = torch.stack(h_rows, dim=0)
                a = actor.sample(spec_norm, h_act).cpu().numpy()
            else:
                z = gnn(g.x.to(device), g.edge_index.to(device))
                a = actor.sample(spec_norm, z).squeeze(0).cpu().numpy()
    return a


_DEFAULT_EVAL_SPECSET = os.path.join(REPO_ROOT, "specset", "specset_eval.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--held-out", required=True)
    ap.add_argument("--encoder", default="gin")
    ap.add_argument("--action-space", default="slot")
    ap.add_argument("--fc-mode", default="spec")
    ap.add_argument("--bounds", default="electrical")
    ap.add_argument(
        "--switch-model", required=True, choices=["ideal", "realistic"],
        help="Required: ideal or realistic (no default)",
    )
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--force-fc", type=float, default=None,
        help="Override every sampled spec's fc_ghz. Separates a band-mismatch "
             "explanation of a holdout failure from a topology-intrinsic one: "
             "re-run the same holdout in-band and see whether it recovers.",
    )
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument(
        "--eval-specset",
        default=_DEFAULT_EVAL_SPECSET if os.path.exists(_DEFAULT_EVAL_SPECSET) else None,
        help="Held-out eval pool JSON (default: specset/specset_eval.json if present)",
    )
    args = ap.parse_args()

    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    gnn, actor = load_models(args.run, args.encoder, args.action_space, device)

    eval_path = args.eval_specset
    if eval_path and os.path.exists(eval_path):
        env = PhaseShifterEnv(
            restrict_to=[args.held_out],
            eval_specset_path=eval_path,
            pool="eval",
        )
    else:
        # Old C3 setting: sample from the train / "seen specs" pool.
        print(
            "[Warning] --eval-specset missing or not found "
            f"({eval_path!r}); falling back to train-pool sampling "
            "(legacy C3 'seen specs' setting)."
        )
        env = PhaseShifterEnv(restrict_to=[args.held_out])

    topo_graphs = None
    if args.encoder == "gin":
        topo_graphs = {args.held_out: get_topology_graph(args.held_out)}

    rewards, prior_pass, compliant = [], 0, 0
    for i in range(args.n):
        obs, _ = env.reset()
        spec = env.current_spec
        if args.force_fc is not None:
            spec = dict(spec)
            spec["fc_ghz"] = float(args.force_fc)
            obs = normalize_spec(spec)
        spec_norm = torch.tensor(obs, dtype=torch.float, device=device).unsqueeze(0)
        a = sample_action(
            actor, gnn, args.held_out, spec, spec_norm,
            args.encoder, args.action_space, device, topo_graphs,
            bounds=args.bounds, switch_model=args.switch_model,
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

        if not check_physics_priors(args.held_out, params, spec["fc_ghz"]):
            rewards.append(-5.0)
            continue
        prior_pass += 1
        nl = make_spice_netlist(
            args.held_out, params, spec_dict=spec, fc_mode=args.fc_mode,
        )
        eb = env.compute_expert_bonus(args.held_out, spec)
        r, _, _ = parallel_eval_worker(
            (nl, spec, args.held_out, eb, [args.held_out], 0.0, params)
        )
        rewards.append(float(r))
        if r > COMPLIANCE_THRESHOLD:
            compliant += 1

    summary = {
        "held_out": args.held_out,
        "n": args.n,
        "compliance": compliant / args.n,
        "prior_pass_rate": prior_pass / args.n,
        "best_reward": float(max(rewards)) if rewards else -5.0,
        "mean_reward_success": float(np.mean([r for r in rewards if r > -4]))
            if any(r > -4 for r in rewards) else -5.0,
        "encoder": args.encoder,
        "action_space": args.action_space,
        "fc_mode": args.fc_mode,
        "bounds": args.bounds,
        "switch_model": args.switch_model,
        "force_fc_ghz": args.force_fc,
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
