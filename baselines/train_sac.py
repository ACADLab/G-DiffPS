"""
Baseline: Soft Actor-Critic (SAC) with Gaussian Policy + GNN Encoder.
Same GNN topology encoder as G-DiffPS. Actor replaced by unimodal Gaussian
policy (mean + log-std MLP) instead of the CFM/DDPM diffusion actor.
Critic and Value net architectures identical to G-DiffPS.
Same log-space parameter scaling, same physics prior, same reward function.

Purpose: Ablation showing G-DiffPS (diffusion) vs. unimodal Gaussian RL on
multimodal RF circuit parameter landscapes.

Usage:
    python3 baselines/train_sac.py --total-timesteps 10000 --seed 42
    python3 baselines/train_sac.py --total-timesteps 10000 --seed 42 --sizing linear

Output: results/baselines/sac/run_<timestamp>/train.log + checkpoints/
"""

import argparse
import datetime
import json
import os
import random
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from env.phaseshifter_env import PhaseShifterEnv
from env.graph_utils import get_topology_graph, TOPOLOGY_PARAMS
from models.gnn_encoder import TopologyEncoder
from models.diffusion_policy import CriticNet, ValueNet
from sim.physics_priors import check_physics_priors
from train_diffusion import action_to_params, make_spice_netlist, parallel_eval_worker
import specset.phaseshifter_scoring as scoring


TOPOLOGY_NAMES = [
    "Loaded_Line", "Switched_Line", "Reflection_Type",
    "Switched_Filter", "Vector_Modulator", "All_Pass"
]

# ── Gaussian Policy Actor ─────────────────────────────────────────────────────

class GaussianPolicyNet(nn.Module):
    """Unimodal Gaussian actor: outputs mean and log-std for each action dim."""
    def __init__(self, spec_dim: int = 12, graph_dim: int = 64, action_dim: int = 9):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(spec_dim + graph_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
        )
        self.mean_head = nn.Linear(256, action_dim)
        self.log_std_head = nn.Linear(256, action_dim)

    def forward(self, spec: torch.Tensor, z_topo: torch.Tensor):
        h = self.net(torch.cat([spec, z_topo], dim=-1))
        mean = torch.sigmoid(self.mean_head(h))          # map to [0,1]
        log_std = self.log_std_head(h).clamp(-4.0, 0.5)  # reasonable std range
        return mean, log_std.exp()

    def sample(self, spec: torch.Tensor, z_topo: torch.Tensor):
        mean, std = self.forward(spec, z_topo)
        eps = torch.randn_like(mean)
        action = (mean + std * eps).clamp(0.0, 1.0)
        return action


# ── Replay Buffer ─────────────────────────────────────────────────────────────

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buf = []
        self.pos = 0
        self.cap = capacity

    def push(self, spec, topo, action, reward):
        if len(self.buf) < self.cap:
            self.buf.append(None)
        self.buf[self.pos] = (spec, topo, action, reward)
        self.pos = (self.pos + 1) % self.cap

    def sample(self, b):
        batch = random.sample(self.buf, b)
        specs, topos, actions, rewards = zip(*batch)
        return (
            torch.tensor(np.array(specs), dtype=torch.float),
            list(topos),
            torch.tensor(np.array(actions), dtype=torch.float),
            torch.tensor(np.array(rewards), dtype=torch.float),
        )

    def __len__(self):
        return len(self.buf)


# ── Training loop ─────────────────────────────────────────────────────────────

def train(args):
    stamp = datetime.datetime.now().strftime("%H%M%S")
    base = args.out_dir if args.out_dir else os.path.join("results", "baselines", "sac")
    run_dir = os.path.join(base, f"run_{stamp}_seed{args.seed}")
    os.makedirs(run_dir, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[SAC] device={device}  seed={args.seed}  steps={args.total_timesteps}  sizing={args.sizing}")

    gnn = TopologyEncoder(in_channels=5, hidden_channels=64, out_channels=64).to(device)
    actor = GaussianPolicyNet(spec_dim=12, graph_dim=64, action_dim=9).to(device)
    critic = CriticNet(spec_dim=12, graph_dim=64, action_dim=9).to(device)
    value_net = ValueNet(spec_dim=12, graph_dim=64).to(device)

    actor_opt = optim.Adam(list(actor.parameters()) + list(gnn.parameters()), lr=args.lr)
    critic_opt = optim.Adam(critic.parameters(), lr=args.lr)
    value_opt = optim.Adam(value_net.parameters(), lr=args.lr)

    buf = ReplayBuffer(args.buffer_size)
    env = PhaseShifterEnv()

    topos = TOPOLOGY_NAMES if args.restrict_to is None else args.restrict_to
    # dataset is a list of dicts with a 'spec' key
    specset = [entry["spec"] for entry in env.dataset]

    # Pre-compute topology graphs
    topo_graphs = {}
    for t in TOPOLOGY_NAMES:
        g = get_topology_graph(t)
        topo_graphs[t] = (g.x.to(device), g.edge_index.to(device))

    log_path = os.path.join(run_dir, "train.log")
    log_f = open(log_path, "w")

    t0 = time.time()
    for step in range(1, args.total_timesteps + 1):
        # Sample spec + topology
        spec_dict = random.choice(specset)
        topo = random.choice(topos)

        # GNN forward (no grad needed for rollout)
        with torch.no_grad():
            x, ei = topo_graphs[topo]
            z = gnn(x, ei)  # [1, 64] — global_mean_pool already adds batch dim
            spec_t = torch.tensor(env._normalize(spec_dict), dtype=torch.float, device=device).unsqueeze(0)
            action_t = actor.sample(spec_t, z)  # [1, 9]

        action_np = action_t.squeeze(0).cpu().numpy()
        params = action_to_params(action_np, topo, spec_dict, sizing=args.sizing)

        expert_bonus = env.compute_expert_bonus(topo, spec_dict)
        passed_prior = check_physics_priors(topo, params, spec_dict.get("fc_ghz", 28.0))

        if not passed_prior:
            reward = -5.0 + expert_bonus
        else:
            netlist_path = make_spice_netlist(topo, params)
            reward, _, _ = parallel_eval_worker(
                (netlist_path, spec_dict, topo, expert_bonus, None)
            )

        spec_norm = env._normalize(spec_dict)
        buf.push(spec_norm, topo, action_np, reward)

        log_entry = {
            "step": step, "topology": topo, "reward": float(reward),
            "passed_prior": bool(passed_prior), "elapsed": time.time() - t0
        }
        log_f.write(json.dumps(log_entry) + "\n")

        if step % 100 == 0:
            elapsed = time.time() - t0
            print(f"[SAC] step={step:5d}  reward={reward:.3f}  "
                  f"buf={len(buf)}  {elapsed:.1f}s")
            log_f.flush()

        # Update networks
        if len(buf) < args.batch_size:
            continue

        specs_b, topos_b, actions_b, rewards_b = buf.sample(args.batch_size)
        specs_b = specs_b.to(device)
        actions_b = actions_b.to(device)
        rewards_b = rewards_b.to(device)

        # GNN forward for batch
        z_batch = []
        for t in topos_b:
            x, ei = topo_graphs[t]
            z_batch.append(gnn(x, ei).squeeze(0))  # [64]
        z_b = torch.stack(z_batch, dim=0)  # [B, 64]
        z_b_det = z_b.detach()  # detached copy for critic/value (only actor+gnn need grad)

        # Critic update
        q_pred = critic(specs_b, z_b_det, actions_b).squeeze(-1)
        critic_loss = nn.functional.mse_loss(q_pred, rewards_b)
        critic_opt.zero_grad()
        critic_loss.backward()
        critic_opt.step()

        # Value update (expectile τ=0.7)
        with torch.no_grad():
            q_vals = critic(specs_b, z_b_det, actions_b).squeeze(-1)
        v_pred = value_net(specs_b, z_b_det).squeeze(-1)
        diff = q_vals - v_pred
        tau = 0.7
        value_loss = (torch.where(diff >= 0, tau, 1.0 - tau) * diff.pow(2)).mean()
        value_opt.zero_grad()
        value_loss.backward()
        value_opt.step()

        # Actor update (advantage-weighted behavioural cloning)
        with torch.no_grad():
            adv = critic(specs_b, z_b_det, actions_b).squeeze(-1) - value_net(specs_b, z_b_det).squeeze(-1)
            weights = torch.clamp(torch.exp(adv / args.tau), max=10.0)

        mean, std = actor(specs_b, z_b)  # z_b (not detached) — backprops into gnn via actor_opt
        # Negative log-likelihood of stored actions under current Gaussian
        log_prob = -0.5 * ((actions_b - mean) / std).pow(2) - std.log()
        actor_loss = -(weights * log_prob.sum(-1)).mean()

        actor_opt.zero_grad()
        actor_loss.backward()
        actor_opt.step()

    log_f.close()
    torch.save({
        "gnn": gnn.state_dict(),
        "actor": actor.state_dict(),
        "critic": critic.state_dict(),
        "value_net": value_net.state_dict(),
    }, os.path.join(run_dir, "checkpoint.pt"))
    print(f"[SAC] Done. Results in {run_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--total-timesteps", type=int, default=10000)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--buffer-size", type=int, default=10000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--tau", type=float, default=0.5, help="Advantage temperature")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--sizing", default="log", choices=["log", "linear"])
    ap.add_argument("--restrict-to", nargs="+", default=None, choices=TOPOLOGY_NAMES)
    ap.add_argument("--out-dir", default=None, help="Override output base dir")
    args = ap.parse_args()
    train(args)


if __name__ == "__main__":
    main()
