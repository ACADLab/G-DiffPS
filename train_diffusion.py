import os
import sys
import argparse
import datetime
import json
import random
import tempfile
import numpy as np
import torch
torch.set_num_threads(1)
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torch.multiprocessing as mp

# Ensure imports work from project root
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from env.phaseshifter_env import PhaseShifterEnv
from env.graph_utils import get_topology_graph, TOPOLOGY_PARAMS
from models.gnn_encoder import TopologyEncoder
from models.diffusion_policy import DiffusionPolicy, FlowMatchingPolicy, CriticNet, ValueNet
from sim.physics_priors import check_physics_priors
import netlist.llm_netlist_gen as llm_netlist_gen

# =============================================================================
# REPLAY BUFFER
# =============================================================================

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, spec, topo_name, action, reward):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (spec, topo_name, action, reward)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        specs, topos, actions, rewards = zip(*batch)
        return (
            torch.tensor(np.array(specs), dtype=torch.float),
            topos,
            torch.tensor(np.array(actions), dtype=torch.float),
            torch.tensor(np.array(rewards), dtype=torch.float)
        )

    def __len__(self):
        return len(self.buffer)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def action_to_params(action, topology_name, spec_dict, sizing="log"):
    """
    Scale continuous [0, 1] action vector to physical SPICE values.

    spec_dict must contain 'fc_ghz' for frequency-adaptive transmission-line bounds.
    sizing='log'    → physics-informed log scaling (default, used for training).
    sizing='linear' → flat linear scaling across same bounds (ablation baseline).
    """
    keys = TOPOLOGY_PARAMS[topology_name]
    params_dict = {}

    fc_ghz = spec_dict.get("fc_ghz", 28.0)
    # Quarter-wavelength in mm at fc using eps_eff=2.5 (matches SPICE templates):
    # vp = c/sqrt(eps_eff) = 3e8/sqrt(2.5) = 1.897e8 m/s → λ/4 = vp/(4fc)
    lam4 = 47.43 / fc_ghz

    def lin(v, lo, hi):
        return lo + v * (hi - lo)

    def log_scale(v, lo, hi):
        if sizing == "linear":
            return lin(v, lo, hi)
        lo_l = np.log10(lo)
        hi_l = np.log10(hi)
        return 10 ** (lo_l + v * (hi_l - lo_l))

    for i, key in enumerate(keys):
        val_01 = float(action[i])

        # ── Universal parameters (same scaling for ALL topologies) ──────────
        if key == "R_off":
            # 3 decades [1k, 1M] Ω — always log regardless of topology
            physical_val = log_scale(val_01, 1e3, 1e6)

        elif key == "R_on":
            physical_val = lin(val_01, 0.5, 10.0)

        # ── Loaded_Line ──────────────────────────────────────────────────────
        elif topology_name == "Loaded_Line":
            if key == "Z0_line":
                physical_val = lin(val_01, 35.0, 75.0)
            elif key == "L_quarter_mm":
                physical_val = lin(val_01, 0.4 * lam4, 2.5 * lam4)
            elif key == "C_load_pf":
                # Log: places 0.05 pF at midpoint; critical for mmWave resonance
                physical_val = log_scale(val_01, 0.005, 0.5)
            else:
                physical_val = lin(val_01, 0.1, 10.0)

        # ── Switched_Line ────────────────────────────────────────────────────
        elif topology_name == "Switched_Line":
            if key == "Z0_line":
                physical_val = lin(val_01, 35.0, 75.0)
            elif key == "L_short_mm":
                # Short path: [0.3, 0.8]×λ/4 — upper bound equals long lower bound,
                # so L_long > L_short is guaranteed by construction
                physical_val = lin(val_01, 0.3 * lam4, 0.8 * lam4)
            elif key == "L_long_mm":
                # Long path: [0.8, 2.5]×λ/4
                physical_val = lin(val_01, 0.8 * lam4, 2.5 * lam4)
            else:
                physical_val = lin(val_01, 0.1, 10.0)

        # ── Reflection_Type ──────────────────────────────────────────────────
        elif topology_name == "Reflection_Type":
            if key == "Z0_main":
                physical_val = lin(val_01, 40.0, 60.0)
            elif key == "Z0_branch":
                # Ideal = Z0_main/√2 ≈ 35 Ω for a 50-Ω system
                physical_val = lin(val_01, 25.0, 45.0)
            elif key == "L_quarter_mm":
                physical_val = lin(val_01, 0.4 * lam4, 2.5 * lam4)
            elif key == "C_base_pf":
                physical_val = log_scale(val_01, 0.01, 1.0)
            elif key == "C_tune_pf":
                physical_val = log_scale(val_01, 0.01, 1.0)
            else:
                physical_val = lin(val_01, 0.1, 10.0)

        # ── Switched_Filter ──────────────────────────────────────────────────
        elif topology_name == "Switched_Filter":
            if key in ("C_hpf_pf", "C_lpf_pf"):
                # Design value: C = 1/(2π·fc·Z0). At 28GHz/50Ω: 0.114 pF
                physical_val = log_scale(val_01, 0.005, 2.0)
            elif key in ("L_hpf_nh", "L_lpf_nh"):
                # Design value: L = Z0/(2π·fc). At 28GHz/50Ω: 0.284 nH
                physical_val = log_scale(val_01, 0.005, 2.0)
            else:
                physical_val = lin(val_01, 0.1, 10.0)

        # ── Vector_Modulator ─────────────────────────────────────────────────
        elif topology_name == "Vector_Modulator":
            if key == "Z0_line":
                physical_val = lin(val_01, 35.0, 75.0)
            elif key == "L_quarter_mm":
                physical_val = lin(val_01, 0.4 * lam4, 2.5 * lam4)
            elif key in ("G_I_scale", "G_Q_scale"):
                # Dimensionless I/Q mismatch compensation; ideal = 1.0.
                # Upper bound capped at 1.0: template IL = -20log10(scale),
                # so scale > 1.0 produces active gain which is unphysical.
                physical_val = lin(val_01, 0.7, 1.0)
            else:
                physical_val = lin(val_01, 0.1, 10.0)

        # ── All_Pass ─────────────────────────────────────────────────────────
        elif topology_name == "All_Pass":
            if key in ("L_apA_nh", "L_apB_nh"):
                physical_val = log_scale(val_01, 0.01, 2.0)
            elif key in ("C_brA_pf", "C_brB_pf"):
                physical_val = log_scale(val_01, 0.005, 0.5)
            elif key == "C_cA_pf":
                # Reparameterized: C_cA = k × C_brA, k ∈ [1.2, 4.0]
                # Bridged-T balance requires C_c ≈ 2×C_br. Encoding C_c as a
                # ratio of C_br couples them in the action space so the policy
                # learns (ratio, C_br) instead of two independent capacitors —
                # guaranteeing C_c/C_br ≥ 1.2 by construction, no prior needed.
                c_brA = log_scale(float(action[keys.index("C_brA_pf")]), 0.005, 0.5)
                k_A = lin(val_01, 1.2, 4.0)
                physical_val = k_A * c_brA
            elif key == "C_cB_pf":
                c_brB = log_scale(float(action[keys.index("C_brB_pf")]), 0.005, 0.5)
                k_B = lin(val_01, 1.2, 4.0)
                physical_val = k_B * c_brB
            else:
                physical_val = lin(val_01, 0.1, 10.0)

        # ── Fallback ─────────────────────────────────────────────────────────
        else:
            info = llm_netlist_gen._PARAM_INFO.get(key)
            if info is not None:
                lo, hi = info[0], info[1]
            else:
                lo, hi = 0.1, 10.0
            physical_val = lin(val_01, lo, hi)

        clamped_str = llm_netlist_gen.clamp_spice_value(key, f"{physical_val:.4e}")
        params_dict[key] = clamped_str

    return params_dict


def make_spice_netlist(topology_name, params_dict):
    """
    Load topology template, inject param overrides, and write temporary netlist.
    """
    skeleton_path = os.path.join(REPO_ROOT, f"specset/templates/{topology_name.lower()}.sp")
    with open(skeleton_path, "r") as f:
        skeleton_content = f.read()

    param_str = ".PARAM " + " ".join([f"{k}={v}" for k, v in params_dict.items()]) + "\n"

    lines = skeleton_content.split('\n')
    new_lines = []
    for l in lines:
        stripped = l.strip().upper()
        if stripped.startswith(".PARAM"):
            if 'DERIVED' in stripped or 'FRAMEWORK_CONTROLLED' in stripped:
                new_lines.append(l)
        else:
            new_lines.append(l)

    if new_lines:
        final_netlist = new_lines[0] + '\n' + param_str + '\n'.join(new_lines[1:])
    else:
        final_netlist = f"* {topology_name}\n" + param_str

    fd, path = tempfile.mkstemp(suffix=".sp", prefix=f"diff_{topology_name.lower()}_")
    with os.fdopen(fd, 'w') as f:
        f.write(final_netlist)
    return path


def parallel_eval_worker(args):
    """
    Multiprocessing worker to evaluate a single netlist in parallel.
    """
    netlist_path, spec_dict, topology_name, expert_bonus, env_restrict_to = args
    
    # Instantiate thread-local/process-local env to evaluate the netlist safely
    env = PhaseShifterEnv(restrict_to=env_restrict_to)
    env.current_spec = spec_dict
    
    try:
        agg, sim_reward, state_indices, bits, ideal_step_deg = env._evaluate_netlist(netlist_path)
        total_reward = sim_reward + expert_bonus
        success = (agg is not None)
    except Exception as e:
        print(f"[Worker Error] {e}")
        agg, total_reward, success = None, -5.0 + expert_bonus, False
        
    # Cleanup temp netlist files
    try:
        if os.path.exists(netlist_path):
            os.remove(netlist_path)
        lis_path = f"{netlist_path}.lis"
        if os.path.exists(lis_path):
            os.remove(lis_path)
    except Exception:
        pass
        
    return total_reward, agg, success


# =============================================================================
# DDP AND TRAINING SETUP
# =============================================================================

def train(rank, world_size, args):
    # Set seed for reproducibility
    seed = args.seed + rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    # Initialize Distributed Data Parallel
    is_ddp = world_size > 1
    if is_ddp:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = args.ddp_port
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(rank)
        device = torch.device(f"cuda:{rank}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        rank = 0

    print(f"[Rank {rank}] Running on {device} (Seed: {seed})")

    # Initialize environment and dataset specs
    env = PhaseShifterEnv(restrict_to=args.restrict_to)
    
    # Neural Networks initialization
    gnn_encoder = TopologyEncoder().to(device)
    if args.actor == "cfm":
        actor = FlowMatchingPolicy(action_dim=9, spec_dim=12, graph_dim=64).to(device)
    else:
        actor = DiffusionPolicy(action_dim=9, spec_dim=12, graph_dim=64).to(device)
    critic = CriticNet(action_dim=9, spec_dim=12, graph_dim=64).to(device)
    value_net = ValueNet(spec_dim=12, graph_dim=64).to(device)

    # Wrap models in DDP if running distributed
    if is_ddp:
        gnn_encoder = DDP(gnn_encoder, device_ids=[rank], find_unused_parameters=True)
        actor = DDP(actor, device_ids=[rank], find_unused_parameters=True)
        critic = DDP(critic, device_ids=[rank])
        value_net = DDP(value_net, device_ids=[rank])

    # Optimizers
    gnn_opt = optim.Adam(gnn_encoder.parameters(), lr=args.lr)
    actor_opt = optim.Adam(actor.parameters(), lr=args.lr)
    critic_opt = optim.Adam(critic.parameters(), lr=args.lr)
    value_opt = optim.Adam(value_net.parameters(), lr=args.lr)

    # Shared Replay Buffer (only rank 0 does logging & checkpoint saving)
    replay_buffer = ReplayBuffer(capacity=args.buffer_size)

    # Output run setup
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(REPO_ROOT, f"runs_diffusion/run_{stamp}")
    if rank == 0:
        os.makedirs(run_dir, exist_ok=True)
        log_fh = open(os.path.join(run_dir, "train.log"), "w", buffering=1)
        print(f"[Rank 0] Saving results to: {run_dir}")
        log_fh.write(json.dumps({"event": "training_start", "time": datetime.datetime.utcnow().isoformat() + "Z"}) + "\n")

    # Dry-run override
    total_timesteps = args.total_timesteps
    if args.dry_run:
        total_timesteps = 5
        print(f"[Rank {rank}] Running dry-run validation (5 steps)")

    # Prep graph PyG data for the 6 topologies on target device
    topo_graphs = {}
    for name in TOPOLOGY_PARAMS.keys():
        g = get_topology_graph(name)
        g.x = g.x.to(device)
        g.edge_index = g.edge_index.to(device)
        topo_graphs[name] = g

    # Pre-fill replay buffer with warm start random runs if desired
    obs, info = env.reset()
    
    # Main online training loop
    for step in range(total_timesteps):
        # 1. Reset Spec and select random/heuristic topologies to explore
        obs_vec, info = env.reset()
        spec_dict = env.current_spec
        
        # Sample random active topology to evaluate continuous action on
        topology_name = random.choice(env._active_topologies)
        
        # 2. Extract topology embedding vector using GNN TopologyEncoder
        graph_data = topo_graphs[topology_name]
        z_topo = gnn_encoder(graph_data.x, graph_data.edge_index)
        
        # Normalize spec vector
        spec_tensor = torch.tensor(obs_vec, dtype=torch.float, device=device).unsqueeze(0)
        
        # 3. Sample continuous parameters action using continuous DDPM Actor
        with torch.no_grad():
            action_tensor = actor.sample(spec_tensor, z_topo)
            
        action = action_tensor.squeeze(0).cpu().numpy()
        
        # 4. Map actions to physical parameters and verify against prior filters
        params_dict = action_to_params(action, topology_name, spec_dict, sizing=args.sizing)
        
        passed_prior = check_physics_priors(topology_name, params_dict, spec_dict["fc_ghz"])
        expert_bonus = env.compute_expert_bonus(topology_name, spec_dict)
        
        agg_metrics = None
        if not passed_prior:
            # immediately penalty fail reward if rejected by the differentiable analytical prior
            total_reward = -5.0 + expert_bonus
            success = False
            if rank == 0:
                print(f"[Step {step:04d}] [{topology_name}] Rejected by physics prior. RL penalty assigned.")
        else:
            # 5. Generate SPICE Netlist and call simulation
            netlist_path = make_spice_netlist(topology_name, params_dict)
            
            # Since we have a single step, we just evaluate it directly
            total_reward, agg_metrics, success = parallel_eval_worker(
                (netlist_path, spec_dict, topology_name, expert_bonus, args.restrict_to)
            )
            
        # 6. Push transition to Replay Buffer
        replay_buffer.push(obs_vec, topology_name, action, total_reward)
        
        # Log to file on rank 0
        if rank == 0:
            log_entry = {
                "step": step,
                "topology": topology_name,
                "passed_prior": passed_prior,
                "reward": float(total_reward),
                "success": success,
                "metrics": agg_metrics,
                "params": params_dict,
            }
            log_fh.write(json.dumps(log_entry) + "\n")
            print(f"[Step {step:04d}] [{topology_name}] Reward: {total_reward:+.3f} (Succeeded: {success})")
            
        # 7. Update networks via Q-value reweighted score matching
        if len(replay_buffer) >= args.batch_size:
            # Sample batch from replay buffer
            specs_b, topos_b, actions_b, rewards_b = replay_buffer.sample(args.batch_size)
            
            specs_b = specs_b.to(device)
            actions_b = actions_b.to(device)
            rewards_b = rewards_b.to(device)
            
            # Map batch of topologies to their GNN embeddings
            z_topo_list = []
            for topo_name in topos_b:
                g = topo_graphs[topo_name]
                z = gnn_encoder(g.x, g.edge_index)
                z_topo_list.append(z)
                
            z_topo_b = torch.cat(z_topo_list, dim=0) # [batch_size, 64]
            
            # --- Update Critic (Q-network) ---
            # Bandit setting: Q targets are exactly the actual immediate rewards.
            # Detach GNN embedding to isolate gradient paths.
            q_pred = critic(specs_b, z_topo_b.detach(), actions_b)
            critic_loss = F.mse_loss(q_pred, rewards_b)
            
            critic_opt.zero_grad()
            critic_loss.backward()
            critic_opt.step()
            
            # Recompute GNN embeddings for actor updates
            z_topo_list = []
            for topo_name in topos_b:
                g = topo_graphs[topo_name]
                z = gnn_encoder(g.x, g.edge_index)
                z_topo_list.append(z)
            z_topo_b = torch.cat(z_topo_list, dim=0)
            
            # --- Update Value network (expectile regression) ---
            # Detach GNN embedding to isolate gradient paths.
            with torch.no_grad():
                q_pred = critic(specs_b, z_topo_b.detach(), actions_b)
                
            v_pred = value_net(specs_b, z_topo_b.detach())
            diff = q_pred - v_pred
            # Expectile parameter tau=0.7 to estimate upper expected value bounds
            weight = torch.where(diff > 0, 0.7, 0.3)
            value_loss = (weight * (diff ** 2)).mean()
            
            value_opt.zero_grad()
            value_loss.backward()
            value_opt.step()
            
            # --- Update Actor (Diffusion Policy via Advantage-weighted Score Matching) ---
            # Detach GNN embedding for the advantage estimation calculations.
            with torch.no_grad():
                q_pred = critic(specs_b, z_topo_b.detach(), actions_b)
                v_pred = value_net(specs_b, z_topo_b.detach())
                
            advantage = q_pred - v_pred
            # Exponential reward-advantage weights, capped at 10.0 to prevent gradient explosions
            weights = torch.clamp(torch.exp(advantage / args.tau), max=10.0)
            
            # Actor update: dispatch on actor type (CFM or DDPM)
            module_actor = actor.module if is_ddp else actor
            if isinstance(module_actor, FlowMatchingPolicy):
                # CFM: straight-line interpolation, predict velocity u = x_1 - x_0
                x_0 = torch.randn_like(actions_b)
                t_cfm = torch.rand(args.batch_size, device=device)
                x_t = (1 - t_cfm.unsqueeze(-1)) * x_0 + t_cfm.unsqueeze(-1) * actions_b
                u_target = actions_b - x_0
                u_pred = actor(x_t, t_cfm, specs_b, z_topo_b)
                actor_loss = (weights.unsqueeze(-1) * (u_target - u_pred) ** 2).mean()
            else:
                # DDPM: forward-process noise corruption, predict added noise
                noise = torch.randn_like(actions_b)
                t = torch.randint(0, module_actor.num_timesteps, (args.batch_size,), device=device).float()
                a_noisy = module_actor.add_noise(actions_b, t.long(), noise)
                noise_pred = actor(a_noisy, t, specs_b, z_topo_b)
                actor_loss = (weights.unsqueeze(-1) * (noise - noise_pred) ** 2).mean()
            
            actor_opt.zero_grad()
            gnn_opt.zero_grad()
            actor_loss.backward()
            actor_opt.step()
            gnn_opt.step()
            
    # Clean up distributed processes
    if rank == 0:
        log_fh.close()
        # Save checkpoints
        torch.save(gnn_encoder.state_dict(), os.path.join(run_dir, "gnn_encoder.pt"))
        torch.save(actor.state_dict(), os.path.join(run_dir, "actor.pt"))
        torch.save(critic.state_dict(), os.path.join(run_dir, "critic.pt"))
        torch.save(value_net.state_dict(), os.path.join(run_dir, "value_net.pt"))
        print(f"[Rank 0] Checkpoints written successfully to: {run_dir}")

    if is_ddp:
        dist.destroy_process_group()


# =============================================================================
# MAIN INVOCATION ENTRY
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Perform a short verification run")
    parser.add_argument("--restrict-to", nargs="+", help="Restrict active topologies to subset", default=None)
    parser.add_argument("--total-timesteps", type=int, default=100, help="Total online training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for updates")
    parser.add_argument("--buffer-size", type=int, default=1000, help="Replay buffer max size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--tau", type=float, default=0.5, help="Temperature for advantage weight exponent")
    parser.add_argument("--ddp-port", type=str, default="29500", help="DDP communication port")
    parser.add_argument("--gpus", type=int, default=1, help="Number of GPUs for distributed DDP")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--sizing", type=str, default="log", choices=["log", "linear"], help="Scaling mode for action parameters")
    parser.add_argument("--actor", type=str, default="cfm", choices=["ddpm", "cfm"], help="Actor type: ddpm (original) or cfm (Conditional Flow Matching)")
    args = parser.parse_args()

    # If running with multiple GPUs, spawn distributed processes
    if args.gpus > 1:
        print(f"Spawning DDP training across {args.gpus} GPUs...")
        mp.spawn(train, nprocs=args.gpus, args=(args.gpus, args), join=True)
    else:
        train(0, 1, args)
