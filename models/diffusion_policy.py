import torch
import torch.nn as nn
import torch.nn.functional as F

from env.graph_utils import SLOT_ACTION_DIM
from specset.schema import SPEC_DIM


class DenoisingScoreNet(nn.Module):
    def __init__(self, action_dim=SLOT_ACTION_DIM, spec_dim=SPEC_DIM, graph_dim=64):
        """
        Denoising Score Matching network (U-Net style or MLP style).
        
        Args:
            action_dim: Dimension of continuous parameter actions (SLOT_ACTION_DIM).
            spec_dim: Dimension of specification input vectors (SPEC_DIM).
            graph_dim: Dimension of the topology embedding vector (default 64).
        """
        super().__init__()
        # Concatenate spec + graph_dim + time steps (1)
        self.cond_layer = nn.Sequential(
            nn.Linear(spec_dim + graph_dim + 1, 128),
            nn.ReLU(),
            nn.Linear(128, 128)
        )
        # Action + Condition Embedding (128)
        self.net = nn.Sequential(
            nn.Linear(action_dim + 128, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim)
        )
        
    def forward(self, a_t, t, spec, z_topo):
        """
        Predict the added noise at step t.
        """
        cond = torch.cat([spec, z_topo, t.unsqueeze(-1)], dim=-1)
        cond_emb = self.cond_layer(cond)
        x = torch.cat([a_t, cond_emb], dim=-1)
        return self.net(x)


class DiffusionPolicy(nn.Module):
    def __init__(self, action_dim=SLOT_ACTION_DIM, spec_dim=SPEC_DIM, graph_dim=64, num_timesteps=10):
        """
        Conditional Denoising Diffusion Policy (DDPM) Actor.
        """
        super().__init__()
        self.action_dim = action_dim
        self.num_timesteps = num_timesteps
        self.model = DenoisingScoreNet(action_dim, spec_dim, graph_dim)
        
        # Noise schedule parameters
        beta = torch.linspace(1e-4, 0.02, num_timesteps)
        alpha = 1.0 - beta
        alpha_bar = torch.cumprod(alpha, dim=0)
        
        self.register_buffer("beta", beta)
        self.register_buffer("alpha", alpha)
        self.register_buffer("alpha_bar", alpha_bar)
        
    def forward(self, a_t, t, spec, z_topo):
        return self.model(a_t, t, spec, z_topo)
        
    def add_noise(self, a_0, t, noise=None):
        """
        Forward process: corrupt clean action a_0 with noise at time step t.
        """
        if noise is None:
            noise = torch.randn_like(a_0)
        
        alpha_bar = self.alpha_bar[t].unsqueeze(-1)
        
        a_t = torch.sqrt(alpha_bar) * a_0 + torch.sqrt(1.0 - alpha_bar) * noise
        return a_t
        
    def sample(self, spec, z_topo):
        """
        Reverse process: sample clean actions a_0 from pure Gaussian noise.
        """
        device = spec.device
        batch_size = spec.shape[0]
        a = torch.randn((batch_size, self.action_dim), device=device)
        
        for t in reversed(range(self.num_timesteps)):
            t_tensor = torch.full((batch_size,), t, device=device, dtype=torch.float)
            eps_pred = self.model(a, t_tensor, spec, z_topo)
            
            alpha = self.alpha[t]
            alpha_bar = self.alpha_bar[t]
            beta = self.beta[t]
            
            if t > 0:
                noise = torch.randn_like(a)
                # DDPM reverse sampling formula
                a = (1.0 / torch.sqrt(alpha)) * (a - (beta / torch.sqrt(1.0 - alpha_bar)) * eps_pred) + torch.sqrt(beta) * noise
            else:
                a = (1.0 / torch.sqrt(alpha)) * (a - (beta / torch.sqrt(1.0 - alpha_bar)) * eps_pred)
                
        # Actions are trained in [0, 1]; clamp keeps reverse sampling consistent.
        return torch.clamp(a, 0.0, 1.0)


class CriticNet(nn.Module):
    def __init__(self, action_dim=SLOT_ACTION_DIM, spec_dim=SPEC_DIM, graph_dim=64):
        """
        Q-value network Q_phi(s, z_topo, a_0) -> R.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(spec_dim + graph_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        
    def forward(self, spec, z_topo, action):
        x = torch.cat([spec, z_topo, action], dim=-1)
        return self.net(x).squeeze(-1)


class ValueNet(nn.Module):
    def __init__(self, spec_dim=SPEC_DIM, graph_dim=64):
        """
        Baseline value network V_psi(s, z_topo) -> R.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(spec_dim + graph_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        
    def forward(self, spec, z_topo):
        x = torch.cat([spec, z_topo], dim=-1)
        return self.net(x).squeeze(-1)


class VectorFieldNet(nn.Module):
    """Velocity field network for Conditional Flow Matching. Same architecture as DenoisingScoreNet."""
    def __init__(self, action_dim=SLOT_ACTION_DIM, spec_dim=SPEC_DIM, graph_dim=64):
        super().__init__()
        self.cond_layer = nn.Sequential(
            nn.Linear(spec_dim + graph_dim + 1, 128),
            nn.ReLU(),
            nn.Linear(128, 128)
        )
        self.net = nn.Sequential(
            nn.Linear(action_dim + 128, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim)
        )

    def forward(self, x_t, t, spec, z_topo):
        """Predict velocity field u(x_t, t | spec, z_topo)."""
        cond = torch.cat([spec, z_topo, t.unsqueeze(-1)], dim=-1)
        cond_emb = self.cond_layer(cond)
        x = torch.cat([x_t, cond_emb], dim=-1)
        return self.net(x)


class FlowMatchingPolicy(nn.Module):
    """Conditional Flow Matching actor. Straight-line ODE from N(0,I) to action manifold.

    Training interpolates toward actions in [0, 1]. Sampling therefore returns a
    clamped unit-interval vector — not sigmoid(x). A trailing sigmoid would map
    a learned endpoint a=0.1 to ~0.525 and break train/sample consistency.
    """
    def __init__(self, action_dim=SLOT_ACTION_DIM, spec_dim=SPEC_DIM, graph_dim=64, num_steps=10):
        super().__init__()
        self.action_dim = action_dim
        self.num_steps = num_steps
        self.model = VectorFieldNet(action_dim, spec_dim, graph_dim)

    def forward(self, x_t, t, spec, z_topo):
        return self.model(x_t, t, spec, z_topo)

    def sample(self, spec, z_topo):
        """Euler ODE integration from t=0 (noise) to t=1 (action in [0, 1])."""
        device = spec.device
        batch_size = spec.shape[0]
        x = torch.randn((batch_size, self.action_dim), device=device)
        dt = 1.0 / self.num_steps
        for i in range(self.num_steps):
            t_val = i / self.num_steps
            t_tensor = torch.full((batch_size,), t_val, device=device, dtype=torch.float)
            v = self.model(x, t_tensor, spec, z_topo)
            x = x + dt * v
        return torch.clamp(x, 0.0, 1.0)


class NodeVectorFieldNet(nn.Module):
    """Per-device CFM velocity field: shared weights over sized devices.

    Inputs are 1-D actions per device, conditioned on (spec, h_d, t).
    """

    def __init__(self, spec_dim=SPEC_DIM, graph_dim=64):
        super().__init__()
        self.cond_layer = nn.Sequential(
            nn.Linear(spec_dim + graph_dim + 1, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
        )
        self.net = nn.Sequential(
            nn.Linear(1 + 128, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, x_t, t, spec, h_dev):
        """
        Args:
            x_t:   [B, N] or [N] current actions per device
            t:     [B] or scalar
            spec:  [B, spec_dim] or [spec_dim]
            h_dev: [B, N, graph_dim] or [N, graph_dim]
        Returns:
            velocity of same shape as x_t
        """
        squeeze = False
        if x_t.dim() == 1:
            x_t = x_t.unsqueeze(0)
            h_dev = h_dev.unsqueeze(0)
            spec = spec.unsqueeze(0) if spec.dim() == 1 else spec
            t = t.unsqueeze(0) if t.dim() == 0 else t
            squeeze = True
        B, N = x_t.shape
        # Broadcast spec/t across devices
        spec_exp = spec.unsqueeze(1).expand(B, N, -1)
        t_exp = t.view(B, 1, 1).expand(B, N, 1)
        cond = torch.cat([spec_exp, h_dev, t_exp], dim=-1)
        cond_emb = self.cond_layer(cond)
        x = torch.cat([x_t.unsqueeze(-1), cond_emb], dim=-1)
        v = self.net(x).squeeze(-1)
        return v.squeeze(0) if squeeze else v


class NodeFlowMatchingPolicy(nn.Module):
    """Permutation-equivariant CFM actor over sized device nodes."""

    def __init__(self, spec_dim=SPEC_DIM, graph_dim=64, num_steps=10, max_devices=16):
        super().__init__()
        self.num_steps = num_steps
        self.max_devices = max_devices
        self.model = NodeVectorFieldNet(spec_dim, graph_dim)

    def forward(self, x_t, t, spec, h_dev):
        return self.model(x_t, t, spec, h_dev)

    def sample(self, spec, h_dev, mask=None):
        """
        Args:
            spec:  [B, spec_dim] or [spec_dim]
            h_dev: [B, N, graph_dim] or [N, graph_dim]
            mask:  [B, N] or [N] bool — True for sized devices
        Returns:
            actions in [0,1], same shape as leading dims of h_dev[...,0]
        """
        squeeze = h_dev.dim() == 2
        if squeeze:
            h_dev = h_dev.unsqueeze(0)
            spec = spec.unsqueeze(0) if spec.dim() == 1 else spec
            if mask is not None and mask.dim() == 1:
                mask = mask.unsqueeze(0)
        device = spec.device
        B, N, _ = h_dev.shape
        x = torch.randn((B, N), device=device)
        dt = 1.0 / self.num_steps
        for i in range(self.num_steps):
            t_val = i / self.num_steps
            t_tensor = torch.full((B,), t_val, device=device, dtype=torch.float)
            v = self.model(x, t_tensor, spec, h_dev)
            x = x + dt * v
        a = torch.clamp(x, 0.0, 1.0)
        if mask is not None:
            a = a * mask.float()
        return a.squeeze(0) if squeeze else a


class NodeCriticNet(nn.Module):
    """Q(s, {a_d}, {h_d}) via masked mean-pool of per-device features."""

    def __init__(self, spec_dim=SPEC_DIM, graph_dim=64):
        super().__init__()
        self.dev_mlp = nn.Sequential(
            nn.Linear(graph_dim + 1, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
        )
        self.net = nn.Sequential(
            nn.Linear(spec_dim + 128, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, spec, h_dev, action, mask=None):
        """
        Args:
            spec:   [B, spec_dim]
            h_dev:  [B, N, graph_dim]
            action: [B, N]
            mask:   [B, N] bool
        """
        x = torch.cat([h_dev, action.unsqueeze(-1)], dim=-1)
        h = self.dev_mlp(x)
        if mask is not None:
            m = mask.float().unsqueeze(-1)
            h = (h * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)
        else:
            h = h.mean(dim=1)
        return self.net(torch.cat([spec, h], dim=-1)).squeeze(-1)
