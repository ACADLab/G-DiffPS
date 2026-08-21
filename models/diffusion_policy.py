import torch
import torch.nn as nn
import torch.nn.functional as F

class DenoisingScoreNet(nn.Module):
    def __init__(self, action_dim=9, spec_dim=12, graph_dim=64):
        """
        Denoising Score Matching network (U-Net style or MLP style).
        
        Args:
            action_dim: Dimension of continuous parameter actions (default 9).
            spec_dim: Dimension of specification input vectors (default 12 for the 12 spec values in env).
            graph_dim: Dimension of the topology embedding vector (default 64).
        """
        super().__init__()
        # Concatenate spec (12) + graph_dim (64) + time steps (1) -> 77
        self.cond_layer = nn.Sequential(
            nn.Linear(spec_dim + graph_dim + 1, 128),
            nn.ReLU(),
            nn.Linear(128, 128)
        )
        # Action (9) + Condition Embedding (128) -> 137
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
    def __init__(self, action_dim=9, spec_dim=12, graph_dim=64, num_timesteps=10):
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
                
        # Sigmoid restricts output parameters to [0, 1] bounds which we map to physical ranges
        return torch.sigmoid(a)


class CriticNet(nn.Module):
    def __init__(self, action_dim=9, spec_dim=12, graph_dim=64):
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
    def __init__(self, spec_dim=12, graph_dim=64):
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
    def __init__(self, action_dim=9, spec_dim=12, graph_dim=64):
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
    """Conditional Flow Matching actor. Straight-line ODE from N(0,I) to action manifold."""
    def __init__(self, action_dim=9, spec_dim=12, graph_dim=64, num_steps=50):
        super().__init__()
        self.action_dim = action_dim
        self.num_steps = num_steps
        self.model = VectorFieldNet(action_dim, spec_dim, graph_dim)

    def forward(self, x_t, t, spec, z_topo):
        return self.model(x_t, t, spec, z_topo)

    def sample(self, spec, z_topo):
        """Euler ODE integration from t=0 (noise) to t=1 (action)."""
        device = spec.device
        batch_size = spec.shape[0]
        x = torch.randn((batch_size, self.action_dim), device=device)
        dt = 1.0 / self.num_steps
        for i in range(self.num_steps):
            t_val = i / self.num_steps
            t_tensor = torch.full((batch_size,), t_val, device=device, dtype=torch.float)
            v = self.model(x, t_tensor, spec, z_topo)
            x = x + dt * v
        return torch.sigmoid(x)
