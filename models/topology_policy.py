"""Hierarchical joint policy factorization (prototype — not trained end-to-end).

    π_joint(τ, a | s) = P_ψ(τ | s) · p_θ(a | s, τ)

P_ψ is a categorical topology head; p_θ is the existing CFM/DDPM sizing actor
conditioned on (s, τ). This module only implements P_ψ, advantage helpers, and
a documented stub update that shows how the two factors would be trained.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from specset.schema import SPEC_DIM


class TopologyCategoricalHead(nn.Module):
    """Categorical topology prior P_ψ(τ | s).

    Shared MLP scores each (spec, z_τ) pair; softmax over the six topology
    scores yields P_ψ(τ | s).

    Args:
        spec_dim: Spec vector size (default SPEC_DIM).
        graph_dim: Topology embedding size (default 64).
        n_topo: Number of topologies (default 6).
    """

    def __init__(self, spec_dim: int = SPEC_DIM, graph_dim: int = 64, n_topo: int = 6):
        super().__init__()
        self.spec_dim = spec_dim
        self.graph_dim = graph_dim
        self.n_topo = n_topo
        self.score_net = nn.Sequential(
            nn.Linear(spec_dim + graph_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def score(self, spec: torch.Tensor, z_topo: torch.Tensor) -> torch.Tensor:
        """Per-(spec, z) logit(s) before softmax.

        Args:
            spec:   [B, spec_dim]
            z_topo: [B, graph_dim] or [B, n_topo, graph_dim]

        Returns:
            [B] if z is 2-D, or [B, n_topo] if z is 3-D.
        """
        if z_topo.dim() == 2:
            x = torch.cat([spec, z_topo], dim=-1)
            return self.score_net(x).squeeze(-1)

        if z_topo.dim() != 3:
            raise ValueError(
                f"z_topo must be [B, graph_dim] or [B, n_topo, graph_dim], got {tuple(z_topo.shape)}"
            )
        if z_topo.size(1) != self.n_topo:
            raise ValueError(
                f"expected n_topo={self.n_topo} stacked embeddings, got {z_topo.size(1)}"
            )

        b, n, d = z_topo.shape
        spec_exp = spec.unsqueeze(1).expand(-1, n, -1).reshape(b * n, -1)
        z_flat = z_topo.reshape(b * n, d)
        logits = self.score_net(torch.cat([spec_exp, z_flat], dim=-1)).view(b, n)
        return logits

    def forward(self, spec: torch.Tensor, z_topo: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        - z_topo [B, graph_dim]           → raw logit [B]
        - z_topo [B, n_topo, graph_dim]   → softmax probs [B, n_topo] (sum to 1)
        """
        logits = self.score(spec, z_topo)
        if logits.dim() == 1:
            return logits
        return F.softmax(logits, dim=-1)

    def log_prob(
        self,
        spec: torch.Tensor,
        z_all: torch.Tensor,
        tau_idx: torch.Tensor,
    ) -> torch.Tensor:
        """log P_ψ(τ | s) for chosen topology indices.

        Args:
            spec:    [B, spec_dim]
            z_all:   [B, n_topo, graph_dim]
            tau_idx: [B] long indices in [0, n_topo)

        Returns:
            [B] log-probabilities.
        """
        logits = self.score(spec, z_all)
        log_probs = F.log_softmax(logits, dim=-1)
        return log_probs.gather(1, tau_idx.view(-1, 1)).squeeze(-1)


def topology_advantage(V_phi: torch.Tensor, V_omega: torch.Tensor) -> torch.Tensor:
    """Topology-level advantage A_topo = V_φ(s, τ) − V_ω(s).

    V_φ is the topology-conditioned value; V_ω is the state (marginal) baseline.
    """
    return V_phi - V_omega


def sizing_advantage(Q: torch.Tensor, V_phi: torch.Tensor) -> torch.Tensor:
    """Sizing-level advantage A_size = Q(s, τ, a) − V_φ(s, τ)."""
    return Q - V_phi


def hierarchical_joint_train_step(
    topo_head: TopologyCategoricalHead,
    actor,
    specs: torch.Tensor,
    z_all: torch.Tensor,
    tau_idx: torch.Tensor,
    z_chosen: torch.Tensor,
    actions: torch.Tensor,
    Q: torch.Tensor,
    V_phi: torch.Tensor,
    V_omega: torch.Tensor,
    tau_temp: float = 0.5,
    weight_cap: float = 10.0,
):
    """Stub training step for hierarchical joint factorization (no SPICE).

    Factorization
    -------------
        π_joint(τ, a | s) = P_ψ(τ | s) · p_θ(a | s, τ)

    Topology update (REINFORCE with A_topo)
    --------------------------------------
        A_topo = V_φ(s, τ) − V_ω(s)
        L_ψ    = − E[ A_topo · log P_ψ(τ | s) ]

    Sizing update (advantage-weighted CFM with A_size)
    --------------------------------------------------
        A_size = Q(s, τ, a) − V_φ(s, τ)
        w      = clip(exp(A_size / τ_temp), max=weight_cap)
        CFM: x_t = (1−t) x_0 + t a,  u* = a − x_0
        L_θ  = E[ w · ||u_θ(x_t, t | s, τ) − u*||² ]

    This function computes the two losses on caller-provided tensors only.
    It does not collect rollouts, call SPICE, or run an optimizer loop.

    Returns:
        dict with keys ``topo_loss``, ``cfm_loss``, ``A_topo``, ``A_size``, ``weights``.
    """
    A_topo = topology_advantage(V_phi, V_omega)
    log_p = topo_head.log_prob(specs, z_all, tau_idx)
    # Detach advantage so the categorical head does not backprop into critics.
    topo_loss = -(A_topo.detach() * log_p).mean()

    A_size = sizing_advantage(Q, V_phi)
    weights = torch.clamp(torch.exp(A_size.detach() / tau_temp), max=weight_cap)

    x_0 = torch.randn_like(actions)
    t = torch.rand(actions.size(0), device=actions.device)
    x_t = (1.0 - t.unsqueeze(-1)) * x_0 + t.unsqueeze(-1) * actions
    u_target = actions - x_0
    u_pred = actor(x_t, t, specs, z_chosen)
    cfm_loss = (weights.unsqueeze(-1) * (u_target - u_pred) ** 2).mean()

    return {
        "topo_loss": topo_loss,
        "cfm_loss": cfm_loss,
        "A_topo": A_topo,
        "A_size": A_size,
        "weights": weights,
    }
