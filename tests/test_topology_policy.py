"""Unit tests for hierarchical topology policy factorization (no SPICE)."""
from __future__ import annotations

import os
import sys

import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from env.graph_utils import SLOT_ACTION_DIM
from models.diffusion_policy import FlowMatchingPolicy
from models.topology_policy import (
    TopologyCategoricalHead,
    hierarchical_joint_train_step,
    sizing_advantage,
    topology_advantage,
)
from specset.schema import SPEC_DIM


def test_forward_shapes_single_and_stacked():
    B, spec_dim, graph_dim, n_topo = 4, SPEC_DIM, 64, 6
    head = TopologyCategoricalHead(spec_dim=spec_dim, graph_dim=graph_dim, n_topo=n_topo)
    spec = torch.randn(B, spec_dim)
    z_one = torch.randn(B, graph_dim)
    z_all = torch.randn(B, n_topo, graph_dim)

    logits = head(spec, z_one)
    assert logits.shape == (B,), f"expected {(B,)}, got {tuple(logits.shape)}"

    probs = head(spec, z_all)
    assert probs.shape == (B, n_topo), f"expected {(B, n_topo)}, got {tuple(probs.shape)}"

    score_one = head.score(spec, z_one)
    score_all = head.score(spec, z_all)
    assert score_one.shape == (B,)
    assert score_all.shape == (B, n_topo)


def test_softmax_sums_to_one():
    B, n_topo = 8, 6
    head = TopologyCategoricalHead(n_topo=n_topo)
    spec = torch.randn(B, SPEC_DIM)
    z_all = torch.randn(B, n_topo, 64)

    probs = head(spec, z_all)
    sums = probs.sum(dim=-1)
    assert torch.allclose(sums, torch.ones(B), atol=1e-5), sums
    assert (probs >= 0).all()


def test_advantage_helpers_algebraically_correct():
    V_phi = torch.tensor([3.0, 1.5, -0.5])
    V_omega = torch.tensor([1.0, 1.0, 0.0])
    Q = torch.tensor([4.0, 0.5, 2.0])

    A_topo = topology_advantage(V_phi, V_omega)
    A_size = sizing_advantage(Q, V_phi)

    assert torch.allclose(A_topo, V_phi - V_omega)
    assert torch.allclose(A_size, Q - V_phi)
    # Hierarchical telescoping: A_topo + A_size = Q - V_omega
    assert torch.allclose(A_topo + A_size, Q - V_omega)


def test_stub_train_step_losses_are_scalars():
    """Smoke-check the stub math path; does not run SPICE or an epoch loop."""
    B, n_topo = 4, 6
    head = TopologyCategoricalHead(n_topo=n_topo)
    actor = FlowMatchingPolicy(
        action_dim=SLOT_ACTION_DIM, spec_dim=SPEC_DIM, graph_dim=64, num_steps=5
    )

    specs = torch.randn(B, SPEC_DIM)
    z_all = torch.randn(B, n_topo, 64)
    tau_idx = torch.randint(0, n_topo, (B,))
    z_chosen = z_all[torch.arange(B), tau_idx]
    actions = torch.rand(B, SLOT_ACTION_DIM)
    Q = torch.randn(B)
    V_phi = torch.randn(B)
    V_omega = torch.randn(B)

    out = hierarchical_joint_train_step(
        head, actor, specs, z_all, tau_idx, z_chosen, actions, Q, V_phi, V_omega
    )
    assert out["topo_loss"].ndim == 0
    assert out["cfm_loss"].ndim == 0
    assert out["A_topo"].shape == (B,)
    assert out["A_size"].shape == (B,)
    assert torch.allclose(out["A_topo"], V_phi - V_omega)
    assert torch.allclose(out["A_size"], Q - V_phi)


if __name__ == "__main__":
    test_forward_shapes_single_and_stacked()
    test_softmax_sums_to_one()
    test_advantage_helpers_algebraically_correct()
    test_stub_train_step_losses_are_scalars()
    print("all topology_policy tests passed")
