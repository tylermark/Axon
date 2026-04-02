"""Unit tests for src/diffusion/forward.py.

Tests ForwardDiffusion: Gaussian coordinate noise, absorbing-state
adjacency noise, joint forward pass, masking, and differentiability.

Q-005: diffusion unit tests (forward process).
"""

from __future__ import annotations

import torch

from src.diffusion.forward import ForwardDiffusion, ForwardDiffusionOutput
from src.diffusion.scheduler import DiffusionScheduler
from tests.fixtures.diffusion_helpers import create_synthetic_graph

# ---------------------------------------------------------------------------
# Shared setup
# ---------------------------------------------------------------------------

B, N = 2, 8


def _make_scheduler(num_t: int = 100) -> DiffusionScheduler:
    return DiffusionScheduler(num_timesteps=num_t)


def _make_fwd(num_t: int = 100) -> ForwardDiffusion:
    return ForwardDiffusion(_make_scheduler(num_t))


# ---------------------------------------------------------------------------
# noise_coordinates
# ---------------------------------------------------------------------------


class TestNoiseCoordinates:
    """Tests for ForwardDiffusion.noise_coordinates."""

    def test_output_shape(self):
        fwd = _make_fwd()
        x_0, _, _ = create_synthetic_graph(B, N)
        t = torch.zeros(B, dtype=torch.long)
        x_t, eps = fwd.noise_coordinates(x_0, t)
        assert x_t.shape == (B, N, 2)
        assert eps.shape == (B, N, 2)

    def test_t0_close_to_input(self):
        """At t=0, alpha_bar ≈ 1 so x_t ≈ x_0."""
        fwd = _make_fwd()
        x_0, _, _ = create_synthetic_graph(B, N)
        t = torch.zeros(B, dtype=torch.long)
        x_t, _ = fwd.noise_coordinates(x_0, t)
        torch.testing.assert_close(x_t, x_0, atol=0.1, rtol=0.1)

    def test_tmax_far_from_input(self):
        """At t=T-1, signal is mostly noise."""
        total_t = 100
        fwd = _make_fwd(total_t)
        x_0, _, _ = create_synthetic_graph(B, N)
        t = torch.full((B,), total_t - 1, dtype=torch.long)
        x_t, _ = fwd.noise_coordinates(x_0, t)
        diff = (x_t - x_0).abs().mean()
        assert diff > 0.1, f"Expected heavy noise, got mean diff {diff:.4f}"

    def test_epsilon_shape(self):
        fwd = _make_fwd()
        x_0, _, _ = create_synthetic_graph(B, N)
        t = torch.tensor([10, 50])
        _, eps = fwd.noise_coordinates(x_0, t)
        assert eps.shape == (B, N, 2)

    def test_custom_noise_used(self):
        fwd = _make_fwd()
        x_0, _, _ = create_synthetic_graph(B, N)
        t = torch.tensor([10, 50])
        custom_noise = torch.ones(B, N, 2)
        _, eps = fwd.noise_coordinates(x_0, t, noise=custom_noise)
        torch.testing.assert_close(eps, custom_noise)


# ---------------------------------------------------------------------------
# noise_adjacency
# ---------------------------------------------------------------------------


class TestNoiseAdjacency:
    """Tests for ForwardDiffusion.noise_adjacency."""

    def test_output_binary(self):
        fwd = _make_fwd()
        _, a_0, _ = create_synthetic_graph(B, N)
        t = torch.tensor([10, 50])
        a_t = fwd.noise_adjacency(a_0, t)
        unique_vals = a_t.unique()
        assert all(v in [0.0, 1.0] for v in unique_vals.tolist())

    def test_output_symmetric(self):
        fwd = _make_fwd()
        _, a_0, _ = create_synthetic_graph(B, N)
        t = torch.tensor([10, 50])
        a_t = fwd.noise_adjacency(a_0, t)
        torch.testing.assert_close(a_t, a_t.transpose(-2, -1))

    def test_zero_diagonal(self):
        fwd = _make_fwd()
        _, a_0, _ = create_synthetic_graph(B, N)
        t = torch.tensor([10, 50])
        a_t = fwd.noise_adjacency(a_0, t)
        for b in range(B):
            assert a_t[b].diag().sum() == 0

    def test_t0_close_to_input(self):
        """At t=0 most edges are kept."""
        fwd = _make_fwd()
        _, a_0, _ = create_synthetic_graph(B, N)
        t = torch.zeros(B, dtype=torch.long)
        # alpha_bar at t=0 is close to 1 so nearly all edges are kept
        a_t = fwd.noise_adjacency(a_0, t)
        agreement = (a_t == a_0).float().mean()
        assert agreement > 0.8, f"Expected high agreement at t=0, got {agreement:.2f}"

    def test_tmax_roughly_random(self):
        """At t=T-1, edges are roughly 50% populated (random)."""
        total_t = 100
        fwd = _make_fwd(total_t)
        _, a_0, _ = create_synthetic_graph(4, 16)
        t = torch.full((4,), total_t - 1, dtype=torch.long)
        a_t = fwd.noise_adjacency(a_0, t)
        # Only upper triangle (excluding diagonal)
        triu_mask = torch.triu(torch.ones(16, 16, dtype=torch.bool), diagonal=1)
        edge_frac = a_t[:, triu_mask].mean().item()
        assert 0.2 < edge_frac < 0.8, f"Expected ~0.5, got {edge_frac:.2f}"


# ---------------------------------------------------------------------------
# forward() — joint pass
# ---------------------------------------------------------------------------


class TestForwardJoint:
    """Tests for ForwardDiffusion.forward."""

    def test_returns_output_dataclass(self):
        fwd = _make_fwd()
        x_0, a_0, mask = create_synthetic_graph(B, N)
        t = torch.tensor([5, 50])
        out = fwd(x_0, a_0, t, node_mask=mask)
        assert isinstance(out, ForwardDiffusionOutput)
        assert out.x_t.shape == (B, N, 2)
        assert out.a_t.shape == (B, N, N)
        assert out.epsilon.shape == (B, N, 2)
        assert out.t.shape == (B,)
        assert out.alpha_bar.shape == (B,)

    def test_node_mask_zeros_padded(self):
        """Padded positions should be zeroed in x_t and a_t."""
        fwd = _make_fwd()
        x_0, a_0, mask = create_synthetic_graph(B, N)
        # Mask out last 2 nodes
        mask[:, -2:] = False
        t = torch.tensor([10, 50])
        out = fwd(x_0, a_0, t, node_mask=mask)
        # Padded coordinate positions should be zero
        assert (out.x_t[:, -2:, :] == 0).all()
        assert (out.epsilon[:, -2:, :] == 0).all()
        # Padded adjacency rows/cols should be zero
        assert (out.a_t[:, -2:, :] == 0).all()
        assert (out.a_t[:, :, -2:] == 0).all()

    def test_differentiability(self):
        """x_t.requires_grad should be True when x_0.requires_grad is True."""
        fwd = _make_fwd()
        x_0, a_0, mask = create_synthetic_graph(B, N)
        x_0.requires_grad_(True)
        t = torch.tensor([10, 50])
        out = fwd(x_0, a_0, t, node_mask=mask)
        assert out.x_t.requires_grad

    def test_without_node_mask(self):
        fwd = _make_fwd()
        x_0, a_0, _ = create_synthetic_graph(B, N)
        t = torch.tensor([10, 50])
        out = fwd(x_0, a_0, t, node_mask=None)
        assert out.x_t.shape == (B, N, 2)
