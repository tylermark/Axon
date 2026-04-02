"""Unit tests for DifferentiableSATSolver, BettiRegularization, and ConstraintSolver."""

from __future__ import annotations

import pytest
import torch

from docs.interfaces.constraint_signals import AxiomResult, ConstraintGradients
from src.constraints.sat_solver import (
    BettiRegularization,
    ConstraintSolver,
    DifferentiableSATSolver,
)
from src.pipeline.config import ConstraintConfig

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def perfect_square():
    positions = torch.tensor(
        [[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]],
        dtype=torch.float32,
    )
    adjacency = torch.zeros(1, 4, 4)
    edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
    for i, j in edges:
        adjacency[0, i, j] = 1.0
        adjacency[0, j, i] = 1.0
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
    return positions, adjacency, edge_index


@pytest.fixture()
def diamond():
    """45°-rotated square — imperfect graph for constraint testing."""
    import math

    s = math.sqrt(2) / 2
    positions = torch.tensor(
        [[[0.0, s], [s, 0.0], [0.0, -s], [-s, 0.0]]],
        dtype=torch.float32,
    )
    adjacency = torch.zeros(1, 4, 4)
    edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
    for i, j in edges:
        adjacency[0, i, j] = 1.0
        adjacency[0, j, i] = 1.0
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
    return positions, adjacency, edge_index


@pytest.fixture()
def config():
    return ConstraintConfig()


# ---------------------------------------------------------------------------
# DifferentiableSATSolver (C-005)
# ---------------------------------------------------------------------------


class TestDifferentiableSATSolver:
    def test_returns_constraint_gradients(self, config, perfect_square):
        pos, adj, ei = perfect_square
        solver = DifferentiableSATSolver(config)
        result = solver(pos, adj, ei)
        assert isinstance(result, ConstraintGradients)

    def test_total_loss_is_scalar(self, config, perfect_square):
        pos, adj, ei = perfect_square
        solver = DifferentiableSATSolver(config)
        result = solver(pos, adj, ei)
        assert result.total_loss.shape == ()

    def test_total_loss_differentiable(self, config, diamond):
        pos, adj, ei = diamond
        pos = pos.clone().requires_grad_(True)
        solver = DifferentiableSATSolver(config)
        result = solver(pos, adj, ei)
        result.total_loss.backward()
        assert pos.grad is not None

    def test_imperfect_graph_positive_loss(self, config, diamond):
        pos, adj, ei = diamond
        solver = DifferentiableSATSolver(config)
        result = solver(pos, adj, ei)
        assert result.total_loss.item() > 0.0

    def test_perfect_square_near_zero_loss(self, config, perfect_square):
        pos, adj, ei = perfect_square
        solver = DifferentiableSATSolver(config)
        result = solver(pos, adj, ei)
        # Perfect square: orthogonal = 0, junction closure is small.
        assert result.total_loss.item() < 5.0

    def test_axiom_results_count(self, config, perfect_square):
        pos, adj, ei = perfect_square
        solver = DifferentiableSATSolver(config)
        result = solver(pos, adj, ei)
        assert len(result.axiom_results) == 4
        for r in result.axiom_results:
            assert isinstance(r, AxiomResult)

    def test_weights_learnable(self):
        config = ConstraintConfig(learn_weights=True)
        solver = DifferentiableSATSolver(config)
        # AxiomRegistry is not an nn.Module, so axiom weights are
        # accessed via registry.parameters(), not solver.parameters().
        weight_params = solver.registry.parameters()
        learnable = [p for p in weight_params if p.requires_grad]
        assert len(learnable) == 4  # one weight per axiom

    def test_weights_frozen(self):
        config = ConstraintConfig(learn_weights=False)
        solver = DifferentiableSATSolver(config)
        # All axiom weight parameters should be frozen.
        for axiom in solver.registry._axioms.values():
            assert not axiom.weight.requires_grad

    def test_edge_angles_populated(self, config, perfect_square):
        pos, adj, ei = perfect_square
        solver = DifferentiableSATSolver(config)
        result = solver(pos, adj, ei)
        assert result.edge_angles is not None
        assert result.edge_angles.shape[0] == ei.shape[1]


# ---------------------------------------------------------------------------
# BettiRegularization (C-006)
# ---------------------------------------------------------------------------


class TestBettiRegularization:
    def test_connected_graph_low_loss(self, perfect_square):
        """Single connected component → Betti-0 ≈ 1, loss ≈ 0."""
        _, adj, _ = perfect_square
        betti = BettiRegularization(target_betti_0=1)
        loss = betti(adj)
        assert loss.item() < 1.0

    def test_disconnected_graph_positive_loss(self):
        """Three disconnected components → soft Betti-0 > target → loss > 0.

        With two components, sigmoid(0)=0.5 per zero eigenvalue gives
        soft count ≈ 1.0 = target, so we need ≥3 components to see loss.
        """
        adj = torch.zeros(1, 6, 6)
        # Component 1: nodes 0-1.
        adj[0, 0, 1] = adj[0, 1, 0] = 1.0
        # Component 2: nodes 2-3.
        adj[0, 2, 3] = adj[0, 3, 2] = 1.0
        # Component 3: nodes 4-5.
        adj[0, 4, 5] = adj[0, 5, 4] = 1.0
        betti = BettiRegularization(target_betti_0=1)
        loss = betti(adj)
        assert loss.item() > 0.01

    def test_differentiable(self):
        adj = torch.randn(1, 4, 4, requires_grad=True)
        betti = BettiRegularization()
        loss = betti(adj)
        loss.backward()
        assert adj.grad is not None

    def test_empty_graph(self):
        """Single node, no edges → should not crash."""
        adj = torch.zeros(1, 1, 1)
        betti = BettiRegularization()
        loss = betti(adj)
        assert loss.shape == ()

    def test_2d_input(self):
        """Unbatched (N, N) adjacency."""
        adj = torch.zeros(4, 4)
        adj[0, 1] = adj[1, 0] = 1.0
        adj[1, 2] = adj[2, 1] = 1.0
        adj[2, 3] = adj[3, 2] = 1.0
        adj[3, 0] = adj[0, 3] = 1.0
        betti = BettiRegularization(target_betti_0=1)
        loss = betti(adj)
        assert loss.item() < 1.0

    def test_node_mask(self):
        """Masked nodes should be excluded."""
        adj = torch.zeros(1, 4, 4)
        adj[0, 0, 1] = adj[0, 1, 0] = 1.0
        mask = torch.tensor([[True, True, False, False]])
        betti = BettiRegularization(target_betti_0=1)
        loss = betti(adj, node_mask=mask)
        assert loss.shape == ()


# ---------------------------------------------------------------------------
# ConstraintSolver (top-level)
# ---------------------------------------------------------------------------


class TestConstraintSolver:
    def test_training_mode_no_projection(self, config, perfect_square):
        pos, adj, ei = perfect_square
        solver = ConstraintSolver(config)
        result = solver(pos, adj, ei, is_inference=False)
        assert result.projected_positions is None
        assert result.projected_adjacency is None

    def test_inference_mode_has_projection(self, config, perfect_square):
        pos, adj, ei = perfect_square
        solver = ConstraintSolver(config)
        result = solver(pos, adj, ei, is_inference=True, denoising_step=0, total_steps=1000)
        assert result.projected_positions is not None
        assert result.projected_adjacency is not None

    def test_inference_beyond_snap_step(self, config, perfect_square):
        """Denoising step > snap_at_step → no projection."""
        pos, adj, ei = perfect_square
        solver = ConstraintSolver(config)
        # config.snap_at_step defaults to 5.
        result = solver(pos, adj, ei, is_inference=True, denoising_step=10, total_steps=1000)
        assert result.projected_positions is None

    def test_total_loss_includes_betti(self, config, perfect_square):
        pos, adj, ei = perfect_square
        solver = ConstraintSolver(config)
        result = solver(pos, adj, ei, is_inference=False)
        # Metadata should contain betti_loss.
        assert "betti_loss" in result.metadata

    def test_returns_constraint_gradients(self, config, perfect_square):
        pos, adj, ei = perfect_square
        solver = ConstraintSolver(config)
        result = solver(pos, adj, ei)
        assert isinstance(result, ConstraintGradients)
        assert len(result.axiom_results) == 4
        assert result.total_loss.shape == ()

    def test_total_loss_differentiable(self, config, diamond):
        pos, adj, ei = diamond
        pos = pos.clone().requires_grad_(True)
        solver = ConstraintSolver(config)
        result = solver(pos, adj, ei, is_inference=False)
        result.total_loss.backward()
        assert pos.grad is not None
