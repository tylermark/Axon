"""Unit tests for constraint axioms, utilities, and registry.

Tests cover:
- Utility functions: compute_edge_directions, compute_edge_angles, edges_from_adjacency
- OrthogonalIntegrityAxiom (C-001)
- ParallelPairConstancyAxiom (C-002)
- JunctionClosureAxiom (C-003)
- SpatialNonIntersectionAxiom (C-004)
- AxiomRegistry (C-008)
"""

from __future__ import annotations

import math

import pytest
import torch

from docs.interfaces.constraint_signals import AxiomResult
from src.constraints.axioms import (
    AxiomRegistry,
    JunctionClosureAxiom,
    OrthogonalIntegrityAxiom,
    ParallelPairConstancyAxiom,
    SpatialNonIntersectionAxiom,
    compute_edge_angles,
    compute_edge_directions,
    edges_from_adjacency,
)
from src.pipeline.config import ConstraintConfig

# ---------------------------------------------------------------------------
# Fixtures — synthetic graphs
# ---------------------------------------------------------------------------


@pytest.fixture()
def perfect_square():
    """4 nodes forming a perfect axis-aligned unit square.

    0--1
    |  |
    3--2

    All angles 90°. Edges: 0-1, 1-2, 2-3, 3-0.
    """
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
    """4 nodes forming a 45° rotated square (diamond).

    Edges at 45°/135° — should trigger orthogonal violations.
    """
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
def graph_with_dangling():
    """5-node graph with one dangling node (degree 1).

    0--1--2
    |     |
    3--4  (node 4 has edge only to 3)

    Node 4 is degree 1 (dangling).
    """
    positions = torch.tensor(
        [[[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [0.0, 1.0], [1.0, 1.0]]],
        dtype=torch.float32,
    )
    adjacency = torch.zeros(1, 5, 5)
    edges = [(0, 1), (1, 2), (0, 3), (3, 4)]
    for i, j in edges:
        adjacency[0, i, j] = 1.0
        adjacency[0, j, i] = 1.0
    edge_index = torch.tensor([[0, 1, 0, 3], [1, 2, 3, 4]], dtype=torch.long)
    return positions, adjacency, edge_index


@pytest.fixture()
def crossing_edges():
    """4 nodes forming an X — two edges that cross.

    Edge 0-2 and edge 1-3 cross in the middle.
    Nodes: 0=(0,0), 1=(1,0), 2=(1,1), 3=(0,1).
    Edges: 0→2 (diagonal) and 1→3 (diagonal), crossing at (0.5, 0.5).
    """
    positions = torch.tensor(
        [[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]],
        dtype=torch.float32,
    )
    adjacency = torch.zeros(1, 4, 4)
    edges = [(0, 2), (1, 3)]
    for i, j in edges:
        adjacency[0, i, j] = 1.0
        adjacency[0, j, i] = 1.0
    edge_index = torch.tensor([[0, 1], [2, 3]], dtype=torch.long)
    return positions, adjacency, edge_index


# ---------------------------------------------------------------------------
# Utility tests
# ---------------------------------------------------------------------------


class TestComputeEdgeDirections:
    def test_horizontal_edge(self):
        positions = torch.tensor([[0.0, 0.0], [1.0, 0.0]])
        edge_index = torch.tensor([[0], [1]], dtype=torch.long)
        dirs = compute_edge_directions(positions, edge_index)
        torch.testing.assert_close(dirs, torch.tensor([[1.0, 0.0]]))

    def test_vertical_edge(self):
        positions = torch.tensor([[0.0, 0.0], [0.0, 1.0]])
        edge_index = torch.tensor([[0], [1]], dtype=torch.long)
        dirs = compute_edge_directions(positions, edge_index)
        torch.testing.assert_close(dirs, torch.tensor([[0.0, 1.0]]))

    def test_diagonal_edge(self):
        positions = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
        edge_index = torch.tensor([[0], [1]], dtype=torch.long)
        dirs = compute_edge_directions(positions, edge_index)
        expected = torch.tensor([[1.0, 1.0]]) / math.sqrt(2)
        torch.testing.assert_close(dirs, expected, atol=1e-6, rtol=1e-5)

    def test_batched_input(self):
        """Accepts (B, N, 2) input."""
        positions = torch.tensor([[[0.0, 0.0], [2.0, 0.0]]])
        edge_index = torch.tensor([[0], [1]], dtype=torch.long)
        dirs = compute_edge_directions(positions, edge_index)
        torch.testing.assert_close(dirs, torch.tensor([[1.0, 0.0]]))

    def test_normalized(self):
        """Output vectors are unit length."""
        positions = torch.tensor([[0.0, 0.0], [3.0, 4.0]])
        edge_index = torch.tensor([[0], [1]], dtype=torch.long)
        dirs = compute_edge_directions(positions, edge_index)
        norms = dirs.norm(dim=-1)
        torch.testing.assert_close(norms, torch.ones(1), atol=1e-6, rtol=1e-5)


class TestComputeEdgeAngles:
    def test_horizontal(self):
        dirs = torch.tensor([[1.0, 0.0]])
        angles = compute_edge_angles(dirs)
        torch.testing.assert_close(angles, torch.tensor([0.0]), atol=1e-6, rtol=1e-5)

    def test_vertical(self):
        dirs = torch.tensor([[0.0, 1.0]])
        angles = compute_edge_angles(dirs)
        torch.testing.assert_close(angles, torch.tensor([math.pi / 2]), atol=1e-6, rtol=1e-5)

    def test_diagonal_45(self):
        s = math.sqrt(2) / 2
        dirs = torch.tensor([[s, s]])
        angles = compute_edge_angles(dirs)
        torch.testing.assert_close(angles, torch.tensor([math.pi / 4]), atol=1e-6, rtol=1e-5)

    def test_opposite_direction_maps_same(self):
        """Edges pointing left and right map to the same angle (mod π)."""
        dirs = torch.tensor([[1.0, 0.0], [-1.0, 0.0]])
        angles = compute_edge_angles(dirs)
        torch.testing.assert_close(angles[0], angles[1], atol=1e-6, rtol=1e-5)


class TestEdgesFromAdjacency:
    def test_basic_square(self, perfect_square):
        _, adjacency, _ = perfect_square
        edge_index = edges_from_adjacency(adjacency)
        # 4 edges in upper triangle.
        assert edge_index.shape[0] == 2
        assert edge_index.shape[1] == 4

    def test_threshold(self):
        adj = torch.tensor([[[0.0, 0.3], [0.3, 0.0]]])
        edge_index = edges_from_adjacency(adj, threshold=0.5)
        assert edge_index.shape[1] == 0
        edge_index = edges_from_adjacency(adj, threshold=0.2)
        assert edge_index.shape[1] == 1

    def test_node_mask(self):
        adj = torch.ones(1, 3, 3)
        adj[:, range(3), range(3)] = 0
        mask = torch.tensor([[True, True, False]])
        edge_index = edges_from_adjacency(adj, node_mask=mask)
        # Only edge 0-1 should survive.
        assert edge_index.shape[1] == 1

    def test_2d_input(self):
        """Accepts unbatched (N, N) adjacency."""
        adj = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
        edge_index = edges_from_adjacency(adj)
        assert edge_index.shape == (2, 1)

    def test_coo_format(self):
        adj = torch.tensor([[[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]]])
        edge_index = edges_from_adjacency(adj)
        # Upper tri: (0,1) and (1,2).
        assert edge_index.shape[1] == 2
        assert edge_index[0, 0] < edge_index[1, 0]  # src < dst in upper-tri


# ---------------------------------------------------------------------------
# Orthogonal Integrity Axiom (C-001)
# ---------------------------------------------------------------------------


class TestOrthogonalIntegrityAxiom:
    def test_perfect_square_low_loss(self, perfect_square):
        pos, adj, ei = perfect_square
        axiom = OrthogonalIntegrityAxiom()
        result = axiom(pos, adj, ei)
        assert result.loss.item() < 1e-5

    def test_non_orthogonal_high_loss(self):
        """Triangle with 60° angles → loss > 0 (not 0° or 90°)."""
        positions = torch.tensor(
            [[[0.0, 0.0], [2.0, 0.0], [1.0, math.sqrt(3)]]],
            dtype=torch.float32,
        )
        adj = torch.zeros(1, 3, 3)
        for i, j in [(0, 1), (1, 2), (2, 0)]:
            adj[0, i, j] = adj[0, j, i] = 1.0
        ei = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
        axiom = OrthogonalIntegrityAxiom()
        result = axiom(positions, adj, ei)
        assert result.loss.item() > 0.01

    def test_single_edge_zero_loss(self):
        """Single edge has no adjacent pairs → loss = 0."""
        pos = torch.tensor([[[0.0, 0.0], [1.0, 0.0]]])
        adj = torch.zeros(1, 2, 2)
        ei = torch.tensor([[0], [1]], dtype=torch.long)
        axiom = OrthogonalIntegrityAxiom()
        result = axiom(pos, adj, ei)
        assert result.loss.item() == pytest.approx(0.0, abs=1e-7)

    def test_returns_axiom_result(self, perfect_square):
        pos, adj, ei = perfect_square
        axiom = OrthogonalIntegrityAxiom()
        result = axiom(pos, adj, ei)
        assert isinstance(result, AxiomResult)
        assert result.name == "orthogonal"
        assert result.loss.shape == ()
        assert isinstance(result.weight, torch.nn.Parameter)

    def test_differentiable(self):
        """Loss is differentiable w.r.t. node positions."""
        # Asymmetric V-shape: two edges at ~60° from a junction.
        positions = torch.tensor(
            [[[0.0, 0.0], [2.0, 0.0], [0.5, 1.5]]],
            dtype=torch.float32,
            requires_grad=True,
        )
        adj = torch.zeros(1, 3, 3)
        adj[0, 0, 1] = adj[0, 1, 0] = 1.0
        adj[0, 0, 2] = adj[0, 2, 0] = 1.0
        ei = torch.tensor([[0, 0], [1, 2]], dtype=torch.long)
        axiom = OrthogonalIntegrityAxiom()
        result = axiom(positions, adj, ei)
        result.loss.backward()
        assert positions.grad is not None
        assert not torch.all(positions.grad == 0)

    def test_no_edges(self):
        """Empty edge set returns zero loss."""
        pos = torch.tensor([[[0.0, 0.0], [1.0, 1.0]]])
        adj = torch.zeros(1, 2, 2)
        ei = torch.zeros(2, 0, dtype=torch.long)
        axiom = OrthogonalIntegrityAxiom()
        result = axiom(pos, adj, ei)
        assert result.loss.item() == pytest.approx(0.0, abs=1e-7)


# ---------------------------------------------------------------------------
# Parallel Pair Constancy Axiom (C-002)
# ---------------------------------------------------------------------------


class TestParallelPairConstancyAxiom:
    def test_uniform_parallel_edges_low_loss(self):
        """Two horizontal parallel edges at uniform distance → near-zero loss."""
        positions = torch.tensor(
            [[[0.0, 0.0], [2.0, 0.0], [0.0, 1.0], [2.0, 1.0]]],
            dtype=torch.float32,
        )
        adj = torch.zeros(1, 4, 4)
        # Edge 0-1 (bottom) and edge 2-3 (top) are parallel.
        edge_index = torch.tensor([[0, 2], [1, 3]], dtype=torch.long)
        axiom = ParallelPairConstancyAxiom()
        result = axiom(positions, adj, edge_index)
        # Only one pair, so no distance variance → loss = 0.
        assert result.loss.item() < 1e-5

    def test_varying_distance_parallel_loss(self):
        """Parallel edges with extreme distance outlier → loss > 0."""
        # 4 horizontal parallel edges: distances from edge 0 are 1.0, 1.0, 1.0, 20.0.
        # The extreme outlier should exceed the IQR margin.
        positions = torch.tensor(
            [
                [
                    [0.0, 0.0],
                    [2.0, 0.0],  # edge 0: y=0
                    [0.0, 1.0],
                    [2.0, 1.0],  # edge 1: y=1
                    [0.0, 2.0],
                    [2.0, 2.0],  # edge 2: y=2
                    [0.0, 3.0],
                    [2.0, 3.0],  # edge 3: y=3
                    [0.0, 20.0],
                    [2.0, 20.0],  # edge 4: y=20 (extreme outlier)
                ]
            ],
            dtype=torch.float32,
        )
        adj = torch.zeros(1, 10, 10)
        edge_index = torch.tensor([[0, 2, 4, 6, 8], [1, 3, 5, 7, 9]], dtype=torch.long)
        axiom = ParallelPairConstancyAxiom()
        result = axiom(positions, adj, edge_index)
        assert result.loss.item() > 0.0

    def test_no_parallel_edges_zero_loss(self):
        """All edges perpendicular → no parallel pairs → zero loss."""
        positions = torch.tensor(
            [[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]],
            dtype=torch.float32,
        )
        adj = torch.zeros(1, 3, 3)
        # Edge 0: horizontal, edge 1: vertical.
        edge_index = torch.tensor([[0, 0], [1, 2]], dtype=torch.long)
        axiom = ParallelPairConstancyAxiom()
        result = axiom(positions, adj, edge_index)
        assert result.loss.item() == pytest.approx(0.0, abs=1e-7)

    def test_returns_axiom_result(self):
        positions = torch.tensor(
            [[[0.0, 0.0], [2.0, 0.0], [0.0, 1.0], [2.0, 1.0]]],
        )
        adj = torch.zeros(1, 4, 4)
        edge_index = torch.tensor([[0, 2], [1, 3]], dtype=torch.long)
        axiom = ParallelPairConstancyAxiom()
        result = axiom(positions, adj, edge_index)
        assert isinstance(result, AxiomResult)
        assert result.name == "parallel_pair"

    def test_fewer_than_two_edges(self):
        """Single edge → no pairs possible → zero loss."""
        positions = torch.tensor([[[0.0, 0.0], [1.0, 0.0]]])
        adj = torch.zeros(1, 2, 2)
        edge_index = torch.tensor([[0], [1]], dtype=torch.long)
        axiom = ParallelPairConstancyAxiom()
        result = axiom(positions, adj, edge_index)
        assert result.loss.item() == pytest.approx(0.0, abs=1e-7)


# ---------------------------------------------------------------------------
# Junction Closure Axiom (C-003)
# ---------------------------------------------------------------------------


class TestJunctionClosureAxiom:
    def test_complete_cycle_low_loss(self, perfect_square):
        """All nodes degree 2 → low loss."""
        pos, adj, ei = perfect_square
        axiom = JunctionClosureAxiom()
        result = axiom(pos, adj, ei)
        # Dangling penalty should be zero (all degree >= 2).
        # Laplacian loss depends on geometry but should be small for a clean square.
        assert result.loss.item() < 5.0  # lenient — Laplacian isn't zero for non-trivial graphs

    def test_dangling_node_violation(self, graph_with_dangling):
        """Node with degree 1 → violation detected."""
        pos, adj, ei = graph_with_dangling
        axiom = JunctionClosureAxiom()
        result = axiom(pos, adj, ei)
        assert result.loss.item() > 0.0
        # At least some violation mask entries should be True.
        assert result.violation_mask.any()

    def test_isolated_node_violation(self):
        """Node with degree 0 → violation detected."""
        positions = torch.tensor(
            [[[0.0, 0.0], [1.0, 0.0], [2.0, 2.0]]],  # node 2 is isolated
            dtype=torch.float32,
        )
        adj = torch.zeros(1, 3, 3)
        adj[0, 0, 1] = 1.0
        adj[0, 1, 0] = 1.0
        edge_index = torch.tensor([[0], [1]], dtype=torch.long)
        axiom = JunctionClosureAxiom()
        result = axiom(positions, adj, edge_index)
        assert result.loss.item() > 0.0
        # Node 2 (degree 0) should be flagged.
        assert result.violation_mask[2].item() is True

    def test_differentiable(self, graph_with_dangling):
        pos, adj, ei = graph_with_dangling
        pos = pos.clone().requires_grad_(True)
        adj_r = adj.clone().requires_grad_(True)
        axiom = JunctionClosureAxiom()
        result = axiom(pos, adj_r, ei)
        result.loss.backward()
        assert pos.grad is not None

    def test_returns_axiom_result(self, perfect_square):
        pos, adj, ei = perfect_square
        axiom = JunctionClosureAxiom()
        result = axiom(pos, adj, ei)
        assert isinstance(result, AxiomResult)
        assert result.name == "junction_closure"

    def test_node_mask(self):
        """Masked nodes should not contribute to violations."""
        positions = torch.tensor(
            [[[0.0, 0.0], [1.0, 0.0], [2.0, 2.0]]],
            dtype=torch.float32,
        )
        adj = torch.zeros(1, 3, 3)
        adj[0, 0, 1] = 1.0
        adj[0, 1, 0] = 1.0
        mask = torch.tensor([[True, True, False]])
        edge_index = torch.tensor([[0], [1]], dtype=torch.long)
        axiom = JunctionClosureAxiom()
        result = axiom(positions, adj, edge_index, node_mask=mask)
        # Node 2 is masked, should not appear as violation.
        assert result.violation_mask[2].item() is False


# ---------------------------------------------------------------------------
# Spatial Non-Intersection Axiom (C-004)
# ---------------------------------------------------------------------------


class TestSpatialNonIntersectionAxiom:
    def test_non_crossing_zero_loss(self, perfect_square):
        """Non-crossing edges → loss near 0."""
        pos, adj, ei = perfect_square
        axiom = SpatialNonIntersectionAxiom()
        result = axiom(pos, adj, ei)
        assert result.loss.item() < 1e-5

    def test_crossing_edges_positive_loss(self, crossing_edges):
        """Crossing edges → loss > 0."""
        pos, adj, ei = crossing_edges
        axiom = SpatialNonIntersectionAxiom()
        result = axiom(pos, adj, ei)
        assert result.loss.item() > 0.0

    def test_adjacent_edges_not_penalized(self):
        """Edges sharing a node are adjacent and should NOT be penalized."""
        # V-shape: edges 0-1 and 1-2 share node 1.
        positions = torch.tensor([[[0.0, 0.0], [0.5, 0.5], [1.0, 0.0]]])
        adj = torch.zeros(1, 3, 3)
        edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
        axiom = SpatialNonIntersectionAxiom()
        result = axiom(positions, adj, edge_index)
        # Adjacent edges are excluded → no non-adjacent pairs → zero loss.
        assert result.loss.item() == pytest.approx(0.0, abs=1e-7)

    def test_differentiable(self, crossing_edges):
        pos, adj, ei = crossing_edges
        pos = pos.clone().requires_grad_(True)
        axiom = SpatialNonIntersectionAxiom()
        result = axiom(pos, adj, ei)
        result.loss.backward()
        assert pos.grad is not None

    def test_single_edge(self):
        """Single edge → no pairs → zero loss."""
        positions = torch.tensor([[[0.0, 0.0], [1.0, 1.0]]])
        adj = torch.zeros(1, 2, 2)
        edge_index = torch.tensor([[0], [1]], dtype=torch.long)
        axiom = SpatialNonIntersectionAxiom()
        result = axiom(positions, adj, edge_index)
        assert result.loss.item() == pytest.approx(0.0, abs=1e-7)

    def test_returns_axiom_result(self, crossing_edges):
        pos, adj, ei = crossing_edges
        axiom = SpatialNonIntersectionAxiom()
        result = axiom(pos, adj, ei)
        assert isinstance(result, AxiomResult)
        assert result.name == "non_intersection"


# ---------------------------------------------------------------------------
# Axiom Registry (C-008)
# ---------------------------------------------------------------------------


class TestAxiomRegistry:
    def test_create_default(self):
        config = ConstraintConfig()
        registry = AxiomRegistry.create_default(config)
        names = registry.list_axioms()
        assert len(names) == 4
        assert "orthogonal" in names
        assert "parallel_pair" in names
        assert "junction_closure" in names
        assert "non_intersection" in names

    def test_register_unregister(self):
        registry = AxiomRegistry()
        axiom = OrthogonalIntegrityAxiom()
        registry.register(axiom)
        assert "orthogonal" in registry.list_axioms()
        registry.unregister("orthogonal")
        assert "orthogonal" not in registry.list_axioms()

    def test_register_duplicate_raises(self):
        registry = AxiomRegistry()
        axiom = OrthogonalIntegrityAxiom()
        registry.register(axiom)
        with pytest.raises(ValueError, match="already registered"):
            registry.register(axiom)

    def test_unregister_missing_raises(self):
        registry = AxiomRegistry()
        with pytest.raises(KeyError, match="not registered"):
            registry.unregister("nonexistent")

    def test_evaluate_all(self, perfect_square):
        pos, adj, ei = perfect_square
        config = ConstraintConfig()
        registry = AxiomRegistry.create_default(config)
        results = registry.evaluate_all(pos, adj, ei)
        assert len(results) == 4
        for r in results:
            assert isinstance(r, AxiomResult)
            assert r.loss.shape == ()

    def test_list_axioms_returns_names(self):
        config = ConstraintConfig()
        registry = AxiomRegistry.create_default(config)
        names = registry.list_axioms()
        assert isinstance(names, list)
        assert all(isinstance(n, str) for n in names)

    def test_get(self):
        config = ConstraintConfig()
        registry = AxiomRegistry.create_default(config)
        assert registry.get("orthogonal") is not None
        assert registry.get("nonexistent") is None

    def test_frozen_weights(self):
        config = ConstraintConfig(learn_weights=False)
        registry = AxiomRegistry.create_default(config)
        for name in registry.list_axioms():
            axiom = registry.get(name)
            assert not axiom.weight.requires_grad
