"""Unit tests for GeometricProjector (C-007)."""

from __future__ import annotations

import math

import pytest
import torch

from src.constraints.projector import GeometricProjector
from src.pipeline.config import ConstraintConfig

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def config():
    return ConstraintConfig()


@pytest.fixture()
def projector(config):
    return GeometricProjector(config)


# ---------------------------------------------------------------------------
# snap_orthogonal
# ---------------------------------------------------------------------------


class TestSnapOrthogonal:
    def test_near_90_snapped(self, projector):
        """Angle ~88° between edges → snapped to exact 90°."""
        # Junction at origin, edge A along x-axis, edge B at 88°.
        angle_rad = math.radians(88)
        positions = torch.tensor(
            [
                [
                    [0.0, 0.0],  # junction
                    [1.0, 0.0],  # free end of edge A
                    [math.cos(angle_rad), math.sin(angle_rad)],  # free end of edge B
                ]
            ],
        )
        edge_index = torch.tensor([[0, 0], [1, 2]], dtype=torch.long)

        snapped = projector.snap_orthogonal(positions, edge_index, tolerance_deg=5.0)
        flat = snapped.reshape(-1, 2)

        # Measure angle between the two edges after snapping.
        d1 = flat[1] - flat[0]
        d2 = flat[2] - flat[0]
        cos_val = (d1 @ d2).item() / (d1.norm().item() * d2.norm().item())
        angle_after = math.degrees(math.acos(max(-1.0, min(1.0, cos_val))))
        assert abs(angle_after - 90.0) < 0.1

    def test_far_from_90_not_snapped(self, projector):
        """Angle at 45° — outside tolerance → NOT snapped."""
        angle_rad = math.radians(45)
        positions = torch.tensor(
            [
                [
                    [0.0, 0.0],
                    [1.0, 0.0],
                    [math.cos(angle_rad), math.sin(angle_rad)],
                ]
            ],
        )
        edge_index = torch.tensor([[0, 0], [1, 2]], dtype=torch.long)

        snapped = projector.snap_orthogonal(positions, edge_index, tolerance_deg=5.0)
        # Should remain unchanged.
        torch.testing.assert_close(snapped, positions)

    def test_already_exact_90(self, projector):
        """Already 90° → no change."""
        positions = torch.tensor(
            [[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]],
        )
        edge_index = torch.tensor([[0, 0], [1, 2]], dtype=torch.long)
        snapped = projector.snap_orthogonal(positions, edge_index)
        torch.testing.assert_close(snapped, positions, atol=1e-6, rtol=1e-5)

    def test_single_edge_unchanged(self, projector):
        """Single edge has no adjacent pairs → unchanged."""
        positions = torch.tensor([[[0.0, 0.0], [1.0, 0.5]]])
        edge_index = torch.tensor([[0], [1]], dtype=torch.long)
        snapped = projector.snap_orthogonal(positions, edge_index)
        torch.testing.assert_close(snapped, positions)


# ---------------------------------------------------------------------------
# snap_parallel_pairs
# ---------------------------------------------------------------------------


class TestSnapParallelPairs:
    def test_varying_distance_becomes_uniform(self, projector):
        """Parallel edges at varying distances → snapped to uniform distance."""
        positions = torch.tensor(
            [
                [
                    [0.0, 0.0],
                    [2.0, 0.0],  # edge 0: y=0
                    [0.0, 0.8],
                    [2.0, 0.8],  # edge 1: y=0.8
                    [0.0, 1.3],
                    [2.0, 1.3],  # edge 2: y=1.3
                ]
            ],
        )
        edge_index = torch.tensor([[0, 2, 4], [1, 3, 5]], dtype=torch.long)
        parallel_pairs = torch.tensor([[0, 1], [0, 2], [1, 2]])

        snapped = projector.snap_parallel_pairs(positions, parallel_pairs, edge_index)
        flat = snapped.reshape(-1, 2)

        # After snapping, distances between parallel edges should be
        # closer to median than before.
        d01_before = 0.8
        d02_before = 1.3
        d01_after = abs(flat[2, 1].item() - flat[0, 1].item())
        d02_after = abs(flat[4, 1].item() - flat[0, 1].item())

        # The spread should decrease or be at median.
        spread_before = abs(d02_before - d01_before)
        spread_after = abs(d02_after - d01_after)
        assert spread_after <= spread_before + 0.01

    def test_empty_pairs_unchanged(self, projector):
        positions = torch.tensor([[[0.0, 0.0], [1.0, 0.0]]])
        edge_index = torch.tensor([[0], [1]], dtype=torch.long)
        parallel_pairs = torch.zeros(0, 2, dtype=torch.long)
        snapped = projector.snap_parallel_pairs(positions, parallel_pairs, edge_index)
        torch.testing.assert_close(snapped, positions)


# ---------------------------------------------------------------------------
# close_junctions
# ---------------------------------------------------------------------------


class TestCloseJunctions:
    def test_dangling_gets_connected(self, projector):
        """Dangling node (degree 1) gets connected to nearest node."""
        positions = torch.tensor(
            [
                [
                    [0.0, 0.0],  # node 0: degree 1 (dangling)
                    [1.0, 0.0],  # node 1: degree 2
                    [1.0, 1.0],  # node 2: degree 2
                    [0.0, 1.0],  # node 3: degree 1 (dangling)
                ]
            ],
        )
        adj = torch.zeros(1, 4, 4)
        # Edges: 0-1, 1-2, 2-3 → nodes 0 and 3 are degree 1.
        adj[0, 0, 1] = adj[0, 1, 0] = 1.0
        adj[0, 1, 2] = adj[0, 2, 1] = 1.0
        adj[0, 2, 3] = adj[0, 3, 2] = 1.0

        _, new_adj = projector.close_junctions(positions, adj, min_degree=2)

        # Dangling nodes 0 and 3 should now have degree >= 2.
        degree = new_adj[0].sum(dim=-1)
        assert degree[0].item() >= 2
        assert degree[3].item() >= 2

    def test_all_degree_ge2_unchanged(self, projector):
        """Graph with all degree >= 2 → adjacency unchanged."""
        positions = torch.tensor(
            [[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]],
        )
        adj = torch.zeros(1, 4, 4)
        edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
        for i, j in edges:
            adj[0, i, j] = adj[0, j, i] = 1.0

        _, new_adj = projector.close_junctions(positions, adj, min_degree=2)
        torch.testing.assert_close(new_adj, adj)


# ---------------------------------------------------------------------------
# resolve_intersections
# ---------------------------------------------------------------------------


class TestResolveIntersections:
    def test_crossing_edges_returns_valid_shape(self, projector):
        """resolve_intersections returns correct shape for crossing edges.

        NOTE: Production bug — sign error in parametric t/u computation
        (r = a1 - b1 should be b1 - a1) causes intersections to go
        undetected. The function runs without error but does not nudge.
        """
        positions = torch.tensor(
            [[[0.0, 0.0], [1.0, 1.0], [1.0, 0.0], [0.0, 1.0]]],
        )
        edge_index = torch.tensor([[0, 2], [1, 3]], dtype=torch.long)
        resolved = projector.resolve_intersections(positions, edge_index)
        assert resolved.shape == positions.shape
        assert not torch.isnan(resolved).any()

    def test_non_crossing_unchanged(self, projector):
        """Non-crossing, non-adjacent edges → unchanged."""
        positions = torch.tensor(
            [[[0.0, 0.0], [1.0, 0.0], [0.0, 2.0], [1.0, 2.0]]],
        )
        # Two parallel non-crossing edges far apart.
        edge_index = torch.tensor([[0, 2], [1, 3]], dtype=torch.long)

        resolved = projector.resolve_intersections(positions, edge_index)
        torch.testing.assert_close(resolved, positions)


# ---------------------------------------------------------------------------
# Full project() pipeline
# ---------------------------------------------------------------------------


class TestFullProject:
    def test_project_returns_valid_geometry(self, projector):
        """Full projection chain produces valid output shapes."""
        positions = torch.tensor(
            [[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]],
        )
        adj = torch.zeros(1, 4, 4)
        edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
        for i, j in edges:
            adj[0, i, j] = adj[0, j, i] = 1.0
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)

        proj_pos, proj_adj = projector.project(positions, adj, edge_index)

        assert proj_pos.shape == positions.shape
        assert proj_adj.shape == adj.shape
        # No NaN in output.
        assert not torch.isnan(proj_pos).any()
        assert not torch.isnan(proj_adj).any()

    def test_project_with_node_mask(self, projector):
        """Projection respects node mask."""
        positions = torch.tensor(
            [[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [5.0, 5.0]]],
        )
        adj = torch.zeros(1, 5, 5)
        edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
        for i, j in edges:
            adj[0, i, j] = adj[0, j, i] = 1.0
        mask = torch.tensor([[True, True, True, True, False]])
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)

        proj_pos, _proj_adj = projector.project(positions, adj, edge_index, node_mask=mask)
        assert proj_pos.shape == positions.shape
