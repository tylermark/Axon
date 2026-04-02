"""Benchmark: Betti Number Error for topological correctness.

Q-010: Betti numbers quantify the topological structure of the extracted graph:
    - Betti-0 = number of connected components (ideally 1 for a valid plan).
    - Betti-1 = number of enclosed loops / rooms.

The Betti error is the absolute difference between predicted and ground truth
Betti numbers.

Reference: ARCHITECTURE.md (Topology Stage), MODEL_SPEC.md.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

# ---------------------------------------------------------------------------
# Betti number implementation
# ---------------------------------------------------------------------------


def compute_betti_numbers(
    nodes: np.ndarray,
    edges: np.ndarray,
) -> tuple[int, int]:
    """Compute Betti-0 and Betti-1 from graph structure.

    Uses the Euler formula for planar graphs:
        V - E + F = 2 * C   (C = connected components)
        betti_1 = E_unique - V + betti_0

    Args:
        nodes: Node positions, shape (N, 2).
        edges: Edge index pairs, shape (E, 2).

    Returns:
        Tuple of (betti_0, betti_1).
    """
    n = len(nodes)
    if n == 0:
        return 0, 0

    e = len(edges)
    if e == 0:
        return n, 0  # Each node is its own connected component.

    row = edges[:, 0]
    col = edges[:, 1]
    data = np.ones(e)
    adj = csr_matrix((data, (row, col)), shape=(n, n))
    adj = adj + adj.T
    adj.data = np.clip(adj.data, 0, 1)

    betti_0 = int(connected_components(adj, directed=False, return_labels=False))

    # Count unique undirected edges.
    unique_edges = len({tuple(sorted(edge)) for edge in edges})
    betti_1 = max(0, unique_edges - n + betti_0)

    return betti_0, betti_1


def compute_betti_error(
    pred_nodes: np.ndarray,
    pred_edges: np.ndarray,
    gt_betti_0: int,
    gt_betti_1: int,
) -> tuple[int, int]:
    """Compute absolute Betti number errors.

    Args:
        pred_nodes: Predicted node positions, shape (N, 2).
        pred_edges: Predicted edge indices, shape (E, 2).
        gt_betti_0: Ground truth Betti-0.
        gt_betti_1: Ground truth Betti-1.

    Returns:
        Tuple of (|b0_pred - b0_gt|, |b1_pred - b1_gt|).
    """
    b0, b1 = compute_betti_numbers(pred_nodes, pred_edges)
    return abs(b0 - gt_betti_0), abs(b1 - gt_betti_1)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.benchmark
class TestBettiNumbers:
    """Unit tests for Betti number computation."""

    def test_betti_single_rectangle(self):
        """4 nodes, 4 edges forming a closed rectangle: B0=1, B1=1."""
        nodes = np.array(
            [[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float64,
        )
        edges = np.array(
            [[0, 1], [1, 2], [2, 3], [3, 0]], dtype=np.int64,
        )
        b0, b1 = compute_betti_numbers(nodes, edges)
        assert b0 == 1
        assert b1 == 1

    def test_betti_two_rooms(self):
        """Two adjacent rectangles sharing a wall: B0=1, B1=2.

        Layout:
            0---1---2
            |   |   |
            3---4---5
        Edges: 0-1, 1-2, 0-3, 1-4, 2-5, 3-4, 4-5
        """
        nodes = np.array(
            [[0, 0], [1, 0], [2, 0], [0, 1], [1, 1], [2, 1]],
            dtype=np.float64,
        )
        edges = np.array(
            [[0, 1], [1, 2], [0, 3], [1, 4], [2, 5], [3, 4], [4, 5]],
            dtype=np.int64,
        )
        b0, b1 = compute_betti_numbers(nodes, edges)
        assert b0 == 1
        assert b1 == 2

    def test_betti_disconnected(self):
        """Two separate rectangles: B0=2, B1=2."""
        nodes = np.array(
            [
                [0, 0], [1, 0], [1, 1], [0, 1],   # Rectangle 1
                [5, 5], [6, 5], [6, 6], [5, 6],   # Rectangle 2
            ],
            dtype=np.float64,
        )
        edges = np.array(
            [
                [0, 1], [1, 2], [2, 3], [3, 0],   # Rectangle 1
                [4, 5], [5, 6], [6, 7], [7, 4],   # Rectangle 2
            ],
            dtype=np.int64,
        )
        b0, b1 = compute_betti_numbers(nodes, edges)
        assert b0 == 2
        assert b1 == 2

    def test_betti_chain(self):
        """Open chain (3 nodes, 2 edges): B0=1, B1=0."""
        nodes = np.array([[0, 0], [1, 0], [2, 0]], dtype=np.float64)
        edges = np.array([[0, 1], [1, 2]], dtype=np.int64)
        b0, b1 = compute_betti_numbers(nodes, edges)
        assert b0 == 1
        assert b1 == 0

    def test_betti_single_node(self):
        """Single node, no edges: B0=1, B1=0."""
        nodes = np.array([[0, 0]], dtype=np.float64)
        edges = np.empty((0, 2), dtype=np.int64)
        b0, b1 = compute_betti_numbers(nodes, edges)
        assert b0 == 1
        assert b1 == 0

    def test_betti_empty_graph(self):
        """No nodes: B0=0, B1=0."""
        nodes = np.empty((0, 2), dtype=np.float64)
        edges = np.empty((0, 2), dtype=np.int64)
        b0, b1 = compute_betti_numbers(nodes, edges)
        assert b0 == 0
        assert b1 == 0


@pytest.mark.benchmark
class TestBettiError:
    """Tests for the Betti error computation helper."""

    def test_betti_error_computation(self):
        """Verify error against known targets."""
        # Single rectangle: B0=1, B1=1.
        nodes = np.array(
            [[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float64,
        )
        edges = np.array(
            [[0, 1], [1, 2], [2, 3], [3, 0]], dtype=np.int64,
        )

        # Perfect match.
        err0, err1 = compute_betti_error(nodes, edges, gt_betti_0=1, gt_betti_1=1)
        assert err0 == 0
        assert err1 == 0

        # Expect 2 rooms but graph has 1.
        err0, err1 = compute_betti_error(nodes, edges, gt_betti_0=1, gt_betti_1=2)
        assert err0 == 0
        assert err1 == 1

        # Expect 1 component but graph has 2 (use disconnected nodes).
        chain_nodes = np.array([[0, 0], [1, 0], [5, 5]], dtype=np.float64)
        chain_edges = np.array([[0, 1]], dtype=np.int64)
        err0, err1 = compute_betti_error(chain_nodes, chain_edges, gt_betti_0=1, gt_betti_1=0)
        assert err0 == 1  # Predicted B0=2, GT B0=1.
        assert err1 == 0


@pytest.mark.slow
@pytest.mark.benchmark
class TestBettiPipeline:
    """End-to-end: verify FinalizedGraph Betti fields are populated."""

    @pytest.fixture(scope="class")
    def pipeline(self):
        from src.pipeline.config import (
            AxonConfig,
            DiffusionConfig,
            NoiseSchedule,
            TokenizerConfig,
        )
        from src.pipeline.config import VisionBackbone as VisionBackboneType
        from src.pipeline.layer1 import Layer1Pipeline

        cfg = AxonConfig(
            tokenizer=TokenizerConfig(
                d_model=256,
                n_heads=8,
                vision_backbone=VisionBackboneType.HRNET_W32,
                dropout=0.0,
            ),
            diffusion=DiffusionConfig(
                d_model=64,
                n_heads=4,
                n_layers=2,
                timesteps_train=100,
                timesteps_inference=10,
                noise_schedule=NoiseSchedule.COSINE,
                max_nodes=16,
                use_hdse=True,
                hdse_max_distance=5,
                dropout=0.0,
            ),
            device="cpu",
        )
        return Layer1Pipeline(config=cfg, device="cpu")

    def test_betti_on_pipeline_output(self, pipeline, tmp_path):
        """FinalizedGraph.betti_0 and betti_1 are populated integers."""
        from tests.fixtures.pdf_factory import create_simple_rect_pdf

        pdf = create_simple_rect_pdf(
            tmp_path / "rect.pdf", x=100, y=100, w=200, h=150, stroke_width=1.5,
        )
        result = pipeline.extract(str(pdf), page_index=0, use_raster=False)

        assert isinstance(result.betti_0, int)
        assert isinstance(result.betti_1, int)
        assert result.betti_0 >= 0
        assert result.betti_1 >= 0

        # If the graph has edges, verify our standalone computation matches.
        if result.nodes.shape[0] > 0 and result.edges.shape[0] > 0:
            b0, b1 = compute_betti_numbers(result.nodes, result.edges)
            assert b0 == result.betti_0
            assert b1 == result.betti_1
