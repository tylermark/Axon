"""Benchmark: Graph Edit Distance (GED) for structural accuracy.

Q-009: GED measures the minimum cost to transform the predicted graph into
the ground truth graph via node/edge insertions, deletions, and
substitutions.  Node substitution cost is the Euclidean distance
between positions (clamped to ``max_cost``).

Reference: ARCHITECTURE.md (Evaluation Metrics), MODEL_SPEC.md.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import pytest

# ---------------------------------------------------------------------------
# GED implementation
# ---------------------------------------------------------------------------


def build_nx_graph(nodes: np.ndarray, edges: np.ndarray) -> nx.Graph:
    """Build a NetworkX graph from node positions and edge indices.

    Args:
        nodes: Node coordinates, shape (N, 2).
        edges: Edge index pairs, shape (E, 2).

    Returns:
        An undirected ``nx.Graph`` with ``pos`` node attributes.
    """
    g = nx.Graph()
    for i, (x, y) in enumerate(nodes):
        g.add_node(i, pos=(float(x), float(y)))
    for src, dst in edges:
        g.add_edge(int(src), int(dst))
    return g


def compute_ged(
    pred_nodes: np.ndarray,
    pred_edges: np.ndarray,
    gt_nodes: np.ndarray,
    gt_edges: np.ndarray,
    max_cost: float = 10.0,
) -> float:
    """Compute approximate Graph Edit Distance with position-aware node cost.

    Uses exact GED for small graphs (combined size <= 20 nodes) and the
    ``optimize_graph_edit_distance`` generator for larger ones.

    Args:
        pred_nodes: Predicted node positions, shape (N1, 2).
        pred_edges: Predicted edges, shape (E1, 2).
        gt_nodes: Ground truth node positions, shape (N2, 2).
        gt_edges: Ground truth edges, shape (E2, 2).
        max_cost: Maximum node substitution cost (clamp).

    Returns:
        The (approximate) graph edit distance.
    """
    g1 = build_nx_graph(pred_nodes, pred_edges)
    g2 = build_nx_graph(gt_nodes, gt_edges)

    def node_subst_cost(n1_attrs: dict, n2_attrs: dict) -> float:
        p1 = np.array(n1_attrs["pos"])
        p2 = np.array(n2_attrs["pos"])
        return min(float(np.linalg.norm(p1 - p2)), max_cost)

    combined_size = len(g1) + len(g2)
    if combined_size > 20:
        # Approximate: take the first (best) upper bound.
        return next(
            nx.optimize_graph_edit_distance(
                g1,
                g2,
                node_subst_cost=node_subst_cost,
            )
        )
    else:
        result = nx.graph_edit_distance(
            g1,
            g2,
            node_subst_cost=node_subst_cost,
        )
        assert result is not None
        return float(result)


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _square_graph() -> tuple[np.ndarray, np.ndarray]:
    """Return a 4-node square graph (nodes + edges)."""
    nodes = np.array(
        [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=np.float64,
    )
    edges = np.array(
        [[0, 1], [1, 2], [2, 3], [3, 0]], dtype=np.int64,
    )
    return nodes, edges


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.benchmark
class TestGED:
    """Unit tests for the GED metric."""

    def test_ged_identical_graphs(self):
        """GED between identical graphs is 0."""
        nodes, edges = _square_graph()
        ged = compute_ged(nodes, edges, nodes, edges)
        assert ged == pytest.approx(0.0, abs=1e-6)

    def test_ged_one_edge_diff(self):
        """Removing one edge from the prediction yields GED = 1."""
        nodes, edges = _square_graph()
        # Remove the last edge (3 -> 0).
        pred_edges = edges[:3]
        ged = compute_ged(nodes, pred_edges, nodes, edges)
        assert ged == pytest.approx(1.0, abs=1e-6)

    def test_ged_node_position_shift(self):
        """Small node shifts increase GED proportionally."""
        nodes, edges = _square_graph()
        shifted = nodes.copy()
        shifted[:, 0] += 0.5  # Shift all x-coords by 0.5

        ged_shifted = compute_ged(shifted, edges, nodes, edges)
        # Each of the 4 nodes incurs substitution cost = 0.5.
        assert ged_shifted == pytest.approx(4 * 0.5, abs=1e-6)

    def test_ged_extra_node(self):
        """An extra node in prediction costs 1 (deletion)."""
        gt_nodes, gt_edges = _square_graph()
        pred_nodes = np.vstack([gt_nodes, [[0.5, 0.5]]])
        pred_edges = gt_edges.copy()
        ged = compute_ged(pred_nodes, pred_edges, gt_nodes, gt_edges)
        assert ged == pytest.approx(1.0, abs=1e-6)

    def test_ged_symmetric(self):
        """GED(A, B) == GED(B, A)."""
        nodes, edges = _square_graph()
        pred_edges = edges[:3]
        ged_ab = compute_ged(nodes, pred_edges, nodes, edges)
        ged_ba = compute_ged(nodes, edges, nodes, pred_edges)
        assert ged_ab == pytest.approx(ged_ba, abs=1e-6)


@pytest.mark.slow
@pytest.mark.benchmark
class TestGEDPipeline:
    """End-to-end: run Layer1Pipeline and compute GED on the output."""

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

    def test_ged_on_pipeline_output(self, pipeline, tmp_path):
        """GED metric runs end-to-end on pipeline output (untrained)."""
        from tests.fixtures.pdf_factory import create_simple_rect_pdf

        pdf = create_simple_rect_pdf(
            tmp_path / "rect.pdf", x=100, y=100, w=200, h=150, stroke_width=1.5,
        )
        result = pipeline.extract(str(pdf), page_index=0, use_raster=False)

        # Ground truth: a simple 4-node rectangle.
        gt_nodes = np.array(
            [[100, 100], [300, 100], [300, 250], [100, 250]], dtype=np.float64,
        )
        gt_edges = np.array(
            [[0, 1], [1, 2], [2, 3], [3, 0]], dtype=np.int64,
        )

        # Untrained models produce arbitrary graphs.  Just verify the
        # metric runs without error and returns a non-negative float.
        if result.nodes.shape[0] > 0 and result.edges.shape[0] > 0:
            ged = compute_ged(result.nodes, result.edges, gt_nodes, gt_edges)
            assert ged >= 0.0
            assert np.isfinite(ged)
