"""Unit tests for src/parser/filters.py — decorative element heuristics."""

from __future__ import annotations

import numpy as np
import pytest

from docs.interfaces.parser_to_tokenizer import RawGraph
from src.parser.filters import (
    apply_filters,
    compute_wall_confidence,
    score_color,
    score_dash_pattern,
    score_edge_length,
    score_geometric_regularity,
    score_hatching_detector,
    score_stroke_width,
)
from src.parser.operators import OperatorType

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_graph(
    nodes: np.ndarray,
    edges: np.ndarray,
    stroke_widths: np.ndarray | None = None,
    stroke_colors: np.ndarray | None = None,
    dash_patterns: list[tuple[list[float], float]] | None = None,
    page_width: float = 612.0,
    page_height: float = 792.0,
) -> RawGraph:
    """Create a minimal RawGraph for testing heuristic functions."""
    ne = len(edges)
    if stroke_widths is None:
        stroke_widths = np.ones(ne, dtype=np.float64)
    if stroke_colors is None:
        stroke_colors = np.tile([0.0, 0.0, 0.0, 1.0], (ne, 1)).astype(np.float64)
    if dash_patterns is None:
        dash_patterns = [([], 0.0)] * ne

    return RawGraph(
        nodes=nodes.astype(np.float64),
        edges=edges.astype(np.int64),
        operator_types=[OperatorType.LINETO] * ne,
        stroke_widths=stroke_widths.astype(np.float64),
        stroke_colors=stroke_colors.astype(np.float64),
        fill_colors=None,
        dash_patterns=dash_patterns,
        path_metadata=[],
        edge_to_path=np.zeros(ne, dtype=np.int64),
        bezier_controls=[None] * ne,
        confidence_wall=np.zeros(ne, dtype=np.float64),
        page_width=page_width,
        page_height=page_height,
    )


# ---------------------------------------------------------------------------
# score_stroke_width
# ---------------------------------------------------------------------------


class TestScoreStrokeWidth:
    def test_in_range_high_score(self) -> None:
        nodes = np.array([[0, 0], [100, 0]])
        edges = np.array([[0, 1]])
        graph = _make_graph(nodes, edges, stroke_widths=np.array([1.0]))
        scores = score_stroke_width(graph)
        assert len(scores) == 1
        assert scores[0] > 0.9  # 1.0 is within default range [0.5, 3.0]

    def test_hairline_low_score(self) -> None:
        nodes = np.array([[0, 0], [100, 0]])
        edges = np.array([[0, 1]])
        graph = _make_graph(nodes, edges, stroke_widths=np.array([0.05]))
        scores = score_stroke_width(graph)
        assert scores[0] < 0.1

    def test_very_thick_low_score(self) -> None:
        nodes = np.array([[0, 0], [100, 0]])
        edges = np.array([[0, 1]])
        graph = _make_graph(nodes, edges, stroke_widths=np.array([10.0]))
        scores = score_stroke_width(graph)
        assert scores[0] < 0.5

    def test_empty_graph(self) -> None:
        graph = _make_graph(
            np.empty((0, 2)),
            np.empty((0, 2)),
            stroke_widths=np.empty(0),
        )
        scores = score_stroke_width(graph)
        assert len(scores) == 0


# ---------------------------------------------------------------------------
# score_color
# ---------------------------------------------------------------------------


class TestScoreColor:
    def test_black_high_score(self) -> None:
        nodes = np.array([[0, 0], [100, 0]])
        edges = np.array([[0, 1]])
        colors = np.array([[0.0, 0.0, 0.0, 1.0]])
        graph = _make_graph(nodes, edges, stroke_colors=colors)
        scores = score_color(graph)
        assert scores[0] > 0.9

    def test_red_lower_score(self) -> None:
        nodes = np.array([[0, 0], [100, 0]])
        edges = np.array([[0, 1]])
        colors = np.array([[1.0, 0.0, 0.0, 1.0]])
        graph = _make_graph(nodes, edges, stroke_colors=colors)
        scores = score_color(graph)
        assert scores[0] < 0.8  # red has high saturation + some luminance

    def test_gray_medium_score(self) -> None:
        nodes = np.array([[0, 0], [100, 0]])
        edges = np.array([[0, 1]])
        colors = np.array([[0.5, 0.5, 0.5, 1.0]])
        graph = _make_graph(nodes, edges, stroke_colors=colors)
        scores = score_color(graph)
        # Gray: low saturation (good) but moderate luminance
        assert 0.3 < scores[0] < 0.9

    def test_empty_graph(self) -> None:
        graph = _make_graph(
            np.empty((0, 2)),
            np.empty((0, 2)),
            stroke_colors=np.empty((0, 4)),
        )
        scores = score_color(graph)
        assert len(scores) == 0


# ---------------------------------------------------------------------------
# score_dash_pattern
# ---------------------------------------------------------------------------


class TestScoreDashPattern:
    def test_solid_line_score_1(self) -> None:
        nodes = np.array([[0, 0], [100, 0]])
        edges = np.array([[0, 1]])
        graph = _make_graph(nodes, edges, dash_patterns=[([], 0.0)])
        scores = score_dash_pattern(graph)
        assert scores[0] == pytest.approx(1.0)

    def test_dashed_line_lower_score(self) -> None:
        nodes = np.array([[0, 0], [100, 0]])
        edges = np.array([[0, 1]])
        graph = _make_graph(nodes, edges, dash_patterns=[([3.0, 3.0], 0.0)])
        scores = score_dash_pattern(graph)
        assert scores[0] < 0.6

    def test_empty_graph(self) -> None:
        graph = _make_graph(
            np.empty((0, 2)),
            np.empty((0, 2)),
            dash_patterns=[],
        )
        scores = score_dash_pattern(graph)
        assert len(scores) == 0


# ---------------------------------------------------------------------------
# score_geometric_regularity
# ---------------------------------------------------------------------------


class TestScoreGeometricRegularity:
    def test_horizontal_edge_high_score(self) -> None:
        nodes = np.array([[0, 0], [100, 0]])
        edges = np.array([[0, 1]])
        graph = _make_graph(nodes, edges)
        scores = score_geometric_regularity(graph)
        assert scores[0] > 0.7

    def test_vertical_edge_high_score(self) -> None:
        nodes = np.array([[0, 0], [0, 100]])
        edges = np.array([[0, 1]])
        graph = _make_graph(nodes, edges)
        scores = score_geometric_regularity(graph)
        assert scores[0] > 0.7

    def test_45_degree_edge_lower_score(self) -> None:
        nodes = np.array([[0, 0], [100, 100]])
        edges = np.array([[0, 1]])
        graph = _make_graph(nodes, edges)
        scores = score_geometric_regularity(graph)
        # 45° is farthest from both axes; still ≥ 0.3 floor
        assert scores[0] >= 0.3
        assert scores[0] < 0.8

    def test_empty_graph(self) -> None:
        graph = _make_graph(np.empty((0, 2)), np.empty((0, 2)))
        scores = score_geometric_regularity(graph)
        assert len(scores) == 0


# ---------------------------------------------------------------------------
# score_hatching_detector
# ---------------------------------------------------------------------------


class TestScoreHatchingDetector:
    def test_parallel_evenly_spaced_low_scores(self) -> None:
        """Parallel evenly-spaced lines should be detected as hatching."""
        # Create 10 parallel vertical lines spaced 5 units apart
        nodes_list = []
        edges_list = []
        for i in range(10):
            x = 100 + i * 5.0
            idx = i * 2
            nodes_list.append([x, 100])
            nodes_list.append([x, 200])
            edges_list.append([idx, idx + 1])

        nodes = np.array(nodes_list, dtype=np.float64)
        edges = np.array(edges_list, dtype=np.int64)
        graph = _make_graph(nodes, edges)
        scores = score_hatching_detector(graph)
        assert len(scores) == 10
        # At least some should be flagged as hatching (score < 0.5)
        low_scores = np.sum(scores < 0.5)
        assert low_scores >= 3

    def test_single_edge_high_score(self) -> None:
        """A single isolated edge should not be flagged as hatching."""
        nodes = np.array([[0, 0], [0, 100]])
        edges = np.array([[0, 1]])
        graph = _make_graph(nodes, edges)
        scores = score_hatching_detector(graph)
        assert scores[0] == pytest.approx(1.0)

    def test_empty_graph(self) -> None:
        graph = _make_graph(np.empty((0, 2)), np.empty((0, 2)))
        scores = score_hatching_detector(graph)
        assert len(scores) == 0


# ---------------------------------------------------------------------------
# score_edge_length
# ---------------------------------------------------------------------------


class TestScoreEdgeLength:
    def test_very_short_edge_low_score(self) -> None:
        nodes = np.array([[0, 0], [0.5, 0]])
        edges = np.array([[0, 1]])
        graph = _make_graph(nodes, edges)
        scores = score_edge_length(graph)
        assert scores[0] < 0.3

    def test_medium_edge_higher_score(self) -> None:
        nodes = np.array([[0, 0], [100, 0]])
        edges = np.array([[0, 1]])
        graph = _make_graph(nodes, edges)
        scores = score_edge_length(graph)
        assert scores[0] > 0.5

    def test_empty_graph(self) -> None:
        graph = _make_graph(np.empty((0, 2)), np.empty((0, 2)))
        scores = score_edge_length(graph)
        assert len(scores) == 0


# ---------------------------------------------------------------------------
# compute_wall_confidence
# ---------------------------------------------------------------------------


class TestComputeWallConfidence:
    def test_returns_correct_shape(self) -> None:
        nodes = np.array([[0, 0], [100, 0], [100, 100]])
        edges = np.array([[0, 1], [1, 2]])
        graph = _make_graph(nodes, edges)
        scores = compute_wall_confidence(graph)
        assert scores.shape == (2,)

    def test_values_in_0_1(self) -> None:
        nodes = np.array([[0, 0], [100, 0], [100, 100], [0, 100]])
        edges = np.array([[0, 1], [1, 2], [2, 3], [3, 0]])
        graph = _make_graph(nodes, edges)
        scores = compute_wall_confidence(graph)
        assert np.all(scores >= 0.0)
        assert np.all(scores <= 1.0)

    def test_empty_graph(self) -> None:
        graph = _make_graph(np.empty((0, 2)), np.empty((0, 2)))
        scores = compute_wall_confidence(graph)
        assert len(scores) == 0


# ---------------------------------------------------------------------------
# apply_filters
# ---------------------------------------------------------------------------


class TestApplyFilters:
    def test_populates_confidence_wall(self) -> None:
        nodes = np.array([[0, 0], [100, 0], [100, 100]])
        edges = np.array([[0, 1], [1, 2]])
        graph = _make_graph(nodes, edges)
        # confidence_wall starts as zeros
        assert np.all(graph.confidence_wall == 0.0)
        result = apply_filters(graph)
        # After filtering, confidence_wall should be non-trivially populated
        assert result is graph  # mutates in place
        assert result.confidence_wall.shape == (2,)
        # For black, solid, width=1.0 lines, should have decent confidence
        assert np.all(result.confidence_wall > 0.0)

    def test_returns_same_graph(self) -> None:
        nodes = np.array([[0, 0], [100, 0]])
        edges = np.array([[0, 1]])
        graph = _make_graph(nodes, edges)
        result = apply_filters(graph)
        assert result is graph

    def test_empty_graph(self) -> None:
        graph = _make_graph(np.empty((0, 2)), np.empty((0, 2)))
        result = apply_filters(graph)
        assert len(result.confidence_wall) == 0
