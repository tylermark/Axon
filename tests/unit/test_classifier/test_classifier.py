"""Unit tests for the top-level wall classifier (CL-005)."""

from __future__ import annotations

import numpy as np
import pytest

from docs.interfaces.classified_wall_graph import (
    ClassifiedWallGraph,
    FireRating,
    WallClassification,
)
from docs.interfaces.graph_to_serializer import (
    FinalizedGraph,
    WallSegment,
    WallType,
)
from src.classifier.classifier import classify_wall_graph
from src.classifier.taxonomy import REVIEW_THRESHOLD, SignalWeights


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_multi_room_graph() -> FinalizedGraph:
    """Create a multi-room floor plan graph for classification testing.

    Layout:
    +--------+------+
    |        |      |
    |  Room1 | Room2|
    |        |      |
    +--------+------+

    7 walls, 6 nodes. Interior wall is thinner than perimeter.
    """
    nodes = np.array([
        [80, 100],   # 0: top-left
        [370, 100],  # 1: top-mid
        [530, 100],  # 2: top-right
        [530, 400],  # 3: bot-right
        [370, 400],  # 4: bot-mid
        [80, 400],   # 5: bot-left
    ], dtype=np.float64)

    edges = np.array([
        [0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 0],  # perimeter
        [1, 4],  # interior wall
    ], dtype=np.int64)

    segments = []
    for i, (s, e) in enumerate(edges):
        delta = nodes[e] - nodes[s]
        length = float(np.linalg.norm(delta))
        angle = float(np.arctan2(delta[1], delta[0]) % np.pi)
        # Interior wall (edge 6) is thinner.
        thickness = 3.0 if i == 6 else 8.0
        segments.append(WallSegment(
            edge_id=i,
            start_node=int(s),
            end_node=int(e),
            start_coord=nodes[s].copy(),
            end_coord=nodes[e].copy(),
            thickness=thickness,
            height=2700.0,
            wall_type=WallType.UNKNOWN,
            angle=angle,
            length=length,
            confidence=1.0,
        ))

    return FinalizedGraph(
        nodes=nodes,
        edges=edges,
        wall_segments=segments,
        openings=[],
        rooms=[],
        page_width=612.0,
        page_height=792.0,
        betti_0=1,
        betti_1=2,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestClassifyWallGraph:
    """Tests for the top-level classify_wall_graph function."""

    def test_returns_classified_wall_graph(self):
        graph = _make_multi_room_graph()
        result = classify_wall_graph(graph)
        assert isinstance(result, ClassifiedWallGraph)

    def test_classification_count_matches_segments(self):
        graph = _make_multi_room_graph()
        result = classify_wall_graph(graph)
        assert len(result.classifications) == len(graph.wall_segments)

    def test_edge_ids_match(self):
        graph = _make_multi_room_graph()
        result = classify_wall_graph(graph)
        for i, cls in enumerate(result.classifications):
            assert cls.edge_id == graph.wall_segments[i].edge_id

    def test_no_unknown_types(self):
        graph = _make_multi_room_graph()
        result = classify_wall_graph(graph)
        for cls in result.classifications:
            assert cls.wall_type != WallType.UNKNOWN

    def test_interior_wall_classified_differently(self):
        graph = _make_multi_room_graph()
        result = classify_wall_graph(graph)
        # Edge 6 is the thin interior wall.
        interior_cls = result.classifications[6]
        # Thin interior wall should lean toward partition.
        assert interior_cls.wall_type in (WallType.PARTITION, WallType.LOAD_BEARING)

    def test_perimeter_walls_detected(self):
        graph = _make_multi_room_graph()
        result = classify_wall_graph(graph)
        # At least some perimeter edges should be identified.
        assert len(result.perimeter_edge_ids) > 0

    def test_perimeter_classification_is_perimeter_flag(self):
        graph = _make_multi_room_graph()
        result = classify_wall_graph(graph)
        for cls in result.classifications:
            if cls.edge_id in result.perimeter_edge_ids:
                assert cls.is_perimeter is True

    def test_interior_not_flagged_as_perimeter(self):
        graph = _make_multi_room_graph()
        result = classify_wall_graph(graph)
        # Edge 6 (interior) should not be perimeter.
        interior_cls = result.classifications[6]
        assert interior_cls.is_perimeter is False

    def test_confidence_in_valid_range(self):
        graph = _make_multi_room_graph()
        result = classify_wall_graph(graph)
        for cls in result.classifications:
            assert 0.0 <= cls.confidence <= 1.0

    def test_signals_dict_populated(self):
        graph = _make_multi_room_graph()
        result = classify_wall_graph(graph)
        for cls in result.classifications:
            assert len(cls.signals) > 0

    def test_classification_summary_counts(self):
        graph = _make_multi_room_graph()
        result = classify_wall_graph(graph)
        total = sum(result.classification_summary.values())
        assert total == len(graph.wall_segments)

    def test_flagged_walls_below_threshold(self):
        graph = _make_multi_room_graph()
        result = classify_wall_graph(graph, review_threshold=0.99)
        # With a very high threshold, most walls should be flagged.
        assert len(result.walls_flagged_for_review) > 0

    def test_low_threshold_flags_nothing(self):
        graph = _make_multi_room_graph()
        result = classify_wall_graph(graph, review_threshold=0.0)
        assert len(result.walls_flagged_for_review) == 0

    def test_fire_rating_with_color_data(self):
        graph = _make_multi_room_graph()
        # Make edge 2 red (fire-rated).
        n_edges = len(graph.wall_segments)
        stroke_colors = np.zeros((n_edges, 4), dtype=np.float64)
        stroke_colors[:, 3] = 1.0  # all opaque black
        stroke_colors[2] = [0.9, 0.1, 0.1, 1.0]  # edge 2 is red

        result = classify_wall_graph(graph, stroke_colors=stroke_colors)
        assert result.classifications[2].fire_rating != FireRating.NONE

    def test_fire_rating_without_color_data(self):
        graph = _make_multi_room_graph()
        result = classify_wall_graph(graph)
        for cls in result.classifications:
            assert cls.fire_rating == FireRating.NONE

    def test_custom_weights(self):
        graph = _make_multi_room_graph()
        # Heavy thickness weight should make thin walls strongly partition.
        weights = SignalWeights(
            thickness=0.8, position=0.05, adjacency=0.05,
            color=0.05, length=0.05,
        )
        result = classify_wall_graph(graph, weights=weights)
        interior = result.classifications[6]
        assert interior.wall_type == WallType.PARTITION

    def test_graph_reference_preserved(self):
        graph = _make_multi_room_graph()
        result = classify_wall_graph(graph)
        assert result.graph is graph

    def test_empty_graph(self):
        graph = FinalizedGraph(
            nodes=np.empty((0, 2), dtype=np.float64),
            edges=np.empty((0, 2), dtype=np.int64),
            wall_segments=[],
            openings=[],
            rooms=[],
            page_width=612.0,
            page_height=792.0,
        )
        result = classify_wall_graph(graph)
        assert len(result.classifications) == 0
        assert len(result.walls_flagged_for_review) == 0
