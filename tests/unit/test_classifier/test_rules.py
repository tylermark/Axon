"""Unit tests for wall classification rules (CL-002, CL-003, CL-004)."""

from __future__ import annotations

import numpy as np
import pytest

from docs.interfaces.classified_wall_graph import FireRating
from docs.interfaces.graph_to_serializer import (
    FinalizedGraph,
    WallSegment,
    WallType,
)
from src.classifier.rules import (
    detect_fire_rating,
    identify_perimeter_edges,
    score_by_adjacency,
    score_by_length,
    score_by_position,
    score_by_thickness,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_segment(
    edge_id: int = 0,
    thickness: float = 6.0,
    length: float = 100.0,
    angle: float = 0.0,
    start_node: int = 0,
    end_node: int = 1,
) -> WallSegment:
    """Create a WallSegment with given properties."""
    return WallSegment(
        edge_id=edge_id,
        start_node=start_node,
        end_node=end_node,
        start_coord=np.array([0.0, 0.0]),
        end_coord=np.array([length, 0.0]),
        thickness=thickness,
        height=2700.0,
        wall_type=WallType.UNKNOWN,
        angle=angle,
        length=length,
        confidence=1.0,
    )


def _make_rect_graph(
    width: float = 450.0,
    height: float = 500.0,
) -> FinalizedGraph:
    """Create a rectangular room FinalizedGraph (4 walls, 4 nodes)."""
    nodes = np.array([
        [80, 100], [530, 100], [530, 600], [80, 600],
    ], dtype=np.float64)

    edges = np.array([
        [0, 1], [1, 2], [2, 3], [3, 0],
    ], dtype=np.int64)

    segments = []
    for i, (s, e) in enumerate(edges):
        delta = nodes[e] - nodes[s]
        length = float(np.linalg.norm(delta))
        angle = float(np.arctan2(delta[1], delta[0]) % np.pi)
        segments.append(WallSegment(
            edge_id=i,
            start_node=int(s),
            end_node=int(e),
            start_coord=nodes[s].copy(),
            end_coord=nodes[e].copy(),
            thickness=6.0,
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
        betti_1=1,
    )


# ---------------------------------------------------------------------------
# CL-002: Thickness-based classification
# ---------------------------------------------------------------------------


class TestScoreByThickness:
    """Tests for thickness-based classification rules."""

    def test_thin_wall_favors_partition(self):
        seg = _make_segment(thickness=2.0)
        scores = score_by_thickness(seg)
        assert scores[WallType.PARTITION] > scores[WallType.LOAD_BEARING]
        assert scores[WallType.PARTITION] > scores[WallType.EXTERIOR]

    def test_thick_wall_favors_structural(self):
        seg = _make_segment(thickness=10.0)
        scores = score_by_thickness(seg)
        structural = scores[WallType.LOAD_BEARING] + scores[WallType.EXTERIOR]
        assert structural > scores[WallType.PARTITION]

    def test_medium_wall_is_ambiguous(self):
        seg = _make_segment(thickness=6.0)
        scores = score_by_thickness(seg)
        # No single type should dominate overwhelmingly.
        max_score = max(scores.values())
        assert max_score < 0.8

    def test_scores_sum_to_one(self):
        for thickness in [1.0, 4.0, 6.0, 10.0, 15.0]:
            seg = _make_segment(thickness=thickness)
            scores = score_by_thickness(seg)
            total = sum(scores.values())
            assert abs(total - 1.0) < 0.01, f"Scores sum to {total} for thickness={thickness}"

    def test_all_types_present(self):
        seg = _make_segment(thickness=6.0)
        scores = score_by_thickness(seg)
        for wt in [WallType.LOAD_BEARING, WallType.PARTITION, WallType.EXTERIOR,
                    WallType.SHEAR, WallType.CURTAIN]:
            assert wt in scores


# ---------------------------------------------------------------------------
# CL-003: Context-based classification
# ---------------------------------------------------------------------------


class TestScoreByPosition:
    """Tests for position-based classification."""

    def test_perimeter_favors_exterior(self):
        seg = _make_segment()
        scores = score_by_position(seg, is_perimeter=True)
        assert scores[WallType.EXTERIOR] > scores[WallType.PARTITION]

    def test_interior_favors_partition(self):
        seg = _make_segment()
        scores = score_by_position(seg, is_perimeter=False)
        assert scores[WallType.PARTITION] > scores[WallType.EXTERIOR]

    def test_interior_excludes_exterior(self):
        seg = _make_segment()
        scores = score_by_position(seg, is_perimeter=False)
        assert scores[WallType.EXTERIOR] == 0.0


class TestScoreByAdjacency:
    """Tests for adjacency-based classification."""

    def test_high_degree_favors_structural(self):
        graph = _make_rect_graph()
        # Corner nodes have degree 2 in a rectangle.
        seg = graph.wall_segments[0]
        scores = score_by_adjacency(seg, graph)
        assert WallType.LOAD_BEARING in scores

    def test_returns_all_types(self):
        graph = _make_rect_graph()
        seg = graph.wall_segments[0]
        scores = score_by_adjacency(seg, graph)
        for wt in [WallType.LOAD_BEARING, WallType.PARTITION, WallType.EXTERIOR]:
            assert wt in scores


class TestScoreByLength:
    """Tests for length-based classification."""

    def test_long_wall_favors_structural(self):
        graph = _make_rect_graph()
        # Top wall spans most of page width.
        seg = graph.wall_segments[0]
        scores = score_by_length(seg, graph)
        structural = scores[WallType.LOAD_BEARING] + scores[WallType.EXTERIOR]
        assert structural >= scores[WallType.PARTITION]

    def test_short_wall_favors_partition(self):
        graph = _make_rect_graph()
        short_seg = _make_segment(length=30.0)
        scores = score_by_length(short_seg, graph)
        assert scores[WallType.PARTITION] > scores[WallType.LOAD_BEARING]


class TestIdentifyPerimeterEdges:
    """Tests for perimeter edge detection."""

    def test_rectangle_all_perimeter(self):
        graph = _make_rect_graph()
        perimeter = identify_perimeter_edges(graph)
        # All 4 walls of a rectangle should be on the perimeter.
        assert len(perimeter) == 4

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
        perimeter = identify_perimeter_edges(graph)
        assert len(perimeter) == 0

    def test_two_nodes_insufficient_for_hull(self):
        nodes = np.array([[0, 0], [100, 0]], dtype=np.float64)
        graph = FinalizedGraph(
            nodes=nodes,
            edges=np.array([[0, 1]], dtype=np.int64),
            wall_segments=[_make_segment()],
            openings=[],
            rooms=[],
            page_width=612.0,
            page_height=792.0,
        )
        perimeter = identify_perimeter_edges(graph)
        assert len(perimeter) == 0


# ---------------------------------------------------------------------------
# CL-004: Fire rating detection
# ---------------------------------------------------------------------------


class TestDetectFireRating:
    """Tests for fire rating detection from color signals."""

    def test_no_color_data_returns_none(self):
        seg = _make_segment()
        rating, conf = detect_fire_rating(seg)
        assert rating == FireRating.NONE
        assert conf > 0.5

    def test_red_stroke_detects_fire_rated(self):
        seg = _make_segment(thickness=8.0)
        stroke_colors = np.array([[0.9, 0.1, 0.1, 1.0]], dtype=np.float64)
        rating, conf = detect_fire_rating(
            seg, stroke_colors=stroke_colors, edge_index=0,
        )
        assert rating != FireRating.NONE

    def test_black_stroke_not_fire_rated(self):
        seg = _make_segment()
        stroke_colors = np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float64)
        rating, conf = detect_fire_rating(
            seg, stroke_colors=stroke_colors, edge_index=0,
        )
        assert rating == FireRating.NONE

    def test_thick_fire_wall_gets_higher_rating(self):
        seg_thin = _make_segment(thickness=5.0)
        seg_thick = _make_segment(thickness=12.0)
        red = np.array([[0.9, 0.1, 0.1, 1.0]], dtype=np.float64)

        rating_thin, _ = detect_fire_rating(seg_thin, stroke_colors=red, edge_index=0)
        rating_thick, _ = detect_fire_rating(seg_thick, stroke_colors=red, edge_index=0)

        # Thick fire wall should get 2-hour, thin gets 1-hour.
        assert rating_thick == FireRating.HOUR_2
        assert rating_thin == FireRating.HOUR_1

    def test_red_fill_detects_fire_rated(self):
        seg = _make_segment()
        fill_colors = np.array([[0.8, 0.1, 0.2, 1.0]], dtype=np.float64)
        rating, _ = detect_fire_rating(
            seg, fill_colors=fill_colors, edge_index=0,
        )
        assert rating != FireRating.NONE
