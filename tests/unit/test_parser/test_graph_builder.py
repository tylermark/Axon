"""Unit tests for src/parser/graph_builder.py — Bézier sampling, dedup, graph builder."""

from __future__ import annotations

import numpy as np

from src.parser.extractor import ExtractedPath, GraphicsState, PathAccumulator
from src.parser.graph_builder import (
    build_raw_graph,
    deduplicate_vertices,
    sample_bezier,
)
from src.parser.operators import OperatorType

# ---------------------------------------------------------------------------
# sample_bezier
# ---------------------------------------------------------------------------


class TestSampleBezier:
    def test_straight_line_collinear(self) -> None:
        """Collinear control points should return [p0, p3]."""
        p0 = np.array([0.0, 0.0])
        p1 = np.array([1.0, 0.0])
        p2 = np.array([2.0, 0.0])
        p3 = np.array([3.0, 0.0])
        result = sample_bezier(p0, p1, p2, p3)
        assert len(result) == 2
        np.testing.assert_array_almost_equal(result[0], p0)
        np.testing.assert_array_almost_equal(result[1], p3)

    def test_resolution_1_returns_2_points(self) -> None:
        """resolution=1 should return [p0, p3] for non-degenerate curve."""
        p0 = np.array([0.0, 0.0])
        p1 = np.array([0.0, 10.0])
        p2 = np.array([10.0, 10.0])
        p3 = np.array([10.0, 0.0])
        result = sample_bezier(p0, p1, p2, p3, resolution=1)
        assert len(result) == 2
        np.testing.assert_array_almost_equal(result[0], p0)
        np.testing.assert_array_almost_equal(result[-1], p3)

    def test_resolution_4_returns_5_points(self) -> None:
        p0 = np.array([0.0, 0.0])
        p1 = np.array([0.0, 10.0])
        p2 = np.array([10.0, 10.0])
        p3 = np.array([10.0, 0.0])
        result = sample_bezier(p0, p1, p2, p3, resolution=4)
        assert len(result) == 5

    def test_degenerate_all_identical(self) -> None:
        """All control points identical should return [p0, p3]."""
        p = np.array([5.0, 5.0])
        result = sample_bezier(p.copy(), p.copy(), p.copy(), p.copy())
        assert len(result) == 2
        np.testing.assert_array_almost_equal(result[0], p)
        np.testing.assert_array_almost_equal(result[1], p)

    def test_endpoints_match(self) -> None:
        """First and last sampled points should match p0 and p3."""
        p0 = np.array([1.0, 2.0])
        p1 = np.array([3.0, 8.0])
        p2 = np.array([7.0, 8.0])
        p3 = np.array([9.0, 2.0])
        result = sample_bezier(p0, p1, p2, p3, resolution=8)
        np.testing.assert_array_almost_equal(result[0], p0)
        np.testing.assert_array_almost_equal(result[-1], p3)

    def test_quarter_circle_off_axis(self) -> None:
        """Quarter-circle approximation: intermediate points should be off-axis."""
        p0 = np.array([0.0, 1.0])
        p1 = np.array([0.0, 1.0 - 0.5523])  # Standard circle Bézier approx
        p2 = np.array([1.0 - 0.5523, 0.0])
        p3 = np.array([1.0, 0.0])
        result = sample_bezier(p0, p1, p2, p3, resolution=8)
        assert len(result) == 9
        # Mid-points should not lie on the axis lines
        mid = result[4]
        assert mid[0] > 0.0
        assert mid[1] > 0.0

    def test_default_resolution_is_8(self) -> None:
        p0 = np.array([0.0, 0.0])
        p1 = np.array([0.0, 10.0])
        p2 = np.array([10.0, 10.0])
        p3 = np.array([10.0, 0.0])
        result = sample_bezier(p0, p1, p2, p3)
        assert len(result) == 9  # 8 + 1


# ---------------------------------------------------------------------------
# deduplicate_vertices
# ---------------------------------------------------------------------------


class TestDeduplicateVertices:
    def test_no_duplicates(self) -> None:
        verts = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0]], dtype=np.float64)
        deduped, mapping = deduplicate_vertices(verts, tolerance=0.5)
        assert len(deduped) == 3
        assert len(mapping) == 3
        # Each vertex maps to itself (in some order)
        assert len(np.unique(mapping)) == 3

    def test_exact_duplicates_merged(self) -> None:
        verts = np.array(
            [[0.0, 0.0], [10.0, 0.0], [0.0, 0.0], [10.0, 0.0]],
            dtype=np.float64,
        )
        deduped, mapping = deduplicate_vertices(verts, tolerance=0.5)
        assert len(deduped) == 2
        # Duplicate pairs should map to the same index
        assert mapping[0] == mapping[2]
        assert mapping[1] == mapping[3]

    def test_near_duplicates_within_tolerance(self) -> None:
        verts = np.array(
            [[0.0, 0.0], [0.3, 0.0], [10.0, 10.0]],
            dtype=np.float64,
        )
        deduped, mapping = deduplicate_vertices(verts, tolerance=0.5)
        # First two are within 0.5 tolerance
        assert len(deduped) == 2
        assert mapping[0] == mapping[1]
        assert mapping[2] != mapping[0]

    def test_near_duplicates_merged_to_centroid(self) -> None:
        verts = np.array([[0.0, 0.0], [0.4, 0.0]], dtype=np.float64)
        deduped, _mapping = deduplicate_vertices(verts, tolerance=0.5)
        assert len(deduped) == 1
        # Centroid should be (0.2, 0.0)
        np.testing.assert_array_almost_equal(deduped[0], [0.2, 0.0])

    def test_points_outside_tolerance_kept_separate(self) -> None:
        verts = np.array(
            [[0.0, 0.0], [1.0, 0.0], [10.0, 10.0]],
            dtype=np.float64,
        )
        deduped, mapping = deduplicate_vertices(verts, tolerance=0.5)
        assert len(deduped) == 3
        assert len(np.unique(mapping)) == 3

    def test_empty_input(self) -> None:
        verts = np.empty((0, 2), dtype=np.float64)
        deduped, mapping = deduplicate_vertices(verts, tolerance=0.5)
        assert deduped.shape == (0, 2)
        assert len(mapping) == 0

    def test_zero_tolerance(self) -> None:
        verts = np.array([[0.0, 0.0], [0.0, 0.0], [1.0, 1.0]], dtype=np.float64)
        deduped, _mapping = deduplicate_vertices(verts, tolerance=0.0)
        # With tolerance=0, no merging happens (tolerance <=0 returns copy)
        assert len(deduped) == 3

    def test_mapping_shape(self) -> None:
        verts = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]], dtype=np.float64)
        _deduped, mapping = deduplicate_vertices(verts, tolerance=0.5)
        assert mapping.shape == (3,)
        assert mapping.dtype == np.int64


# ---------------------------------------------------------------------------
# build_raw_graph
# ---------------------------------------------------------------------------


def _make_extracted_path(
    segments_data: list[tuple[str, list[float], list[float]]],
    stroke_width: float = 1.0,
    stroke_color: tuple[float, ...] = (0.0, 0.0, 0.0, 1.0),
) -> ExtractedPath:
    """Helper to build an ExtractedPath from segment specs.

    segments_data: list of (operator_str, start_xy, end_xy)
    """
    acc = PathAccumulator()
    for op_str, start, end in segments_data:
        if acc._current_point is None or not np.allclose(acc._current_point, start, atol=1e-10):
            acc.moveto(start[0], start[1])
        if op_str == "lineto":
            acc.lineto(end[0], end[1])
        elif op_str == "closepath":
            acc.closepath()

    gs = GraphicsState(
        stroke_width=stroke_width,
        stroke_color=np.array(stroke_color, dtype=np.float64),
    )
    return acc.finalize(gs, is_stroked=True, is_filled=False)


class TestBuildRawGraph:
    def test_simple_l_shape(self) -> None:
        """Two line segments forming an L shape."""
        path = _make_extracted_path(
            [
                ("lineto", [0.0, 0.0], [100.0, 0.0]),
                ("lineto", [100.0, 0.0], [100.0, 100.0]),
            ]
        )
        graph = build_raw_graph([path])
        assert graph.nodes.shape[1] == 2
        assert graph.edges.shape[1] == 2
        assert len(graph.edges) == 2
        assert len(graph.operator_types) == 2
        assert len(graph.stroke_widths) == 2

    def test_empty_input(self) -> None:
        graph = build_raw_graph([])
        assert graph.nodes.shape == (0, 2)
        assert graph.edges.shape == (0, 2)
        assert len(graph.operator_types) == 0
        assert graph.confidence_wall.shape == (0,)

    def test_bezier_curve_expansion(self) -> None:
        """Bézier curves should be expanded into multiple edges."""
        acc = PathAccumulator()
        acc.moveto(0.0, 0.0)
        acc.curveto(0.0, 50.0, 50.0, 50.0, 50.0, 0.0)
        gs = GraphicsState(stroke_width=1.0)
        path = acc.finalize(gs, is_stroked=True, is_filled=False)
        graph = build_raw_graph([path])
        # Default resolution=8, so a non-degenerate curve → 8 edges (or 2 if collinear)
        assert len(graph.edges) >= 2
        # All edges should be CURVETO type
        for op in graph.operator_types:
            assert op == OperatorType.CURVETO

    def test_vertex_deduplication(self) -> None:
        """Overlapping endpoints should be deduplicated."""
        # Two line segments sharing an endpoint
        path1 = _make_extracted_path([("lineto", [0.0, 0.0], [50.0, 0.0])])
        path2 = _make_extracted_path([("lineto", [50.0, 0.0], [50.0, 50.0])])
        graph = build_raw_graph([path1, path2])
        # Without dedup: 4 vertices, with dedup: 3 (shared midpoint)
        assert graph.nodes.shape[0] == 3

    def test_metadata_consistency(self) -> None:
        """All per-edge arrays should have consistent lengths."""
        path = _make_extracted_path(
            [
                ("lineto", [0.0, 0.0], [100.0, 0.0]),
                ("lineto", [100.0, 0.0], [100.0, 100.0]),
                ("lineto", [100.0, 100.0], [0.0, 100.0]),
            ]
        )
        graph = build_raw_graph([path])
        ne = len(graph.edges)
        assert len(graph.operator_types) == ne
        assert len(graph.stroke_widths) == ne
        assert len(graph.stroke_colors) == ne
        assert len(graph.dash_patterns) == ne
        assert len(graph.edge_to_path) == ne
        assert len(graph.bezier_controls) == ne
        assert len(graph.confidence_wall) == ne

    def test_page_metadata_preserved(self) -> None:
        path = _make_extracted_path([("lineto", [0.0, 0.0], [100.0, 0.0])])
        graph = build_raw_graph(
            [path],
            page_width=595.0,
            page_height=842.0,
            page_index=2,
            source_path="/test.pdf",
        )
        assert graph.page_width == 595.0
        assert graph.page_height == 842.0
        assert graph.page_index == 2
        assert graph.source_path == "/test.pdf"

    def test_raw_graph_nodes_dtype(self) -> None:
        path = _make_extracted_path([("lineto", [0.0, 0.0], [100.0, 0.0])])
        graph = build_raw_graph([path])
        assert graph.nodes.dtype == np.float64
        assert graph.edges.dtype == np.int64
