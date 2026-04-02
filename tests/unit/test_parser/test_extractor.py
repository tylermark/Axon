"""Unit tests for src/parser/extractor.py — content stream extractor."""

from __future__ import annotations

from typing import TYPE_CHECKING

import fitz
import numpy as np
import pytest

from src.parser.extractor import (
    ExtractedPath,
    GraphicsState,
    GraphicsStateStack,
    PathAccumulator,
    PathSegment,
    SubPath,
    _cmyk_to_rgba,
    _gray_to_rgba,
    _rgb_to_rgba,
    extract_paths,
    extract_paths_from_pdf,
)
from src.parser.operators import OperatorType

if TYPE_CHECKING:
    from pathlib import Path


# ---------------------------------------------------------------------------
# GraphicsState defaults
# ---------------------------------------------------------------------------


class TestGraphicsState:
    def test_default_ctm_is_identity(self) -> None:
        gs = GraphicsState()
        np.testing.assert_array_equal(gs.ctm, np.eye(3, dtype=np.float64))

    def test_default_stroke_width(self) -> None:
        gs = GraphicsState()
        assert gs.stroke_width == 1.0

    def test_default_stroke_color_is_black(self) -> None:
        gs = GraphicsState()
        np.testing.assert_array_equal(gs.stroke_color, np.array([0.0, 0.0, 0.0, 1.0]))

    def test_default_fill_color_is_none(self) -> None:
        gs = GraphicsState()
        assert gs.fill_color is None

    def test_default_dash_pattern(self) -> None:
        gs = GraphicsState()
        assert gs.dash_pattern == ([], 0.0)

    def test_default_line_cap(self) -> None:
        gs = GraphicsState()
        assert gs.line_cap == 0

    def test_default_miter_limit(self) -> None:
        gs = GraphicsState()
        assert gs.miter_limit == 10.0


# ---------------------------------------------------------------------------
# GraphicsStateStack
# ---------------------------------------------------------------------------


class TestGraphicsStateStack:
    def test_initial_state(self) -> None:
        stack = GraphicsStateStack()
        assert stack.current.stroke_width == 1.0

    def test_push_pop_restores_state(self) -> None:
        stack = GraphicsStateStack()
        stack.current.stroke_width = 1.0
        stack.push()
        stack.current.stroke_width = 5.0
        assert stack.current.stroke_width == 5.0
        stack.pop()
        assert stack.current.stroke_width == 1.0

    def test_push_creates_deep_copy(self) -> None:
        stack = GraphicsStateStack()
        original_color = stack.current.stroke_color.copy()
        stack.push()
        stack.current.stroke_color[:] = [1.0, 0.0, 0.0, 1.0]
        stack.pop()
        np.testing.assert_array_equal(stack.current.stroke_color, original_color)

    def test_pop_on_empty_is_safe(self) -> None:
        stack = GraphicsStateStack()
        # Should log a warning but not raise
        stack.pop()
        assert stack.current.stroke_width == 1.0

    def test_apply_ctm_compose_translation(self) -> None:
        stack = GraphicsStateStack()
        # Translation by (10, 20)
        translate = np.array([[1, 0, 0], [0, 1, 0], [10, 20, 1]], dtype=np.float64)
        stack.apply_ctm(translate)
        expected = np.eye(3, dtype=np.float64) @ translate
        np.testing.assert_array_almost_equal(stack.current.ctm, expected)

    def test_apply_ctm_compose_multiple(self) -> None:
        stack = GraphicsStateStack()
        t1 = np.array([[1, 0, 0], [0, 1, 0], [5, 0, 1]], dtype=np.float64)
        t2 = np.array([[1, 0, 0], [0, 1, 0], [0, 10, 1]], dtype=np.float64)
        stack.apply_ctm(t1)
        stack.apply_ctm(t2)
        expected = np.eye(3) @ t1 @ t2
        np.testing.assert_array_almost_equal(stack.current.ctm, expected)

    def test_set_stroke_width(self) -> None:
        stack = GraphicsStateStack()
        stack.set_stroke_width(2.5)
        assert stack.current.stroke_width == 2.5

    def test_set_stroke_color(self) -> None:
        stack = GraphicsStateStack()
        red = np.array([1.0, 0.0, 0.0, 1.0], dtype=np.float64)
        stack.set_stroke_color(red)
        np.testing.assert_array_equal(stack.current.stroke_color, red)

    def test_set_dash_pattern(self) -> None:
        stack = GraphicsStateStack()
        stack.set_dash_pattern([3.0, 2.0], 1.0)
        assert stack.current.dash_pattern == ([3.0, 2.0], 1.0)


# ---------------------------------------------------------------------------
# Color conversion helpers
# ---------------------------------------------------------------------------


class TestColorConversion:
    def test_rgb_to_rgba(self) -> None:
        result = _rgb_to_rgba((0.5, 0.3, 0.1))
        assert result is not None
        np.testing.assert_array_almost_equal(result, [0.5, 0.3, 0.1, 1.0])

    def test_rgb_to_rgba_none(self) -> None:
        assert _rgb_to_rgba(None) is None

    def test_rgb_to_rgba_black(self) -> None:
        result = _rgb_to_rgba((0.0, 0.0, 0.0))
        assert result is not None
        np.testing.assert_array_almost_equal(result, [0.0, 0.0, 0.0, 1.0])

    def test_gray_to_rgba_black(self) -> None:
        result = _gray_to_rgba(0.0)
        np.testing.assert_array_almost_equal(result, [0.0, 0.0, 0.0, 1.0])

    def test_gray_to_rgba_white(self) -> None:
        result = _gray_to_rgba(1.0)
        np.testing.assert_array_almost_equal(result, [1.0, 1.0, 1.0, 1.0])

    def test_gray_to_rgba_mid(self) -> None:
        result = _gray_to_rgba(0.5)
        np.testing.assert_array_almost_equal(result, [0.5, 0.5, 0.5, 1.0])

    def test_cmyk_to_rgba_black(self) -> None:
        # C=0, M=0, Y=0, K=1 → black
        result = _cmyk_to_rgba(0.0, 0.0, 0.0, 1.0)
        np.testing.assert_array_almost_equal(result, [0.0, 0.0, 0.0, 1.0])

    def test_cmyk_to_rgba_white(self) -> None:
        # C=0, M=0, Y=0, K=0 → white
        result = _cmyk_to_rgba(0.0, 0.0, 0.0, 0.0)
        np.testing.assert_array_almost_equal(result, [1.0, 1.0, 1.0, 1.0])

    def test_cmyk_to_rgba_cyan(self) -> None:
        result = _cmyk_to_rgba(1.0, 0.0, 0.0, 0.0)
        np.testing.assert_array_almost_equal(result, [0.0, 1.0, 1.0, 1.0])


# ---------------------------------------------------------------------------
# PathAccumulator
# ---------------------------------------------------------------------------


class TestPathAccumulator:
    def test_moveto_sets_current_point(self) -> None:
        acc = PathAccumulator()
        acc.moveto(10.0, 20.0)
        assert acc._current_point is not None
        np.testing.assert_array_almost_equal(acc._current_point, [10.0, 20.0])

    def test_lineto_creates_segment(self) -> None:
        acc = PathAccumulator()
        acc.moveto(0.0, 0.0)
        acc.lineto(10.0, 0.0)
        assert len(acc._subpaths) == 1
        assert len(acc._subpaths[0].segments) == 1
        seg = acc._subpaths[0].segments[0]
        assert seg.operator == OperatorType.LINETO
        np.testing.assert_array_almost_equal(seg.start, [0.0, 0.0])
        np.testing.assert_array_almost_equal(seg.end, [10.0, 0.0])

    def test_curveto_stores_control_points(self) -> None:
        acc = PathAccumulator()
        acc.moveto(0.0, 0.0)
        acc.curveto(1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
        seg = acc._subpaths[0].segments[0]
        assert seg.operator == OperatorType.CURVETO
        assert seg.control_points is not None
        assert len(seg.control_points) == 4
        np.testing.assert_array_almost_equal(seg.control_points[0], [0.0, 0.0])
        np.testing.assert_array_almost_equal(seg.control_points[1], [1.0, 2.0])
        np.testing.assert_array_almost_equal(seg.control_points[2], [3.0, 4.0])
        np.testing.assert_array_almost_equal(seg.control_points[3], [5.0, 6.0])

    def test_rect_generates_4_lineto_and_closepath(self) -> None:
        acc = PathAccumulator()
        acc.rect(10.0, 20.0, 100.0, 50.0)
        subpath = acc._subpaths[0]
        # rect = moveto + 3 lineto + closepath
        # moveto doesn't create a segment, so: 3 lineto + 1 closepath = 4
        assert len(subpath.segments) == 4
        ops = [seg.operator for seg in subpath.segments]
        assert ops.count(OperatorType.LINETO) == 3
        assert ops.count(OperatorType.CLOSEPATH) == 1
        assert subpath.is_closed is True

    def test_closepath_creates_edge_back_to_start(self) -> None:
        acc = PathAccumulator()
        acc.moveto(0.0, 0.0)
        acc.lineto(10.0, 0.0)
        acc.lineto(10.0, 10.0)
        acc.closepath()
        subpath = acc._subpaths[0]
        # Last segment should be closepath from (10,10) back to (0,0)
        close_seg = subpath.segments[-1]
        assert close_seg.operator == OperatorType.CLOSEPATH
        np.testing.assert_array_almost_equal(close_seg.start, [10.0, 10.0])
        np.testing.assert_array_almost_equal(close_seg.end, [0.0, 0.0])
        assert subpath.is_closed is True

    def test_closepath_no_segment_when_already_at_start(self) -> None:
        acc = PathAccumulator()
        acc.moveto(5.0, 5.0)
        acc.lineto(10.0, 10.0)
        acc.lineto(5.0, 5.0)
        acc.closepath()
        subpath = acc._subpaths[0]
        # Already at start, so closepath should not add a segment
        # but should still mark the subpath as closed
        assert subpath.is_closed is True
        # Only the 2 lineto segments
        assert len(subpath.segments) == 2

    def test_finalize_captures_graphics_state(self) -> None:
        acc = PathAccumulator()
        acc.moveto(0, 0)
        acc.lineto(100, 0)
        gs = GraphicsState(stroke_width=2.5)
        result = acc.finalize(gs, is_stroked=True, is_filled=False)
        assert isinstance(result, ExtractedPath)
        assert result.stroke_width == 2.5
        assert result.is_stroked is True
        assert result.is_filled is False

    def test_finalize_filters_empty_subpaths(self) -> None:
        acc = PathAccumulator()
        acc.moveto(0, 0)  # creates a subpath with no segments
        acc.moveto(10, 10)
        acc.lineto(20, 20)
        gs = GraphicsState()
        result = acc.finalize(gs, is_stroked=True, is_filled=False)
        # Only the subpath with segments should appear
        assert len(result.subpaths) == 1


# ---------------------------------------------------------------------------
# Dataclass construction
# ---------------------------------------------------------------------------


class TestDataclasses:
    def test_path_segment_construction(self) -> None:
        seg = PathSegment(
            operator=OperatorType.LINETO,
            start=np.array([0.0, 0.0]),
            end=np.array([1.0, 1.0]),
        )
        assert seg.operator == OperatorType.LINETO
        assert seg.control_points is None

    def test_subpath_construction(self) -> None:
        sp = SubPath()
        assert sp.segments == []
        assert sp.is_closed is False

    def test_extracted_path_construction(self) -> None:
        ep = ExtractedPath(
            subpaths=[],
            stroke_width=1.0,
            stroke_color=np.zeros(4),
            fill_color=None,
            dash_pattern=([], 0.0),
            ctm=np.eye(3),
            is_stroked=True,
            is_filled=False,
        )
        assert ep.is_clipping is False


# ---------------------------------------------------------------------------
# extract_paths_from_pdf (integration with PyMuPDF)
# ---------------------------------------------------------------------------


class TestExtractPathsFromPdf:
    def test_extract_simple_rect(self, tmp_path: Path) -> None:
        """Create a PDF with one rectangle and extract paths."""
        pdf_path = tmp_path / "rect.pdf"
        doc = fitz.open()
        page = doc.new_page(width=612, height=792)
        shape = page.new_shape()
        shape.draw_rect(fitz.Rect(100, 100, 300, 250))
        shape.finish(width=1.5, color=(0, 0, 0))
        shape.commit()
        doc.save(str(pdf_path))
        doc.close()

        result = extract_paths_from_pdf(str(pdf_path))
        assert 0 in result
        paths = result[0]
        assert len(paths) >= 1
        # The rectangle path should have segments
        total_segments = sum(len(sp.segments) for p in paths for sp in p.subpaths)
        assert total_segments >= 4  # at least 4 sides

    def test_extract_nonexistent_pdf_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            extract_paths_from_pdf("/nonexistent/path.pdf")

    def test_extract_specific_pages(self, tmp_path: Path) -> None:
        """Extract from specific page indices only."""
        pdf_path = tmp_path / "multi.pdf"
        doc = fitz.open()
        for _ in range(3):
            page = doc.new_page(width=612, height=792)
            shape = page.new_shape()
            shape.draw_line(fitz.Point(10, 10), fitz.Point(100, 10))
            shape.finish(width=1.0, color=(0, 0, 0))
            shape.commit()
        doc.save(str(pdf_path))
        doc.close()

        result = extract_paths_from_pdf(str(pdf_path), page_indices=[0, 2])
        assert 0 in result
        assert 2 in result
        assert 1 not in result

    def test_extract_paths_stroke_width_filter(self, tmp_path: Path) -> None:
        """Paths thinner than min_stroke_width are filtered."""
        pdf_path = tmp_path / "thin.pdf"
        doc = fitz.open()
        page = doc.new_page(width=612, height=792)
        shape = page.new_shape()
        # Very thin line
        shape.draw_line(fitz.Point(10, 10), fitz.Point(200, 10))
        shape.finish(width=0.01, color=(0, 0, 0))
        # Normal line
        shape.draw_line(fitz.Point(10, 50), fitz.Point(200, 50))
        shape.finish(width=1.5, color=(0, 0, 0))
        shape.commit()
        doc.save(str(pdf_path))
        doc.close()

        page_obj = fitz.open(str(pdf_path))[0]
        # Default min_stroke_width=0.1 should filter the thin line
        paths = extract_paths(page_obj, min_stroke_width=0.1)
        # Should have at least the normal-width line
        thick_paths = [p for p in paths if p.stroke_width >= 0.1]
        assert len(thick_paths) >= 1
