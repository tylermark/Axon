"""Integration test: full parser pipeline PDF → RawGraph with filters.

Q-002: End-to-end test through extract_paths_from_pdf → build_raw_graph → apply_filters.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

if TYPE_CHECKING:
    from pathlib import Path

from src.parser.extractor import extract_paths_from_pdf
from src.parser.filters import apply_filters
from src.parser.graph_builder import build_raw_graph
from tests.fixtures.pdf_factory import (
    create_bezier_pdf,
    create_complex_pdf,
    create_room_pdf,
    create_simple_rect_pdf,
)


class TestParserPipelineRect:
    """End-to-end: simple rectangle PDF through the full pipeline."""

    def test_rect_pdf_full_pipeline(self, tmp_path: Path) -> None:
        # -- Create PDF --
        pdf_path = create_simple_rect_pdf(
            tmp_path / "rect.pdf",
            x=100,
            y=100,
            w=200,
            h=150,
            stroke_width=1.5,
        )

        # -- Extract paths --
        pages = extract_paths_from_pdf(str(pdf_path))
        assert 0 in pages
        paths = pages[0]
        assert len(paths) >= 1

        # -- Build graph --
        graph = build_raw_graph(paths)
        assert graph.nodes.shape[0] >= 4  # rectangle has 4 corners
        assert graph.nodes.shape[1] == 2
        assert graph.edges.shape[0] >= 4  # at least 4 edge segments
        assert graph.edges.shape[1] == 2

        # -- Verify coordinates approximately match input --
        xs = graph.nodes[:, 0]
        ys = graph.nodes[:, 1]
        assert np.min(xs) == pytest.approx(100.0, abs=2.0)
        assert np.max(xs) == pytest.approx(300.0, abs=2.0)
        assert np.min(ys) == pytest.approx(100.0, abs=2.0)
        assert np.max(ys) == pytest.approx(250.0, abs=2.0)

        # -- Apply filters --
        graph = apply_filters(graph)
        assert graph.confidence_wall.shape[0] == len(graph.edges)
        # Thick black solid lines should have decent confidence
        assert np.mean(graph.confidence_wall) > 0.3

        # -- Metadata consistency --
        ne = len(graph.edges)
        assert len(graph.operator_types) == ne
        assert len(graph.stroke_widths) == ne
        assert len(graph.stroke_colors) == ne
        assert len(graph.dash_patterns) == ne
        assert len(graph.edge_to_path) == ne
        assert len(graph.bezier_controls) == ne


class TestParserPipelineRoom:
    """End-to-end: rectangular room (4 wall segments) through the full pipeline."""

    def test_room_pdf_full_pipeline(self, tmp_path: Path) -> None:
        pdf_path = create_room_pdf(
            tmp_path / "room.pdf",
            walls=[
                (100, 100, 400, 100),
                (400, 100, 400, 350),
                (400, 350, 100, 350),
                (100, 350, 100, 100),
            ],
            stroke_width=1.5,
        )

        pages = extract_paths_from_pdf(str(pdf_path))
        paths = pages[0]
        assert len(paths) >= 4  # 4 wall line segments

        graph = build_raw_graph(paths)
        # 4 wall segments → 4 corners after dedup
        assert graph.nodes.shape[0] >= 4
        assert graph.edges.shape[0] >= 4

        graph = apply_filters(graph)
        assert np.all(graph.confidence_wall >= 0.0)
        assert np.all(graph.confidence_wall <= 1.0)
        # Thick, black, solid, axis-aligned → high confidence
        assert np.mean(graph.confidence_wall) > 0.5


class TestParserPipelineComplex:
    """End-to-end: complex PDF with walls + decorative elements."""

    def test_complex_pdf_differentiates_walls_from_decorative(self, tmp_path: Path) -> None:
        pdf_path = create_complex_pdf(tmp_path / "complex.pdf")

        pages = extract_paths_from_pdf(str(pdf_path))
        paths = pages[0]
        assert len(paths) >= 5  # walls + dimension + annotation + hatching

        graph = build_raw_graph(paths)
        graph = apply_filters(graph)

        assert graph.nodes.shape[0] >= 4
        assert graph.edges.shape[0] >= 4
        assert np.all(graph.confidence_wall >= 0.0)
        assert np.all(graph.confidence_wall <= 1.0)

        # All required fields populated
        ne = len(graph.edges)
        assert len(graph.operator_types) == ne
        assert graph.stroke_widths.shape == (ne,)
        assert graph.stroke_colors.shape == (ne, 4)
        assert len(graph.dash_patterns) == ne


class TestParserPipelineBezier:
    """End-to-end: PDF with Bézier curves through the pipeline."""

    def test_bezier_pdf_expands_curves(self, tmp_path: Path) -> None:
        pdf_path = create_bezier_pdf(tmp_path / "bezier.pdf")

        pages = extract_paths_from_pdf(str(pdf_path))
        paths = pages[0]
        assert len(paths) >= 1

        graph = build_raw_graph(paths)
        # Bézier curve should produce multiple edges
        assert graph.edges.shape[0] >= 2
        assert graph.nodes.shape[0] >= 2

        graph = apply_filters(graph)
        assert graph.confidence_wall.shape[0] == len(graph.edges)


class TestParserPipelineEdgeCases:
    """Edge cases for the pipeline."""

    def test_nonexistent_pdf_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            extract_paths_from_pdf("/does/not/exist.pdf")

    def test_empty_page_produces_empty_graph(self, tmp_path: Path) -> None:
        """A blank PDF page should produce an empty graph."""
        import fitz

        pdf_path = tmp_path / "blank.pdf"
        doc = fitz.open()
        doc.new_page(width=612, height=792)
        doc.save(str(pdf_path))
        doc.close()

        pages = extract_paths_from_pdf(str(pdf_path))
        paths = pages[0]
        graph = build_raw_graph(paths)
        assert graph.nodes.shape[0] == 0
        assert graph.edges.shape[0] == 0

        graph = apply_filters(graph)
        assert len(graph.confidence_wall) == 0
