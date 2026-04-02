"""Integration test: Axon parser against real Capsule Manufacturing floor plans.

Q-021: Smoke-tests the Layer 1 parser (extract_paths_from_pdf → build_raw_graph
→ apply_filters) against the 103 real-world architectural PDFs in
``CapsuleFloorPlans/Floorplans/``.

Three test classes:

* ``TestCapsuleSampleSmoke``  — parametrized over 10 diverse sample PDFs;
  verifies RawGraph shape, coordinate sanity, and timing (< 10 s each).
* ``TestCapsuleGraphStatistics`` — graph-quality bounds on the same samples.
* ``TestCapsuleRobustness`` — full 103-PDF sweep; marked ``slow``, skipped
  when the data directory is absent (CI environments).

The PDF directory is expected at::

    /mnt/c/Users/tyler/Axon/CapsuleFloorPlans/Floorplans/

Both the sample and robustness tests skip automatically when that path does
not exist, so the test suite remains green in CI without the data files.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# PDF directory fixture
# ---------------------------------------------------------------------------

_CAPSULE_PDF_DIR = Path("/mnt/c/Users/tyler/Axon/CapsuleFloorPlans/Floorplans")

_PDFS_AVAILABLE = _CAPSULE_PDF_DIR.is_dir() and any(_CAPSULE_PDF_DIR.glob("*.pdf"))

_SKIP_NO_PDFS = pytest.mark.skipif(
    not _PDFS_AVAILABLE,
    reason=(
        "Capsule PDF directory not found or empty: "
        f"{_CAPSULE_PDF_DIR}. "
        "Skipping real-data integration tests."
    ),
)


def _all_pdf_paths() -> list[Path]:
    """Return all PDF paths from the Capsule floor-plan directory, sorted."""
    if not _PDFS_AVAILABLE:
        return []
    return sorted(_CAPSULE_PDF_DIR.glob("*.pdf"))


# ---------------------------------------------------------------------------
# Sample selection — 10 diverse PDFs spanning different projects / naming styles
# ---------------------------------------------------------------------------

_SAMPLE_PDF_NAMES: list[str] = [
    # Simple/standalone plans
    "TH-A10-FLOOR-PLAN-Rev.0.pdf",
    "FH-A10-FLOOR-PLAN-Rev.0.pdf",
    "A-121-RESIDENTIAL LEVEL 01 FLOOR PLAN.pdf",
    "A3.02A-FLOOR-PLAN---LEVEL-2---EAST-Rev.5.pdf",
    # Multi-project extracts with bracket naming
    "04 - ARCHITECTURAL Rev.1 Extract[10].pdf",
    "2023-04-05_1535_Sunset_Raffi_Reduced W ADU Extract[12].pdf",
    "231215 400 Oceangate SPR Submittal - COMBINED (2) [26].pdf",
    "URC ANNEX AT COVELL - PRE-APPLICATION SUBMITTAL SET_05-31-2023 TAKEOFF [2].pdf",
    "Working Drawings_Parkside Apart_02-22-23 (1) Bid Set 2.23.23 [22].pdf",
    # Unusual naming
    "(B) Capsule + Crate - SoLa Fernando Scope Review - Updated for Approval.pdf",
]


def _sample_pdf_paths() -> list[Path]:
    """Return the 10 sample PDF paths that exist on disk.

    Falls back gracefully: if a named file doesn't exist (e.g., data was
    partially installed) it is silently omitted.
    """
    return [_CAPSULE_PDF_DIR / name for name in _SAMPLE_PDF_NAMES if (_CAPSULE_PDF_DIR / name).is_file()]


# ---------------------------------------------------------------------------
# Shared parsing helper
# ---------------------------------------------------------------------------


def _parse_pdf(pdf_path: Path):
    """Run the full Layer 1 parsing chain on a single PDF (page 0).

    Returns:
        ``RawGraph`` with ``confidence_wall`` populated.

    Raises:
        Any exception that escapes the parser (test will fail, as intended).
    """
    import fitz  # third-party — confirms PyMuPDF is installed

    from src.parser.extractor import extract_paths_from_pdf
    from src.parser.filters import apply_filters
    from src.parser.graph_builder import build_raw_graph

    doc = fitz.open(str(pdf_path))
    try:
        page = doc[0]
        page_width = float(page.rect.width)
        page_height = float(page.rect.height)
    finally:
        doc.close()

    pages = extract_paths_from_pdf(str(pdf_path), page_indices=[0])
    paths = pages.get(0, [])

    from src.pipeline.config import ParserConfig

    config = ParserConfig()
    graph = build_raw_graph(
        paths,
        config=config,
        page_width=page_width,
        page_height=page_height,
        page_index=0,
        source_path=str(pdf_path),
    )
    graph = apply_filters(graph, config=config)
    return graph


# ---------------------------------------------------------------------------
# Q-021-A: Smoke test — 10 sample PDFs
# ---------------------------------------------------------------------------


@_SKIP_NO_PDFS
class TestCapsuleSampleSmoke:
    """Parser smoke test on 10 diverse Capsule floor-plan PDFs.

    Each test is parametrized over the sample set.  Tests assert that the
    parser returns a valid ``RawGraph`` and does not crash.
    """

    @pytest.fixture(scope="class")
    def sample_graphs(self) -> dict[str, object]:
        """Parse all sample PDFs once and cache the results for the class."""
        results = {}
        for pdf_path in _sample_pdf_paths():
            results[pdf_path.name] = _parse_pdf(pdf_path)
        return results

    @pytest.mark.parametrize("pdf_name", _SAMPLE_PDF_NAMES)
    def test_returns_raw_graph(self, pdf_name: str, sample_graphs: dict) -> None:
        """Parser returns a RawGraph instance for each sample PDF."""
        from docs.interfaces.parser_to_tokenizer import RawGraph

        if pdf_name not in sample_graphs:
            pytest.skip(f"PDF not present on disk: {pdf_name}")

        graph = sample_graphs[pdf_name]
        assert isinstance(graph, RawGraph), (
            f"{pdf_name}: expected RawGraph, got {type(graph).__name__}"
        )

    @pytest.mark.parametrize("pdf_name", _SAMPLE_PDF_NAMES)
    def test_nodes_shape(self, pdf_name: str, sample_graphs: dict) -> None:
        """Node array is 2-D with shape (N, 2) and dtype float64."""
        if pdf_name not in sample_graphs:
            pytest.skip(f"PDF not present on disk: {pdf_name}")

        graph = sample_graphs[pdf_name]
        assert graph.nodes.ndim == 2, f"{pdf_name}: nodes.ndim={graph.nodes.ndim}"
        assert graph.nodes.shape[1] == 2, f"{pdf_name}: nodes shape={graph.nodes.shape}"
        assert graph.nodes.dtype == np.float64, (
            f"{pdf_name}: nodes dtype={graph.nodes.dtype}"
        )

    @pytest.mark.parametrize("pdf_name", _SAMPLE_PDF_NAMES)
    def test_edges_shape(self, pdf_name: str, sample_graphs: dict) -> None:
        """Edge array is 2-D with shape (E, 2) and dtype int64 when non-empty."""
        if pdf_name not in sample_graphs:
            pytest.skip(f"PDF not present on disk: {pdf_name}")

        graph = sample_graphs[pdf_name]
        assert graph.edges.ndim == 2, f"{pdf_name}: edges.ndim={graph.edges.ndim}"
        assert graph.edges.shape[1] == 2, f"{pdf_name}: edges shape={graph.edges.shape}"
        assert graph.edges.dtype == np.int64, (
            f"{pdf_name}: edges dtype={graph.edges.dtype}"
        )

    @pytest.mark.parametrize("pdf_name", _SAMPLE_PDF_NAMES)
    def test_node_count_positive(self, pdf_name: str, sample_graphs: dict) -> None:
        """Real floor plans produce at least one node."""
        if pdf_name not in sample_graphs:
            pytest.skip(f"PDF not present on disk: {pdf_name}")

        graph = sample_graphs[pdf_name]
        assert graph.nodes.shape[0] > 0, (
            f"{pdf_name}: parser returned 0 nodes — unexpected for a real floor plan"
        )

    @pytest.mark.parametrize("pdf_name", _SAMPLE_PDF_NAMES)
    def test_edge_count_positive(self, pdf_name: str, sample_graphs: dict) -> None:
        """Real floor plans produce at least one edge."""
        if pdf_name not in sample_graphs:
            pytest.skip(f"PDF not present on disk: {pdf_name}")

        graph = sample_graphs[pdf_name]
        assert graph.edges.shape[0] > 0, (
            f"{pdf_name}: parser returned 0 edges — unexpected for a real floor plan"
        )

    @pytest.mark.parametrize("pdf_name", _SAMPLE_PDF_NAMES)
    def test_source_path_populated(self, pdf_name: str, sample_graphs: dict) -> None:
        """source_path is set to the input PDF path."""
        if pdf_name not in sample_graphs:
            pytest.skip(f"PDF not present on disk: {pdf_name}")

        graph = sample_graphs[pdf_name]
        assert graph.source_path != "", f"{pdf_name}: source_path is empty"
        assert pdf_name in graph.source_path, (
            f"{pdf_name}: source_path='{graph.source_path}' doesn't contain PDF name"
        )

    @pytest.mark.parametrize("pdf_name", _SAMPLE_PDF_NAMES)
    def test_page_dimensions_positive(self, pdf_name: str, sample_graphs: dict) -> None:
        """page_width and page_height are positive and finite."""
        if pdf_name not in sample_graphs:
            pytest.skip(f"PDF not present on disk: {pdf_name}")

        graph = sample_graphs[pdf_name]
        assert graph.page_width > 0.0, f"{pdf_name}: page_width={graph.page_width}"
        assert graph.page_height > 0.0, f"{pdf_name}: page_height={graph.page_height}"
        assert np.isfinite(graph.page_width), f"{pdf_name}: page_width not finite"
        assert np.isfinite(graph.page_height), f"{pdf_name}: page_height not finite"

    @pytest.mark.parametrize("pdf_name", _SAMPLE_PDF_NAMES)
    def test_timing_under_10_seconds(self, pdf_name: str) -> None:
        """Parser completes within 10 seconds for a single real PDF (page 0).

        This test re-parses to get an accurate wall-clock measurement; the
        ``sample_graphs`` fixture result is intentionally not reused here.
        """
        pdf_path = _CAPSULE_PDF_DIR / pdf_name
        if not pdf_path.is_file():
            pytest.skip(f"PDF not present on disk: {pdf_name}")

        start = time.perf_counter()
        _parse_pdf(pdf_path)
        elapsed = time.perf_counter() - start

        assert elapsed < 10.0, (
            f"{pdf_name}: parsing took {elapsed:.2f}s — exceeds 10s budget"
        )


# ---------------------------------------------------------------------------
# Q-021-B: Graph quality / statistics on sample PDFs
# ---------------------------------------------------------------------------


@_SKIP_NO_PDFS
class TestCapsuleGraphStatistics:
    """Graph quality assertions for the 10 sample PDFs.

    Bounds are intentionally wide to accommodate the full range of floor-plan
    complexity (simple ADU vs. large multi-unit residential building).
    """

    # Shared parse results (same fixture approach as smoke class)
    @pytest.fixture(scope="class")
    def sample_graphs(self) -> dict[str, object]:
        """Parse all sample PDFs once."""
        results = {}
        for pdf_path in _sample_pdf_paths():
            results[pdf_path.name] = _parse_pdf(pdf_path)
        return results

    @pytest.mark.parametrize("pdf_name", _SAMPLE_PDF_NAMES)
    def test_node_count_in_reasonable_range(
        self, pdf_name: str, sample_graphs: dict
    ) -> None:
        """Node count is between 4 and 50,000 for any real floor plan."""
        if pdf_name not in sample_graphs:
            pytest.skip(f"PDF not present on disk: {pdf_name}")

        graph = sample_graphs[pdf_name]
        n = graph.nodes.shape[0]
        assert n >= 4, f"{pdf_name}: node count {n} < 4 — unexpectedly sparse"
        assert n <= 50_000, f"{pdf_name}: node count {n} > 50,000 — unexpectedly dense"

    @pytest.mark.parametrize("pdf_name", _SAMPLE_PDF_NAMES)
    def test_edge_count_in_reasonable_range(
        self, pdf_name: str, sample_graphs: dict
    ) -> None:
        """Edge count is between 2 and 100,000 for any real floor plan."""
        if pdf_name not in sample_graphs:
            pytest.skip(f"PDF not present on disk: {pdf_name}")

        graph = sample_graphs[pdf_name]
        e = graph.edges.shape[0]
        assert e >= 2, f"{pdf_name}: edge count {e} < 2 — unexpectedly sparse"
        assert e <= 100_000, f"{pdf_name}: edge count {e} > 100,000 — unexpectedly dense"

    @pytest.mark.parametrize("pdf_name", _SAMPLE_PDF_NAMES)
    def test_coordinates_finite(self, pdf_name: str, sample_graphs: dict) -> None:
        """All node coordinates are finite (no NaN or Inf)."""
        if pdf_name not in sample_graphs:
            pytest.skip(f"PDF not present on disk: {pdf_name}")

        graph = sample_graphs[pdf_name]
        if graph.nodes.shape[0] == 0:
            pytest.skip(f"{pdf_name}: empty graph — no coordinates to check")

        assert np.all(np.isfinite(graph.nodes)), (
            f"{pdf_name}: node coordinates contain NaN or Inf"
        )

    @pytest.mark.parametrize("pdf_name", _SAMPLE_PDF_NAMES)
    def test_coordinates_positive(self, pdf_name: str, sample_graphs: dict) -> None:
        """Node coordinates are non-negative (PDF user space has origin at corner)."""
        if pdf_name not in sample_graphs:
            pytest.skip(f"PDF not present on disk: {pdf_name}")

        graph = sample_graphs[pdf_name]
        if graph.nodes.shape[0] == 0:
            pytest.skip(f"{pdf_name}: empty graph — no coordinates to check")

        # Allow a small negative margin for floating-point precision around origin
        assert np.all(graph.nodes >= -1.0), (
            f"{pdf_name}: node coordinates contain values < -1 "
            f"(min={graph.nodes.min():.3f})"
        )

    @pytest.mark.parametrize("pdf_name", _SAMPLE_PDF_NAMES)
    def test_coordinates_within_page_bounds(
        self, pdf_name: str, sample_graphs: dict
    ) -> None:
        """Node coordinates do not exceed page dimensions by more than 1 unit."""
        if pdf_name not in sample_graphs:
            pytest.skip(f"PDF not present on disk: {pdf_name}")

        graph = sample_graphs[pdf_name]
        if graph.nodes.shape[0] == 0:
            pytest.skip(f"{pdf_name}: empty graph — no coordinates to check")

        margin = 1.0  # PDF unit tolerance for rounding
        xs = graph.nodes[:, 0]
        ys = graph.nodes[:, 1]

        assert np.all(xs <= graph.page_width + margin), (
            f"{pdf_name}: x coordinates exceed page_width={graph.page_width:.1f} "
            f"(max x={xs.max():.3f})"
        )
        assert np.all(ys <= graph.page_height + margin), (
            f"{pdf_name}: y coordinates exceed page_height={graph.page_height:.1f} "
            f"(max y={ys.max():.3f})"
        )

    @pytest.mark.parametrize("pdf_name", _SAMPLE_PDF_NAMES)
    def test_no_duplicate_edges(self, pdf_name: str, sample_graphs: dict) -> None:
        """Edge list contains no exact duplicate (i, j) pairs."""
        if pdf_name not in sample_graphs:
            pytest.skip(f"PDF not present on disk: {pdf_name}")

        graph = sample_graphs[pdf_name]
        if graph.edges.shape[0] == 0:
            pytest.skip(f"{pdf_name}: empty edge list — nothing to check")

        edge_tuples = [tuple(row) for row in graph.edges.tolist()]
        unique_count = len(set(edge_tuples))
        assert unique_count == len(edge_tuples), (
            f"{pdf_name}: {len(edge_tuples) - unique_count} duplicate edges found"
        )

    @pytest.mark.parametrize("pdf_name", _SAMPLE_PDF_NAMES)
    def test_edge_indices_in_bounds(self, pdf_name: str, sample_graphs: dict) -> None:
        """All edge indices reference valid rows in the node array."""
        if pdf_name not in sample_graphs:
            pytest.skip(f"PDF not present on disk: {pdf_name}")

        graph = sample_graphs[pdf_name]
        if graph.edges.shape[0] == 0:
            pytest.skip(f"{pdf_name}: empty edge list — nothing to check")

        n_nodes = graph.nodes.shape[0]
        assert int(graph.edges.min()) >= 0, (
            f"{pdf_name}: edge index < 0"
        )
        assert int(graph.edges.max()) < n_nodes, (
            f"{pdf_name}: edge index {graph.edges.max()} >= n_nodes={n_nodes}"
        )

    @pytest.mark.parametrize("pdf_name", _SAMPLE_PDF_NAMES)
    def test_confidence_wall_range(self, pdf_name: str, sample_graphs: dict) -> None:
        """Wall confidence scores are in [0, 1] for all edges."""
        if pdf_name not in sample_graphs:
            pytest.skip(f"PDF not present on disk: {pdf_name}")

        graph = sample_graphs[pdf_name]
        if graph.confidence_wall.shape[0] == 0:
            pytest.skip(f"{pdf_name}: empty graph — no confidence scores")

        assert np.all(graph.confidence_wall >= 0.0), (
            f"{pdf_name}: confidence_wall has values < 0"
        )
        assert np.all(graph.confidence_wall <= 1.0), (
            f"{pdf_name}: confidence_wall has values > 1"
        )

    @pytest.mark.parametrize("pdf_name", _SAMPLE_PDF_NAMES)
    def test_metadata_arrays_consistent(self, pdf_name: str, sample_graphs: dict) -> None:
        """All per-edge metadata arrays have matching lengths."""
        if pdf_name not in sample_graphs:
            pytest.skip(f"PDF not present on disk: {pdf_name}")

        graph = sample_graphs[pdf_name]
        n_edges = graph.edges.shape[0]

        assert len(graph.operator_types) == n_edges, (
            f"{pdf_name}: operator_types length mismatch: "
            f"{len(graph.operator_types)} != {n_edges}"
        )
        assert graph.stroke_widths.shape[0] == n_edges, (
            f"{pdf_name}: stroke_widths length mismatch: "
            f"{graph.stroke_widths.shape[0]} != {n_edges}"
        )
        assert graph.stroke_colors.shape[0] == n_edges, (
            f"{pdf_name}: stroke_colors length mismatch: "
            f"{graph.stroke_colors.shape[0]} != {n_edges}"
        )
        assert len(graph.dash_patterns) == n_edges, (
            f"{pdf_name}: dash_patterns length mismatch: "
            f"{len(graph.dash_patterns)} != {n_edges}"
        )
        assert graph.edge_to_path.shape[0] == n_edges, (
            f"{pdf_name}: edge_to_path length mismatch: "
            f"{graph.edge_to_path.shape[0]} != {n_edges}"
        )
        assert len(graph.bezier_controls) == n_edges, (
            f"{pdf_name}: bezier_controls length mismatch: "
            f"{len(graph.bezier_controls)} != {n_edges}"
        )
        assert graph.confidence_wall.shape[0] == n_edges, (
            f"{pdf_name}: confidence_wall length mismatch: "
            f"{graph.confidence_wall.shape[0]} != {n_edges}"
        )

    @pytest.mark.parametrize("pdf_name", _SAMPLE_PDF_NAMES)
    def test_stroke_colors_rgba(self, pdf_name: str, sample_graphs: dict) -> None:
        """Stroke color array has shape (E, 4) with values in [0, 1]."""
        if pdf_name not in sample_graphs:
            pytest.skip(f"PDF not present on disk: {pdf_name}")

        graph = sample_graphs[pdf_name]
        if graph.edges.shape[0] == 0:
            pytest.skip(f"{pdf_name}: empty graph — no colors to check")

        assert graph.stroke_colors.ndim == 2, (
            f"{pdf_name}: stroke_colors.ndim={graph.stroke_colors.ndim}, expected 2"
        )
        assert graph.stroke_colors.shape[1] == 4, (
            f"{pdf_name}: stroke_colors.shape={graph.stroke_colors.shape}, expected (E, 4)"
        )
        assert np.all(graph.stroke_colors >= 0.0), (
            f"{pdf_name}: stroke_colors contains values < 0"
        )
        assert np.all(graph.stroke_colors <= 1.0), (
            f"{pdf_name}: stroke_colors contains values > 1"
        )


# ---------------------------------------------------------------------------
# Q-021-C: Robustness — all 103 PDFs (slow, skip if data absent)
# ---------------------------------------------------------------------------


@pytest.mark.slow
@_SKIP_NO_PDFS
class TestCapsuleRobustness:
    """Full 103-PDF sweep: the parser must not raise on any real Capsule PDF.

    This test is marked ``slow`` — run it explicitly with::

        pytest -m slow tests/integration/test_capsule_pdfs.py

    An empty graph (0 nodes / 0 edges) is acceptable for pages that contain
    no vector geometry (scanned rasters, title blocks, etc.).  The only
    failure condition is an unhandled exception from the parser.
    """

    def test_no_crash_on_all_capsule_pdfs(self) -> None:
        """Parser completes without exception on all 103 Capsule PDFs.

        Iterates every PDF in the Capsule floor-plan directory (page 0 of
        each).  Collects exceptions rather than failing immediately so that
        the full failure list is reported at once.

        Reports:
            - Success count
            - Empty-graph count (no nodes/edges — acceptable)
            - Failure count with exception details
        """
        all_pdfs = _all_pdf_paths()
        assert len(all_pdfs) > 0, "No PDFs found — check the Capsule PDF directory"

        successes: list[str] = []
        empty_graphs: list[str] = []
        failures: list[tuple[str, Exception]] = []

        for pdf_path in all_pdfs:
            try:
                graph = _parse_pdf(pdf_path)
                if graph.nodes.shape[0] == 0 or graph.edges.shape[0] == 0:
                    empty_graphs.append(pdf_path.name)
                else:
                    successes.append(pdf_path.name)
            except Exception as exc:
                failures.append((pdf_path.name, exc))

        total = len(all_pdfs)
        n_success = len(successes)
        n_empty = len(empty_graphs)
        n_fail = len(failures)

        # Build a human-readable summary for pytest output
        summary_lines = [
            "",
            "=" * 60,
            "Capsule PDF Robustness Report",
            "=" * 60,
            f"Total PDFs:    {total}",
            f"Parsed OK:     {n_success}",
            f"Empty graphs:  {n_empty} (acceptable — no vector geometry on page 0)",
            f"Failures:      {n_fail}",
        ]

        if failures:
            summary_lines.append("")
            summary_lines.append("FAILED PDFs:")
            for name, exc in failures:
                summary_lines.append(f"  [{type(exc).__name__}] {name}")
                summary_lines.append(f"    {exc}")

        summary = "\n".join(summary_lines)

        assert n_fail == 0, (
            f"Parser raised exceptions on {n_fail}/{total} PDFs.\n{summary}"
        )

        # Print the summary regardless of pass/fail (visible with pytest -s)
        print(summary)
