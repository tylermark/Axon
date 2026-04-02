"""PyMuPDF content stream extractor with graphics state tracking.

Extracts geometric paths from PDF pages via PyMuPDF's ``get_drawings()`` API,
preserving full graphics state metadata (stroke width, color, dash pattern,
CTM). Each path is decomposed into typed segments (line, curve, closepath)
ready for graph construction in ``graph_builder.py``.

Tasks: P-002 (page extraction), P-003 (graphics state stack).
Reference: PDF Reference 1.7 §8.5, MODEL_SPEC.md §PDF Vector Primitives,
           ARCHITECTURE.md §Stage 1.
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from src.parser.operators import OperatorType

if TYPE_CHECKING:
    import fitz

    from src.pipeline.config import ParserConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class PathSegment:
    """A single segment within a subpath."""

    operator: OperatorType
    start: np.ndarray  # (2,) start point
    end: np.ndarray  # (2,) end point
    control_points: tuple[np.ndarray, ...] | None = None  # For CURVETO: (p0, p1, p2, p3)


@dataclass
class SubPath:
    """A continuous sub-path within a PDF path object."""

    segments: list[PathSegment] = field(default_factory=list)
    is_closed: bool = False


@dataclass
class ExtractedPath:
    """A single PDF path with all its segments and graphics state."""

    subpaths: list[SubPath]
    stroke_width: float
    stroke_color: np.ndarray  # RGBA (4,)
    fill_color: np.ndarray | None  # RGBA (4,) or None
    dash_pattern: tuple[list[float], float]
    ctm: np.ndarray  # (3, 3)
    is_stroked: bool
    is_filled: bool
    is_clipping: bool = False


# ---------------------------------------------------------------------------
# Graphics State
# ---------------------------------------------------------------------------


@dataclass
class GraphicsState:
    """Current PDF graphics state."""

    ctm: np.ndarray = field(default_factory=lambda: np.eye(3, dtype=np.float64))
    stroke_width: float = 1.0
    stroke_color: np.ndarray = field(
        default_factory=lambda: np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    )
    fill_color: np.ndarray | None = None
    dash_pattern: tuple[list[float], float] = field(default_factory=lambda: ([], 0.0))
    line_cap: int = 0
    line_join: int = 0
    miter_limit: float = 10.0
    clipping_path: list[np.ndarray] | None = None


class GraphicsStateStack:
    """Manages a stack of ``GraphicsState`` objects for nested q/Q operators."""

    def __init__(self) -> None:
        self._stack: list[GraphicsState] = [GraphicsState()]

    @property
    def current(self) -> GraphicsState:
        """Return the current (top-of-stack) graphics state."""
        return self._stack[-1]

    def push(self) -> None:
        """Deep-copy the current state onto the stack (``q`` operator)."""
        self._stack.append(copy.deepcopy(self._stack[-1]))

    def pop(self) -> None:
        """Restore the previous state (``Q`` operator).

        If only the base state remains, this is a no-op with a warning.
        """
        if len(self._stack) <= 1:
            logger.warning("GraphicsStateStack.pop() called with no saved state")
            return
        self._stack.pop()

    def apply_ctm(self, matrix: np.ndarray) -> None:
        """Compose a transformation matrix into the current CTM.

        Args:
            matrix: 3x3 affine matrix. For a ``cm`` operator with params
                ``[a, b, c, d, e, f]``, build::

                    [[a, b, 0], [c, d, 0], [e, f, 1]]
        """
        self.current.ctm = self.current.ctm @ matrix

    def set_stroke_width(self, width: float) -> None:
        """Set current stroke width."""
        self.current.stroke_width = width

    def set_stroke_color(self, rgba: np.ndarray) -> None:
        """Set current stroke color as RGBA float64."""
        self.current.stroke_color = rgba

    def set_fill_color(self, rgba: np.ndarray | None) -> None:
        """Set current fill color as RGBA float64, or None."""
        self.current.fill_color = rgba

    def set_dash_pattern(self, dash_array: list[float], dash_phase: float) -> None:
        """Set current dash pattern."""
        self.current.dash_pattern = (dash_array, dash_phase)

    def set_line_cap(self, cap: int) -> None:
        """Set line cap style."""
        self.current.line_cap = cap

    def set_line_join(self, join: int) -> None:
        """Set line join style."""
        self.current.line_join = join

    def set_miter_limit(self, limit: float) -> None:
        """Set miter limit."""
        self.current.miter_limit = limit


# ---------------------------------------------------------------------------
# Color conversion helpers
# ---------------------------------------------------------------------------


def _rgb_to_rgba(rgb: tuple[float, ...] | list[float] | None) -> np.ndarray | None:
    """Convert an RGB tuple (0-1) to RGBA float64 array. None → None."""
    if rgb is None:
        return None
    r, g, b = rgb[0], rgb[1], rgb[2]
    return np.array([r, g, b, 1.0], dtype=np.float64)


def _gray_to_rgba(gray: float) -> np.ndarray:
    """Convert a grayscale value (0-1) to RGBA float64."""
    return np.array([gray, gray, gray, 1.0], dtype=np.float64)


def _cmyk_to_rgba(c: float, m: float, y: float, k: float) -> np.ndarray:
    """Convert CMYK (0-1) to RGBA float64 using standard conversion."""
    r = (1.0 - c) * (1.0 - k)
    g = (1.0 - m) * (1.0 - k)
    b = (1.0 - y) * (1.0 - k)
    return np.array([r, g, b, 1.0], dtype=np.float64)


# ---------------------------------------------------------------------------
# Path Accumulator
# ---------------------------------------------------------------------------


class PathAccumulator:
    """Collects drawing commands within a single PDF path.

    Coordinates are stored directly as received — PyMuPDF's ``get_drawings()``
    returns coordinates already resolved to page space, so no CTM transform
    is needed during accumulation.
    """

    def __init__(self) -> None:
        self._subpaths: list[SubPath] = []
        self._current_subpath: SubPath | None = None
        self._current_point: np.ndarray | None = None
        self._subpath_start: np.ndarray | None = None

    def _ensure_subpath(self) -> SubPath:
        """Ensure a current subpath exists, creating one if needed."""
        if self._current_subpath is None:
            self._current_subpath = SubPath()
            self._subpaths.append(self._current_subpath)
        return self._current_subpath

    def moveto(self, x: float, y: float) -> None:
        """Start a new subpath at (x, y)."""
        pt = np.array([x, y], dtype=np.float64)
        # Close out previous subpath (if any) and start fresh
        self._current_subpath = SubPath()
        self._subpaths.append(self._current_subpath)
        self._current_point = pt
        self._subpath_start = pt.copy()

    def lineto(self, x: float, y: float) -> None:
        """Add a line segment from the current point to (x, y)."""
        end = np.array([x, y], dtype=np.float64)
        subpath = self._ensure_subpath()
        if self._current_point is None:
            self._current_point = np.array([0.0, 0.0], dtype=np.float64)
            self._subpath_start = self._current_point.copy()
        subpath.segments.append(
            PathSegment(
                operator=OperatorType.LINETO,
                start=self._current_point.copy(),
                end=end,
            )
        )
        self._current_point = end

    def curveto(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        x3: float,
        y3: float,
    ) -> None:
        """Add a cubic Bezier from the current point through (x1,y1), (x2,y2) to (x3,y3)."""
        subpath = self._ensure_subpath()
        if self._current_point is None:
            self._current_point = np.array([0.0, 0.0], dtype=np.float64)
            self._subpath_start = self._current_point.copy()
        p0 = self._current_point.copy()
        p1 = np.array([x1, y1], dtype=np.float64)
        p2 = np.array([x2, y2], dtype=np.float64)
        p3 = np.array([x3, y3], dtype=np.float64)
        subpath.segments.append(
            PathSegment(
                operator=OperatorType.CURVETO,
                start=p0,
                end=p3,
                control_points=(p0, p1, p2, p3),
            )
        )
        self._current_point = p3

    def curveto_v(self, x2: float, y2: float, x3: float, y3: float) -> None:
        """Cubic Bezier where the first control point equals the current point."""
        if self._current_point is None:
            self._current_point = np.array([0.0, 0.0], dtype=np.float64)
            self._subpath_start = self._current_point.copy()
        cp = self._current_point
        self.curveto(cp[0], cp[1], x2, y2, x3, y3)

    def curveto_y(self, x1: float, y1: float, x3: float, y3: float) -> None:
        """Cubic Bezier where the last control point equals the endpoint."""
        self.curveto(x1, y1, x3, y3, x3, y3)

    def closepath(self) -> None:
        """Close the current subpath back to its starting point."""
        subpath = self._ensure_subpath()
        if self._current_point is not None and self._subpath_start is not None:
            if not np.allclose(self._current_point, self._subpath_start, atol=1e-10):
                subpath.segments.append(
                    PathSegment(
                        operator=OperatorType.CLOSEPATH,
                        start=self._current_point.copy(),
                        end=self._subpath_start.copy(),
                    )
                )
            subpath.is_closed = True
            self._current_point = self._subpath_start.copy()

    def rect(self, x: float, y: float, w: float, h: float) -> None:
        """Add a rectangle as moveto + 3 lineto + closepath."""
        self.moveto(x, y)
        self.lineto(x + w, y)
        self.lineto(x + w, y + h)
        self.lineto(x, y + h)
        self.closepath()

    def finalize(
        self, graphics_state: GraphicsState, *, is_stroked: bool, is_filled: bool
    ) -> ExtractedPath:
        """Snapshot the accumulated path with its graphics state.

        Args:
            graphics_state: Current graphics state to capture.
            is_stroked: Whether this path is stroked.
            is_filled: Whether this path is filled.

        Returns:
            A finalized ``ExtractedPath``.
        """
        return ExtractedPath(
            subpaths=[sp for sp in self._subpaths if sp.segments],
            stroke_width=graphics_state.stroke_width,
            stroke_color=graphics_state.stroke_color.copy(),
            fill_color=graphics_state.fill_color.copy()
            if graphics_state.fill_color is not None
            else None,
            dash_pattern=graphics_state.dash_pattern,
            ctm=graphics_state.ctm.copy(),
            is_stroked=is_stroked,
            is_filled=is_filled,
        )


# ---------------------------------------------------------------------------
# PyMuPDF drawing → ExtractedPath conversion
# ---------------------------------------------------------------------------


def _point_to_array(pt: fitz.Point) -> np.ndarray:
    """Convert a fitz.Point to a numpy (2,) array."""
    return np.array([pt.x, pt.y], dtype=np.float64)


def _build_path_from_drawing(drawing: dict) -> ExtractedPath:
    """Convert a single PyMuPDF ``get_drawings()`` dict to an ExtractedPath.

    PyMuPDF's ``get_drawings()`` returns pre-resolved graphics state so we
    read colors/widths directly from the dict rather than tracking a state
    stack ourselves.

    Args:
        drawing: A dict from ``page.get_drawings()``.

    Returns:
        An ``ExtractedPath`` with all segments and metadata.
    """
    accumulator = PathAccumulator()

    items: list[tuple] = drawing.get("items", [])
    for item in items:
        op = item[0]  # operator string: "l", "c", "re", "qu"
        if op == "l":
            # lineto: (op, start_point, end_point)
            start, end = item[1], item[2]
            # Set current point on first segment via implicit moveto
            if accumulator._current_point is None or not np.allclose(
                accumulator._current_point,
                np.array([start.x, start.y]),
                atol=1e-10,
            ):
                accumulator.moveto(start.x, start.y)
            accumulator.lineto(end.x, end.y)

        elif op == "c":
            # curveto: (op, start, ctrl1, ctrl2, end)
            start, ctrl1, ctrl2, end = item[1], item[2], item[3], item[4]
            if accumulator._current_point is None or not np.allclose(
                accumulator._current_point,
                np.array([start.x, start.y]),
                atol=1e-10,
            ):
                accumulator.moveto(start.x, start.y)
            accumulator.curveto(ctrl1.x, ctrl1.y, ctrl2.x, ctrl2.y, end.x, end.y)

        elif op == "re":
            # rectangle: (op, top_left_rect_as_Rect)
            rect = item[1]
            accumulator.rect(rect.x0, rect.y0, rect.width, rect.height)

        elif op == "qu":
            # quad: (op, ul, ur, ll, lr) — four corner points
            ul, ur, ll, lr = item[1], item[2], item[3], item[4]
            accumulator.moveto(ul.x, ul.y)
            accumulator.lineto(ur.x, ur.y)
            accumulator.lineto(lr.x, lr.y)
            accumulator.lineto(ll.x, ll.y)
            accumulator.closepath()

    # Determine stroke/fill from drawing dict
    stroke_color_rgb = drawing.get("color")
    fill_color_rgb = drawing.get("fill")
    is_stroked = stroke_color_rgb is not None
    is_filled = fill_color_rgb is not None

    stroke_rgba = (
        _rgb_to_rgba(stroke_color_rgb)
        if stroke_color_rgb is not None
        else np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    )
    fill_rgba = _rgb_to_rgba(fill_color_rgb)

    stroke_width = drawing.get("width", 1.0)
    if stroke_width is None:
        stroke_width = 0.0

    # Dash pattern
    dashes_raw = drawing.get("dashes")
    if dashes_raw and isinstance(dashes_raw, str):
        dash_pattern: tuple[list[float], float] = ([], 0.0)
    elif dashes_raw:
        dash_pattern = (list(dashes_raw), 0.0)
    else:
        dash_pattern = ([], 0.0)

    # Handle closePath flag
    if drawing.get("closePath", False):
        accumulator.closepath()

    # Build the graphics state for finalization
    state = GraphicsState(
        stroke_width=stroke_width,
        stroke_color=stroke_rgba,
        fill_color=fill_rgba,
        dash_pattern=dash_pattern,
        line_cap=drawing.get("lineCap", 0),
        line_join=drawing.get("lineJoin", 0),
    )

    return accumulator.finalize(state, is_stroked=is_stroked, is_filled=is_filled)


# ---------------------------------------------------------------------------
# Page-level extraction
# ---------------------------------------------------------------------------


def extract_paths(
    page: fitz.Page,
    *,
    min_stroke_width: float = 0.1,
    max_paths_per_page: int = 200_000,
) -> list[ExtractedPath]:
    """Extract all vector paths from a single PDF page.

    Uses PyMuPDF's ``page.get_drawings()`` to retrieve pre-resolved drawing
    commands, then converts each to an ``ExtractedPath`` with full metadata.

    Args:
        page: A ``fitz.Page`` object.
        min_stroke_width: Discard stroked paths thinner than this (PDF units).
        max_paths_per_page: Safety cap on number of paths processed.

    Returns:
        List of ``ExtractedPath`` objects for the page.
    """
    drawings = page.get_drawings()

    if len(drawings) > max_paths_per_page:
        logger.warning(
            "Page %d has %d paths, exceeding limit of %d — truncating",
            page.number,
            len(drawings),
            max_paths_per_page,
        )
        drawings = drawings[:max_paths_per_page]

    paths: list[ExtractedPath] = []
    for drawing in drawings:
        extracted = _build_path_from_drawing(drawing)

        # Skip empty paths
        if not extracted.subpaths:
            continue

        # Filter thin strokes (only for stroked paths)
        if extracted.is_stroked and extracted.stroke_width < min_stroke_width:
            continue

        paths.append(extracted)

    logger.debug("Extracted %d paths from page %d", len(paths), page.number)
    return paths


def extract_paths_from_pdf(
    pdf_path: str | Path,
    page_indices: list[int] | None = None,
    config: ParserConfig | None = None,
) -> dict[int, list[ExtractedPath]]:
    """Extract paths from a PDF file.

    Args:
        pdf_path: Path to the PDF file.
        page_indices: Specific page indices to process (0-based). If None,
            all pages are processed.
        config: Parser configuration. Uses defaults if None.

    Returns:
        Dict mapping ``{page_index: [ExtractedPath, ...]}``.

    Raises:
        FileNotFoundError: If the PDF file does not exist.
        RuntimeError: If PyMuPDF cannot open the file.
    """
    import fitz as fitz_module

    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    if config is None:
        from src.pipeline.config import ParserConfig as _ParserConfig

        config = _ParserConfig()

    doc = fitz_module.open(str(pdf_path))
    try:
        indices = page_indices if page_indices is not None else list(range(len(doc)))
        result: dict[int, list[ExtractedPath]] = {}

        for idx in indices:
            if idx < 0 or idx >= len(doc):
                logger.warning("Skipping invalid page index %d (doc has %d pages)", idx, len(doc))
                continue

            page = doc[idx]
            result[idx] = extract_paths(
                page,
                min_stroke_width=config.min_stroke_width,
                max_paths_per_page=config.max_paths_per_page,
            )
            logger.info(
                "Page %d: extracted %d paths",
                idx,
                len(result[idx]),
            )

        return result
    finally:
        doc.close()
