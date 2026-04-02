"""Interface contract: Parser Agent → Tokenizer Agent.

Defines the RawGraph dataclass — the output of Stage 1 (PDF Vector Parsing)
and input to Stage 2 (Cross-Modal Tokenization).

Mathematical basis:
    G_init = (V_init, E_init)  [MODEL_SPEC.md §PDF Vector Primitives, EQ-04]

    V_init contains continuous coordinates of all moveto/lineto/curveto endpoints.
    E_init contains drawn segments connecting consecutive points within a path.

    Bézier curves (EQ-03) are sampled into polylines at configurable resolution
    (default: 8 segments per curve). Original control points are preserved as
    edge metadata.

    Duplicate vertices are merged within configurable tolerance (default: 0.5 PDF
    units) using a KD-tree spatial index.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

import numpy as np


class OperatorType(str, Enum):
    """PDF path operator types mapped to their PostScript equivalents.

    Reference: MODEL_SPEC.md Table 1.
    """

    MOVETO = "moveto"
    LINETO = "lineto"
    CURVETO = "curveto"
    CLOSEPATH = "closepath"


@dataclass
class PathMetadata:
    """Per-path metadata from the PDF graphics state.

    Captures the full graphics state context for a single PDF path object,
    including the cumulative Current Transformation Matrix (CTM) resolved
    from nested q/Q state stacks.

    Reference: ARCHITECTURE.md §Stage 1, MODEL_SPEC.md §PDF Vector Primitives.
    """

    stroke_width: float
    """Stroke width in PDF user units."""

    stroke_color: np.ndarray
    """Stroke color as RGBA float64 array, shape (4,)."""

    fill_color: np.ndarray | None
    """Fill color as RGBA float64 array, shape (4,), or None if no fill."""

    dash_pattern: tuple[list[float], float]
    """Dash pattern as (dash_array, dash_phase). Empty array = solid line."""

    ctm: np.ndarray
    """Cumulative Current Transformation Matrix, shape (3, 3) float64.

    Composed from all nested `cm` operators within the q/Q graphics state stack.
    Transforms path coordinates from user space to device space.
    """


@dataclass
class BezierControlPoints:
    """Original cubic Bézier control points preserved as edge metadata.

    When a curveto operator (EQ-03) is sampled into polyline segments,
    the original control points are stored here for potential reconstruction.

    B(t) = (1-t)^3 P0 + 3(1-t)^2 t P1 + 3(1-t)t^2 P2 + t^3 P3

    Reference: MODEL_SPEC.md §PDF Vector Primitives, EQ-03.
    """

    p0: np.ndarray
    """Start point, shape (2,) float64."""

    p1: np.ndarray
    """First control point, shape (2,) float64."""

    p2: np.ndarray
    """Second control point, shape (2,) float64."""

    p3: np.ndarray
    """End point, shape (2,) float64."""


@dataclass
class RawGraph:
    """Raw spatial graph G₀ extracted from a PDF page.

    This is the foundational data structure of the Axon pipeline. It captures
    every geometric primitive drawn on a PDF page as a directed spatial graph,
    preserving continuous coordinates, stroke properties, and graphics state.

    Decorative elements (hatching, dimension lines, furniture symbols) are
    flagged with a confidence score but NOT removed — the Tokenizer Agent
    makes the final semantic decision.

    Attributes map directly to the mathematical formulation:
        G_init = (V_init, E_init)  where
        V_init ∈ R^(N×2) — continuous endpoint coordinates
        E_init ∈ Z^(E×2) — node index pairs defining edges

    Reference: MODEL_SPEC.md §PDF Vector Primitives, ARCHITECTURE.md §Stage 1.
    """

    nodes: np.ndarray
    """Node coordinate matrix, shape (N, 2) float64.

    Each row is an (x, y) coordinate in PDF user space (72 units/inch).
    Coordinates are post-CTM-transform and post-deduplication.
    """

    edges: np.ndarray
    """Edge index array, shape (E, 2) int64.

    Each row [i, j] is a directed edge from node i to node j.
    Indices reference rows in the `nodes` array.
    """

    operator_types: list[OperatorType]
    """Per-edge operator type, length E.

    Indicates which PDF drawing command generated this edge segment:
    LINETO for straight lines, CURVETO for sampled Bézier segments,
    CLOSEPATH for path-closing edges.
    """

    stroke_widths: np.ndarray
    """Per-edge stroke width, shape (E,) float64.

    In PDF user units. Critical heuristic for wall detection:
    structural walls typically have stroke widths in [0.5, 3.0] range.
    """

    stroke_colors: np.ndarray
    """Per-edge stroke color, shape (E, 4) float64.

    RGBA color values in [0, 1] range. Black strokes (0,0,0,1) are
    most commonly associated with structural elements.
    """

    fill_colors: np.ndarray | None
    """Per-edge fill color, shape (E, 4) float64, or None if no fills.

    Present when paths have fill operations. Used by the Tokenizer
    for semantic classification (e.g., filled polygons may indicate
    structural columns or hatched sections).
    """

    dash_patterns: list[tuple[list[float], float]]
    """Per-edge dash pattern, length E.

    Each entry is (dash_array, dash_phase). Dashed lines often
    indicate non-structural elements (hidden lines, projections).
    """

    path_metadata: list[PathMetadata]
    """Per-path metadata, length P (number of original PDF paths).

    Contains full graphics state for each source path. Multiple edges
    may share the same path metadata (one path = multiple segments).
    """

    edge_to_path: np.ndarray
    """Maps each edge to its source path index, shape (E,) int64.

    edge_to_path[i] gives the index into `path_metadata` for edge i.
    """

    bezier_controls: list[BezierControlPoints | None]
    """Per-edge original Bézier control points, length E.

    Non-None only for edges generated by CURVETO sampling (EQ-03).
    Enables reconstruction of the original smooth curve if needed.
    """

    confidence_wall: np.ndarray
    """Per-edge wall confidence score, shape (E,) float64 in [0, 1].

    Heuristic score based on stroke width, color, geometric regularity,
    and dash pattern. Higher values suggest structural wall segments.
    The Parser flags but does NOT filter — Tokenizer decides.

    Reference: ARCHITECTURE.md §Stage 1 Key Design Decisions.
    """

    page_width: float
    """PDF page width in user units (72 units/inch)."""

    page_height: float
    """PDF page height in user units (72 units/inch)."""

    page_index: int = 0
    """Zero-based page index within the source PDF."""

    source_path: str = ""
    """File path of the source PDF document."""

    vertex_merge_tolerance: float = 0.5
    """Tolerance used for KD-tree vertex deduplication, in PDF user units."""

    bezier_sample_resolution: int = 8
    """Number of polyline segments per sampled Bézier curve."""

    num_original_paths: int = 0
    """Total number of PDF path objects on the page before processing."""

    metadata: dict[str, object] = field(default_factory=dict)
    """Arbitrary additional metadata from the parsing stage."""
