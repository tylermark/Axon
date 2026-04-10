"""Raw spatial graph G₀ construction with Bézier sampling and vertex deduplication.

Converts extracted PDF paths into the unified RawGraph data structure by:
1. Sampling cubic Bézier curves into polyline segments (EQ-03)
2. Deduplicating nearby vertices via KD-tree spatial index
3. Building per-edge metadata arrays for downstream consumption

Tasks: P-004 (Bézier sampling), P-005 (KD-tree dedup), P-006 (graph builder).
Reference: MODEL_SPEC.md §PDF Vector Primitives, ARCHITECTURE.md §Stage 1.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
from scipy.spatial import KDTree

from docs.interfaces.parser_to_tokenizer import (
    BezierControlPoints,
    PathMetadata,
    RawGraph,
)
from src.parser.operators import OperatorType

if TYPE_CHECKING:
    from src.parser.extractor import ExtractedPath
    from src.pipeline.config import ParserConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# P-004: Bézier Curve Sampling
# ---------------------------------------------------------------------------


def sample_bezier(
    p0: np.ndarray,
    p1: np.ndarray,
    p2: np.ndarray,
    p3: np.ndarray,
    resolution: int = 8,
) -> list[np.ndarray]:
    """Sample a cubic Bézier curve into a polyline.

    Evaluates B(t) = (1-t)³P₀ + 3(1-t)²tP₁ + 3(1-t)t²P₂ + t³P₃
    at uniform parameter values to produce ``resolution + 1`` points.

    Args:
        p0: Start point, shape (2,).
        p1: First control point, shape (2,).
        p2: Second control point, shape (2,).
        p3: End point, shape (2,).
        resolution: Number of polyline segments (default 8).

    Returns:
        List of ``resolution + 1`` points for non-degenerate curves,
        or ``[p0, p3]`` for collinear/degenerate control points.
    """
    # Degenerate: all control points identical
    if np.allclose(p0, p1) and np.allclose(p1, p2) and np.allclose(p2, p3):
        return [p0.copy(), p3.copy()]

    # Collinear check: all cross products of difference vectors ≈ 0
    d1 = p1 - p0
    d2 = p2 - p0
    d3 = p3 - p0
    cross1 = d1[0] * d2[1] - d1[1] * d2[0]
    cross2 = d1[0] * d3[1] - d1[1] * d3[0]
    cross3 = d2[0] * d3[1] - d2[1] * d3[0]
    if abs(cross1) < 1e-10 and abs(cross2) < 1e-10 and abs(cross3) < 1e-10:
        return [p0.copy(), p3.copy()]

    # Uniform parameter sampling
    t = np.linspace(0.0, 1.0, resolution + 1)
    omt = 1.0 - t

    # B(t) vectorized: shape (resolution+1, 2)
    points = (
        np.outer(omt**3, p0)
        + np.outer(3.0 * omt**2 * t, p1)
        + np.outer(3.0 * omt * t**2, p2)
        + np.outer(t**3, p3)
    )
    return [points[i] for i in range(len(points))]


# ---------------------------------------------------------------------------
# P-005: KD-tree Vertex Deduplication
# ---------------------------------------------------------------------------


def deduplicate_vertices(
    vertices: np.ndarray,
    tolerance: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Merge nearby vertices using KD-tree spatial indexing.

    Builds a KD-tree, finds clusters of vertices within ``tolerance``
    distance using union-find, and replaces each cluster with its centroid.

    Args:
        vertices: Vertex coordinates, shape (N, 2) float64.
        tolerance: Maximum distance for merging, in PDF user units.

    Returns:
        Tuple of ``(deduplicated_vertices, index_mapping)`` where
        ``deduplicated_vertices`` has shape (M, 2) with M <= N, and
        ``index_mapping`` has shape (N,) mapping original → new index.
    """
    n = len(vertices)
    if n == 0:
        return np.empty((0, 2), dtype=np.float64), np.empty(0, dtype=np.int64)

    if tolerance <= 0.0:
        return vertices.copy(), np.arange(n, dtype=np.int64)

    # Union-find with path compression
    parent = np.arange(n, dtype=np.int64)

    def find(x: int) -> int:
        root = x
        while parent[root] != root:
            root = int(parent[root])
        # Path compression
        while parent[x] != root:
            next_x = int(parent[x])
            parent[x] = root
            x = next_x
        return root

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    # Find all vertex pairs within tolerance
    tree = KDTree(vertices)
    pairs = tree.query_pairs(tolerance, output_type="ndarray")

    for i, j in pairs:
        union(int(i), int(j))

    # Resolve all roots
    roots = np.array([find(i) for i in range(n)], dtype=np.int64)

    # Compute centroids per cluster
    unique_roots, inverse = np.unique(roots, return_inverse=True)
    num_clusters = len(unique_roots)
    centroids = np.zeros((num_clusters, 2), dtype=np.float64)
    counts = np.zeros(num_clusters, dtype=np.int64)

    for i in range(n):
        cluster_idx = inverse[i]
        centroids[cluster_idx] += vertices[i]
        counts[cluster_idx] += 1

    centroids /= counts[:, np.newaxis]

    return centroids, inverse.astype(np.int64)


# ---------------------------------------------------------------------------
# P-006: Raw Graph G₀ Builder
# ---------------------------------------------------------------------------


def build_raw_graph(
    paths: list[ExtractedPath],
    config: ParserConfig | None = None,
    page_width: float = 612.0,
    page_height: float = 792.0,
    page_index: int = 0,
    source_path: str = "",
) -> RawGraph:
    """Construct the raw spatial graph G₀ from extracted PDF paths.

    Processes all paths by sampling Bézier curves, collecting vertices,
    deduplicating via KD-tree, and assembling per-edge metadata arrays.

    Args:
        paths: Extracted PDF paths from the content stream extractor.
        config: Parser configuration. Uses defaults if None.
        page_width: PDF page width in user units (default US Letter).
        page_height: PDF page height in user units (default US Letter).
        page_index: Zero-based page index.
        source_path: File path of the source PDF.

    Returns:
        A fully populated ``RawGraph`` dataclass.
    """
    if config is None:
        from src.pipeline.config import ParserConfig as _ParserConfig

        config = _ParserConfig()

    resolution = config.bezier_sample_resolution
    tolerance = config.vertex_merge_tolerance

    # Respect path limit
    if len(paths) > config.max_paths_per_page:
        logger.warning(
            "build_raw_graph received %d paths, truncating to %d",
            len(paths),
            config.max_paths_per_page,
        )
        paths = paths[: config.max_paths_per_page]

    # -- Accumulation buffers --
    all_vertices: list[np.ndarray] = []
    edge_starts: list[int] = []
    edge_ends: list[int] = []
    edge_op_types: list[OperatorType] = []
    edge_stroke_widths: list[float] = []
    edge_stroke_colors: list[np.ndarray] = []
    edge_fill_colors: list[np.ndarray | None] = []
    edge_dash_patterns: list[tuple[list[float], float]] = []
    edge_to_path_list: list[int] = []
    edge_bezier_controls: list[BezierControlPoints | None] = []
    path_metadata_list: list[PathMetadata] = []
    has_any_fill = False

    vertex_count = 0

    def _add_vertex(pt: np.ndarray) -> int:
        """Append a vertex and return its index."""
        nonlocal vertex_count
        all_vertices.append(pt)
        idx = vertex_count
        vertex_count += 1
        return idx

    def _add_edge(
        start_idx: int,
        end_idx: int,
        op: OperatorType,
        path: ExtractedPath,
        meta_idx: int,
        bezier: BezierControlPoints | None,
    ) -> None:
        """Record an edge with all its metadata."""
        edge_starts.append(start_idx)
        edge_ends.append(end_idx)
        edge_op_types.append(op)
        edge_stroke_widths.append(path.stroke_width)
        edge_stroke_colors.append(path.stroke_color)
        edge_fill_colors.append(path.fill_color)
        edge_dash_patterns.append(path.dash_pattern)
        edge_to_path_list.append(meta_idx)
        edge_bezier_controls.append(bezier)

    for path in paths:
        if not path.subpaths:
            continue

        # Build PathMetadata snapshot
        path_metadata_list.append(
            PathMetadata(
                stroke_width=path.stroke_width,
                stroke_color=path.stroke_color.copy(),
                fill_color=path.fill_color.copy() if path.fill_color is not None else None,
                dash_pattern=path.dash_pattern,
                ctm=path.ctm.copy(),
            )
        )
        meta_idx = len(path_metadata_list) - 1

        if path.fill_color is not None:
            has_any_fill = True

        for subpath in path.subpaths:
            for segment in subpath.segments:
                if segment.operator == OperatorType.CURVETO and segment.control_points is not None:
                    cp = segment.control_points
                    sampled = sample_bezier(cp[0], cp[1], cp[2], cp[3], resolution)

                    if len(sampled) < 2:
                        continue

                    bezier_meta = BezierControlPoints(
                        p0=cp[0].copy(),
                        p1=cp[1].copy(),
                        p2=cp[2].copy(),
                        p3=cp[3].copy(),
                    )

                    for k in range(len(sampled) - 1):
                        si = _add_vertex(sampled[k].copy())
                        ei = _add_vertex(sampled[k + 1].copy())
                        _add_edge(si, ei, OperatorType.CURVETO, path, meta_idx, bezier_meta)

                elif segment.operator in (OperatorType.LINETO, OperatorType.CLOSEPATH):
                    si = _add_vertex(segment.start.copy())
                    ei = _add_vertex(segment.end.copy())
                    _add_edge(si, ei, segment.operator, path, meta_idx, None)

    # -- Empty graph shortcut --
    num_edges = len(edge_starts)
    if vertex_count == 0 or num_edges == 0:
        return RawGraph(
            nodes=np.empty((0, 2), dtype=np.float64),
            edges=np.empty((0, 2), dtype=np.int64),
            operator_types=[],
            stroke_widths=np.empty(0, dtype=np.float64),
            stroke_colors=np.empty((0, 4), dtype=np.float64),
            fill_colors=None,
            dash_patterns=[],
            path_metadata=path_metadata_list,
            edge_to_path=np.empty(0, dtype=np.int64),
            bezier_controls=[],
            confidence_wall=np.empty(0, dtype=np.float64),
            page_width=page_width,
            page_height=page_height,
            page_index=page_index,
            source_path=source_path,
            vertex_merge_tolerance=tolerance,
            bezier_sample_resolution=resolution,
            num_original_paths=len(paths),
        )

    # -- Vertex deduplication --
    vertices_array = np.array(all_vertices, dtype=np.float64)
    deduped_vertices, index_mapping = deduplicate_vertices(vertices_array, tolerance)

    # -- Remap edges and remove zero-length --
    raw_starts = np.array(edge_starts, dtype=np.int64)
    raw_ends = np.array(edge_ends, dtype=np.int64)
    mapped_starts = index_mapping[raw_starts]
    mapped_ends = index_mapping[raw_ends]

    valid_mask = mapped_starts != mapped_ends
    valid_indices = np.nonzero(valid_mask)[0]

    edges = np.stack([mapped_starts[valid_indices], mapped_ends[valid_indices]], axis=1)

    # -- Build filtered metadata arrays --
    n_raw = len(raw_starts)
    if len(edge_op_types) != n_raw:
        raise ValueError(
            f"edge metadata length mismatch: {len(edge_op_types)} op_types vs {n_raw} edges"
        )
    operator_types = [edge_op_types[i] for i in valid_indices]

    stroke_widths = np.array(edge_stroke_widths, dtype=np.float64)[valid_indices]

    stroke_colors = np.stack(edge_stroke_colors, axis=0)[valid_indices]

    if has_any_fill:
        fill_raw = [
            fc if fc is not None else np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64)
            for fc in edge_fill_colors
        ]
        fill_colors: np.ndarray | None = np.stack(fill_raw, axis=0)[valid_indices]
    else:
        fill_colors = None

    dash_patterns = [edge_dash_patterns[i] for i in valid_indices]
    edge_to_path_arr = np.array(edge_to_path_list, dtype=np.int64)[valid_indices]
    bezier_controls = [edge_bezier_controls[i] for i in valid_indices]
    confidence_wall = np.zeros(len(edges), dtype=np.float64)

    logger.info(
        "Built raw graph G₀: %d nodes, %d edges (from %d paths, %d pre-dedup vertices)",
        len(deduped_vertices),
        len(edges),
        len(paths),
        len(vertices_array),
    )

    return RawGraph(
        nodes=deduped_vertices,
        edges=edges,
        operator_types=operator_types,
        stroke_widths=stroke_widths,
        stroke_colors=stroke_colors,
        fill_colors=fill_colors,
        dash_patterns=dash_patterns,
        path_metadata=path_metadata_list,
        edge_to_path=edge_to_path_arr,
        bezier_controls=bezier_controls,
        confidence_wall=confidence_wall,
        page_width=page_width,
        page_height=page_height,
        page_index=page_index,
        source_path=source_path,
        vertex_merge_tolerance=tolerance,
        bezier_sample_resolution=resolution,
        num_original_paths=len(paths),
    )
