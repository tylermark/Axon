"""Rule-based wall classification — thickness, context, and fire rating.

Implements CL-002 (thickness rules), CL-003 (context rules), and
CL-004 (fire rating detection) as composable rule functions that
produce per-signal scores.

Each rule function takes wall and graph data and returns a score dict
mapping WallType → float in [0, 1], representing evidence strength.

Reference: AGENTS.md §Wall Classifier Agent, TASKS.md CL-002..CL-004.
"""

from __future__ import annotations

from math import pi

import numpy as np

from docs.interfaces.classified_wall_graph import FireRating
from docs.interfaces.graph_to_serializer import FinalizedGraph, WallSegment, WallType
from src.classifier.taxonomy import (
    FIRE_RATING_BLUE_MAX,
    FIRE_RATING_GREEN_MAX,
    FIRE_RATING_RED_THRESHOLD,
    FIRE_RATING_THICKNESS_MAP,
    THICKNESS_MEDIUM_MAX,
    THICKNESS_MEDIUM_MIN,
    THICKNESS_THICK_MIN,
    THICKNESS_THIN_MAX,
)


# ---------------------------------------------------------------------------
# CL-002: Thickness-based classification
# ---------------------------------------------------------------------------


def score_by_thickness(segment: WallSegment) -> dict[WallType, float]:
    """Score wall type likelihood based on wall thickness.

    Thicker walls are more likely load-bearing or exterior.
    Thinner walls are more likely partitions.

    Args:
        segment: The wall segment to classify.

    Returns:
        Dict of WallType → evidence score in [0, 1].
    """
    t = segment.thickness

    if t <= THICKNESS_THIN_MAX:
        return {
            WallType.PARTITION: 0.8,
            WallType.LOAD_BEARING: 0.1,
            WallType.EXTERIOR: 0.0,
            WallType.SHEAR: 0.05,
            WallType.CURTAIN: 0.05,
        }

    if t <= THICKNESS_MEDIUM_MAX:
        # Ambiguous range — could be structural or partition.
        # Normalize position within range for smooth interpolation.
        frac = (t - THICKNESS_MEDIUM_MIN) / (THICKNESS_MEDIUM_MAX - THICKNESS_MEDIUM_MIN)
        # Coefficients chosen so frac-dependent terms cancel:
        # sum = (0.25+0.45+0.10+0.10+0.10) + frac*(0.15-0.30+0.15) = 1.0
        return {
            WallType.LOAD_BEARING: 0.25 + 0.15 * frac,
            WallType.PARTITION: 0.45 - 0.30 * frac,
            WallType.EXTERIOR: 0.10 + 0.15 * frac,
            WallType.SHEAR: 0.10,
            WallType.CURTAIN: 0.10,
        }

    # Thick walls — likely structural or exterior.
    return {
        WallType.LOAD_BEARING: 0.4,
        WallType.EXTERIOR: 0.35,
        WallType.SHEAR: 0.15,
        WallType.PARTITION: 0.05,
        WallType.CURTAIN: 0.05,
    }


# ---------------------------------------------------------------------------
# CL-003: Context-based classification (position + adjacency + length)
# ---------------------------------------------------------------------------


def identify_perimeter_edges(graph: FinalizedGraph) -> set[int]:
    """Identify wall segments that lie on the building perimeter.

    A wall is on the perimeter if both its endpoints are near the
    convex hull of all junction nodes. This is a geometric heuristic —
    it works well for rectangular floor plans and degrades gracefully
    for irregular shapes.

    Args:
        graph: The finalized graph.

    Returns:
        Set of edge_ids on the perimeter.
    """
    if graph.nodes.shape[0] < 3:
        return set()

    from scipy.spatial import ConvexHull

    try:
        hull = ConvexHull(graph.nodes)
    except Exception:
        return set()

    hull_tolerance = max(graph.page_width, graph.page_height) * 0.01

    # Build the set of hull edges (pairs of node indices that form hull boundary).
    hull_edge_set: set[tuple[int, int]] = set()
    for simplex in hull.simplices:
        a, b = int(simplex[0]), int(simplex[1])
        hull_edge_set.add((min(a, b), max(a, b)))

    # A wall is on the perimeter if its edge closely aligns with a hull edge.
    # Both endpoints must lie within tolerance of the same hull edge segment.
    perimeter_edges: set[int] = set()

    for seg in graph.wall_segments:
        sp = graph.nodes[seg.start_node]
        ep = graph.nodes[seg.end_node]

        for ha, hb in hull_edge_set:
            hp0 = graph.nodes[ha]
            hp1 = graph.nodes[hb]
            hull_vec = hp1 - hp0
            hull_len = np.linalg.norm(hull_vec)
            if hull_len < 1e-9:
                continue
            hull_dir = hull_vec / hull_len

            # Check both wall endpoints against this hull edge.
            both_on_edge = True
            for pt in (sp, ep):
                rel = pt - hp0
                proj = np.dot(rel, hull_dir)
                if proj < -hull_tolerance or proj > hull_len + hull_tolerance:
                    both_on_edge = False
                    break
                perp_dist = abs(rel[0] * hull_dir[1] - rel[1] * hull_dir[0])
                if perp_dist > hull_tolerance:
                    both_on_edge = False
                    break

            if both_on_edge:
                perimeter_edges.add(seg.edge_id)
                break

    return perimeter_edges


def score_by_position(
    segment: WallSegment,
    is_perimeter: bool,
) -> dict[WallType, float]:
    """Score wall type based on perimeter vs interior position.

    Perimeter walls are likely exterior or load-bearing.
    Interior walls are likely partitions or load-bearing.

    Args:
        segment: The wall segment.
        is_perimeter: Whether this wall is on the building perimeter.

    Returns:
        Dict of WallType → evidence score in [0, 1].
    """
    if is_perimeter:
        return {
            WallType.EXTERIOR: 0.6,
            WallType.LOAD_BEARING: 0.25,
            WallType.SHEAR: 0.1,
            WallType.PARTITION: 0.03,
            WallType.CURTAIN: 0.02,
        }

    return {
        WallType.PARTITION: 0.5,
        WallType.LOAD_BEARING: 0.3,
        WallType.SHEAR: 0.1,
        WallType.EXTERIOR: 0.0,
        WallType.CURTAIN: 0.0,
    }


def score_by_adjacency(
    segment: WallSegment,
    graph: FinalizedGraph,
) -> dict[WallType, float]:
    """Score wall type based on structural adjacency context.

    Walls connected to many other walls at junctions are more likely
    structural (load-bearing or shear). Walls with degree-1 endpoints
    (dead ends) are more likely partitions.

    Args:
        segment: The wall segment.
        graph: The finalized graph (for node degree calculation).

    Returns:
        Dict of WallType → evidence score in [0, 1].
    """
    # Compute node degrees from edge list.
    degree = np.zeros(graph.nodes.shape[0], dtype=int)
    for edge in graph.edges:
        degree[edge[0]] += 1
        degree[edge[1]] += 1

    start_deg = degree[segment.start_node]
    end_deg = degree[segment.end_node]
    avg_deg = (start_deg + end_deg) / 2.0

    # High-degree junctions (T-junctions, cross junctions) suggest structural walls.
    if avg_deg >= 3.0:
        return {
            WallType.LOAD_BEARING: 0.45,
            WallType.SHEAR: 0.15,
            WallType.EXTERIOR: 0.2,
            WallType.PARTITION: 0.15,
            WallType.CURTAIN: 0.05,
        }

    if avg_deg >= 2.0:
        return {
            WallType.LOAD_BEARING: 0.3,
            WallType.PARTITION: 0.3,
            WallType.EXTERIOR: 0.2,
            WallType.SHEAR: 0.1,
            WallType.CURTAIN: 0.1,
        }

    # Dead-end walls (degree 1) are likely partitions or curtain walls.
    return {
        WallType.PARTITION: 0.6,
        WallType.CURTAIN: 0.15,
        WallType.LOAD_BEARING: 0.1,
        WallType.SHEAR: 0.05,
        WallType.EXTERIOR: 0.1,
    }


def score_by_length(
    segment: WallSegment,
    graph: FinalizedGraph,
) -> dict[WallType, float]:
    """Score wall type based on wall length relative to the floor plan.

    Long walls spanning significant portions of the floor plan are more
    likely structural. Short walls are more likely partitions.

    Args:
        segment: The wall segment.
        graph: The finalized graph (for page dimensions).

    Returns:
        Dict of WallType → evidence score in [0, 1].
    """
    page_diag = (graph.page_width**2 + graph.page_height**2) ** 0.5
    if page_diag < 1e-9:
        return {t: 0.2 for t in WallType if t != WallType.UNKNOWN}

    length_ratio = segment.length / page_diag

    if length_ratio > 0.3:
        # Long wall — likely structural.
        return {
            WallType.LOAD_BEARING: 0.4,
            WallType.EXTERIOR: 0.3,
            WallType.SHEAR: 0.15,
            WallType.PARTITION: 0.1,
            WallType.CURTAIN: 0.05,
        }

    if length_ratio > 0.1:
        # Medium wall — ambiguous.
        return {
            WallType.LOAD_BEARING: 0.3,
            WallType.PARTITION: 0.3,
            WallType.EXTERIOR: 0.2,
            WallType.SHEAR: 0.1,
            WallType.CURTAIN: 0.1,
        }

    # Short wall — likely partition.
    return {
        WallType.PARTITION: 0.5,
        WallType.CURTAIN: 0.2,
        WallType.LOAD_BEARING: 0.15,
        WallType.SHEAR: 0.1,
        WallType.EXTERIOR: 0.05,
    }


# ---------------------------------------------------------------------------
# CL-004: Fire rating detection
# ---------------------------------------------------------------------------


def detect_fire_rating(
    segment: WallSegment,
    stroke_colors: np.ndarray | None = None,
    fill_colors: np.ndarray | None = None,
    edge_index: int | None = None,
) -> tuple[FireRating, float]:
    """Detect fire rating from color signals and thickness.

    Architectural conventions often use red or magenta fills/strokes
    to indicate fire-rated walls. Thick fire-rated walls with additional
    gypsum layers receive a higher rating.

    Args:
        segment: The wall segment.
        stroke_colors: Per-edge stroke colors from RawGraph, shape (E, 4).
        fill_colors: Per-edge fill colors from RawGraph, shape (E, 4) or None.
        edge_index: Index into the color arrays for this segment.

    Returns:
        Tuple of (fire_rating, confidence) where confidence is in [0, 1].
    """
    fire_signal = 0.0

    # Check stroke color.
    if stroke_colors is not None and edge_index is not None and edge_index < stroke_colors.shape[0]:
        color = stroke_colors[edge_index]
        if (
            color[0] >= FIRE_RATING_RED_THRESHOLD
            and color[1] <= FIRE_RATING_GREEN_MAX
            and color[2] <= FIRE_RATING_BLUE_MAX
        ):
            fire_signal = max(fire_signal, color[0])

    # Check fill color.
    if fill_colors is not None and edge_index is not None and edge_index < fill_colors.shape[0]:
        color = fill_colors[edge_index]
        if (
            color[0] >= FIRE_RATING_RED_THRESHOLD
            and color[1] <= FIRE_RATING_GREEN_MAX
            and color[2] <= FIRE_RATING_BLUE_MAX
        ):
            fire_signal = max(fire_signal, color[0])

    if fire_signal < FIRE_RATING_RED_THRESHOLD:
        return FireRating.NONE, 0.9

    # Infer rating level from thickness.
    for thickness_threshold, rating in FIRE_RATING_THICKNESS_MAP:
        if segment.thickness >= thickness_threshold:
            return rating, fire_signal

    return FireRating.HOUR_1, fire_signal
