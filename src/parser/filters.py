"""Decorative element flagging heuristics for raw graph G₀.

Computes per-edge ``confidence_wall`` scores indicating how likely each
edge is to be a structural wall vs. a decorative element (hatching,
dimension lines, furniture symbols, annotations). Edges are **flagged,
not removed** — the Tokenizer Agent makes the final semantic decision.

Tasks: P-007.
Reference: ARCHITECTURE.md §Stage 1, MODEL_SPEC.md §PDF Vector Primitives.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from docs.interfaces.parser_to_tokenizer import RawGraph
    from src.pipeline.config import ParserConfig

logger = logging.getLogger(__name__)

# Default weights for composite scoring.
DEFAULT_WEIGHTS: dict[str, float] = {
    "stroke_width": 0.30,
    "color": 0.15,
    "dash_pattern": 0.15,
    "geometric_regularity": 0.15,
    "hatching": 0.15,
    "edge_length": 0.10,
}


# ---------------------------------------------------------------------------
# Individual Heuristic Scoring Functions
# ---------------------------------------------------------------------------


def score_stroke_width(graph: RawGraph, config: ParserConfig | None = None) -> np.ndarray:
    """Score edges by stroke width proximity to typical wall widths.

    Edges within ``config.wall_stroke_width_range`` receive score 1.0.
    Scores decay smoothly outside this range using a Gaussian falloff.

    Args:
        graph: Raw spatial graph with per-edge stroke widths.
        config: Parser config providing wall_stroke_width_range.

    Returns:
        Per-edge scores, shape (E,) in [0, 1].
    """
    if config is None:
        from src.pipeline.config import ParserConfig as _ParserConfig

        config = _ParserConfig()

    widths = graph.stroke_widths
    num_edges = len(widths)
    if num_edges == 0:
        return np.empty(0, dtype=np.float64)

    lo, hi = config.wall_stroke_width_range
    scores = np.ones(num_edges, dtype=np.float64)

    # Gaussian decay outside the optimal range (sigma = 1.0 PDF units)
    sigma = 1.0
    below = widths < lo
    above = widths > hi
    scores[below] = np.exp(-0.5 * ((widths[below] - lo) / sigma) ** 2)
    scores[above] = np.exp(-0.5 * ((widths[above] - hi) / sigma) ** 2)

    # Hairlines get near-zero score
    hairline = widths < 0.1
    scores[hairline] = np.clip(widths[hairline] / 0.1, 0.0, 1.0) * 0.1

    return scores


def score_color(graph: RawGraph) -> np.ndarray:
    """Score edges by stroke color — black is most structural.

    Uses color saturation (max(RGB) - min(RGB)) and luminance as signals.
    Low-saturation dark strokes score highest.

    Args:
        graph: Raw spatial graph with per-edge RGBA stroke colors.

    Returns:
        Per-edge scores, shape (E,) in [0, 1].
    """
    colors = graph.stroke_colors
    num_edges = len(colors)
    if num_edges == 0:
        return np.empty(0, dtype=np.float64)

    rgb = colors[:, :3]  # (E, 3)

    # Luminance: darker = more likely structural
    luminance = 0.299 * rgb[:, 0] + 0.587 * rgb[:, 1] + 0.114 * rgb[:, 2]
    darkness_score = 1.0 - luminance  # black → 1.0, white → 0.0

    # Saturation: low saturation = gray/black = likely structural
    saturation = np.max(rgb, axis=1) - np.min(rgb, axis=1)
    saturation_score = 1.0 - saturation  # unsaturated → 1.0

    # Combine: both dark and unsaturated → high confidence
    scores = 0.6 * darkness_score + 0.4 * saturation_score

    return np.clip(scores, 0.0, 1.0)


def score_dash_pattern(graph: RawGraph) -> np.ndarray:
    """Score edges by dash pattern — solid lines are most structural.

    Solid lines (empty dash array) receive 1.0. Dashed lines receive
    reduced scores, with longer dash segments scoring slightly higher
    than short dots.

    Args:
        graph: Raw spatial graph with per-edge dash patterns.

    Returns:
        Per-edge scores, shape (E,) in [0, 1].
    """
    patterns = graph.dash_patterns
    num_edges = len(patterns)
    if num_edges == 0:
        return np.empty(0, dtype=np.float64)

    scores = np.ones(num_edges, dtype=np.float64)

    for i, (dash_array, _phase) in enumerate(patterns):
        if not dash_array:
            # Solid line
            continue
        # Average dash segment length — longer dashes score higher
        avg_dash = sum(dash_array) / len(dash_array)
        # Score based on average dash length: very short dots → ~0.2, long dashes → ~0.5
        scores[i] = np.clip(0.2 + 0.3 * (avg_dash / 10.0), 0.1, 0.5)

    return scores


def score_geometric_regularity(graph: RawGraph) -> np.ndarray:
    """Score edges by geometric regularity — axis-aligned edges score higher.

    Walls in architectural plans are predominantly horizontal or vertical.
    Edges aligned to cardinal axes (within tolerance) receive bonuses.

    Args:
        graph: Raw spatial graph with node coordinates and edges.

    Returns:
        Per-edge scores, shape (E,) in [0, 1].
    """
    edges = graph.edges
    nodes = graph.nodes
    num_edges = len(edges)
    if num_edges == 0:
        return np.empty(0, dtype=np.float64)

    starts = nodes[edges[:, 0]]  # (E, 2)
    ends = nodes[edges[:, 1]]  # (E, 2)
    deltas = ends - starts  # (E, 2)

    # Edge angles relative to horizontal (0 to π)
    angles = np.abs(np.arctan2(deltas[:, 1], deltas[:, 0]))  # [0, π]

    # Distance from nearest cardinal axis (0°, 90°, 180°)
    angle_to_h = np.minimum(angles, np.pi - angles)  # distance to 0° or 180°
    angle_to_v = np.abs(angles - np.pi / 2)  # distance to 90°
    min_axis_dist = np.minimum(angle_to_h, angle_to_v)  # radians from nearest axis

    # Axis alignment score: 0 radians → 1.0, π/4 (45°) → ~0.3
    tolerance_rad = np.radians(5.0)
    axis_score = np.exp(-0.5 * (min_axis_dist / tolerance_rad) ** 2)
    # Don't penalize diagonal edges too harshly — floor at 0.3
    axis_score = np.clip(axis_score, 0.3, 1.0)

    # Edge lengths (longer is more likely a wall)
    lengths = np.sqrt(deltas[:, 0] ** 2 + deltas[:, 1] ** 2)
    min_wall_length = 10.0  # PDF units
    length_score = np.clip(lengths / min_wall_length, 0.0, 1.0)

    scores = 0.6 * axis_score + 0.4 * length_score

    return np.clip(scores, 0.0, 1.0)


def score_hatching_detector(graph: RawGraph) -> np.ndarray:
    """Score edges by hatching pattern membership — hatching lines score low.

    Detects groups of closely-spaced parallel lines (hatching) by checking
    for clusters of edges with similar angles and uniform spacing. Edges
    belonging to hatching groups receive low scores.

    This uses a spatial bucketing approach: edges are binned by quantized
    angle, then within each bin, spatial proximity and parallelism are
    checked to identify hatching clusters.

    Args:
        graph: Raw spatial graph.

    Returns:
        Per-edge scores, shape (E,) in [0, 1]. Low = likely hatching.
    """
    edges = graph.edges
    nodes = graph.nodes
    num_edges = len(edges)
    if num_edges == 0:
        return np.empty(0, dtype=np.float64)

    # Start with all edges as non-hatching (score 1.0)
    scores = np.ones(num_edges, dtype=np.float64)

    starts = nodes[edges[:, 0]]
    ends = nodes[edges[:, 1]]
    deltas = ends - starts

    # Edge midpoints, angles, and lengths
    midpoints = (starts + ends) / 2.0
    angles = np.arctan2(deltas[:, 1], deltas[:, 0])  # [-π, π]
    # Normalize to [0, π) — parallel lines in opposite directions are equivalent
    angles = angles % np.pi
    lengths = np.sqrt(deltas[:, 0] ** 2 + deltas[:, 1] ** 2)

    # Skip very short edges
    min_hatch_len = 3.0
    candidate_mask = lengths >= min_hatch_len

    if np.sum(candidate_mask) < 4:
        return scores

    candidate_indices = np.nonzero(candidate_mask)[0]
    cand_angles = angles[candidate_indices]
    cand_midpoints = midpoints[candidate_indices]

    # Quantize angles into bins (~2° wide)
    angle_bin_width = np.radians(2.0)
    angle_bins = (cand_angles / angle_bin_width).astype(np.int64)

    # Process each angle bin
    unique_bins = np.unique(angle_bins)

    # Direction unit vectors for perpendicular distance computation
    cand_dirs = deltas[candidate_indices]
    cand_dir_norms = lengths[candidate_indices]
    # Normalized direction vectors
    safe_norms = np.where(cand_dir_norms > 1e-12, cand_dir_norms, 1.0)
    cand_unit_dirs = cand_dirs / safe_norms[:, np.newaxis]

    for angle_bin in unique_bins:
        bin_mask = angle_bins == angle_bin
        bin_local_indices = np.nonzero(bin_mask)[0]

        if len(bin_local_indices) < 4:
            continue

        # Compute perpendicular distances between all pairs in this bin
        # using the average direction of the bin
        avg_dir = np.mean(cand_unit_dirs[bin_local_indices], axis=0)
        avg_dir_norm = np.sqrt(avg_dir[0] ** 2 + avg_dir[1] ** 2)
        if avg_dir_norm < 1e-12:
            continue
        avg_dir = avg_dir / avg_dir_norm

        # Perpendicular direction
        perp = np.array([-avg_dir[1], avg_dir[0]])

        # Project midpoints onto perpendicular axis
        mids = cand_midpoints[bin_local_indices]
        perp_dists = mids @ perp  # scalar projection

        # Sort by perpendicular distance and check for uniform spacing
        sort_order = np.argsort(perp_dists)
        sorted_dists = perp_dists[sort_order]
        spacings = np.diff(sorted_dists)

        if len(spacings) < 3:
            continue

        # Detect uniform spacing: low coefficient of variation
        median_spacing = np.median(spacings)
        if median_spacing < 0.5:  # Extremely close — likely same line or noise
            continue

        # Edges with spacing close to median are part of hatching
        spacing_tolerance = max(median_spacing * 0.4, 1.0)
        uniform_mask = np.abs(spacings - median_spacing) < spacing_tolerance

        # At least 3 consecutive uniform spacings → hatching
        consecutive = 0
        hatch_set: set[int] = set()
        for k, is_uniform in enumerate(uniform_mask):
            if is_uniform:
                consecutive += 1
                if consecutive >= 3:
                    # Mark all edges in this consecutive run
                    for j in range(k - consecutive + 1, k + 2):
                        hatch_set.add(sort_order[j])
            else:
                consecutive = 0

        # Apply low score to detected hatching edges
        for local_idx in hatch_set:
            global_idx = candidate_indices[bin_local_indices[local_idx]]
            scores[global_idx] = 0.1

    return scores


def score_edge_length(graph: RawGraph) -> np.ndarray:
    """Score edges by length — very short edges are likely noise.

    Normalizes edge length by page diagonal for scale invariance.
    Very short edges (< 2 PDF units) get low scores.

    Args:
        graph: Raw spatial graph.

    Returns:
        Per-edge scores, shape (E,) in [0, 1].
    """
    edges = graph.edges
    nodes = graph.nodes
    num_edges = len(edges)
    if num_edges == 0:
        return np.empty(0, dtype=np.float64)

    starts = nodes[edges[:, 0]]
    ends = nodes[edges[:, 1]]
    deltas = ends - starts
    lengths = np.sqrt(deltas[:, 0] ** 2 + deltas[:, 1] ** 2)

    # Page diagonal for normalization
    page_diag = np.sqrt(graph.page_width**2 + graph.page_height**2)
    if page_diag < 1e-12:
        page_diag = 1.0

    # Very short edges → low score, medium+ edges → high score
    # Sigmoid-like: score = 1 - exp(-length / scale)
    noise_threshold = 2.0  # PDF units
    scale = max(page_diag * 0.01, noise_threshold)

    scores = 1.0 - np.exp(-lengths / scale)

    # Hard penalty for extremely short edges
    scores[lengths < noise_threshold] *= lengths[lengths < noise_threshold] / noise_threshold

    return np.clip(scores, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Composite Scoring
# ---------------------------------------------------------------------------


def compute_wall_confidence(
    graph: RawGraph,
    config: ParserConfig | None = None,
    weights: dict[str, float] | None = None,
) -> np.ndarray:
    """Compute composite wall confidence scores for all edges.

    Combines individual heuristic scores via weighted average.

    Args:
        graph: Raw spatial graph.
        config: Parser configuration. Uses defaults if None.
        weights: Per-heuristic weights. Uses DEFAULT_WEIGHTS if None.

    Returns:
        Per-edge confidence scores, shape (E,) in [0, 1].
    """
    if config is None:
        from src.pipeline.config import ParserConfig as _ParserConfig

        config = _ParserConfig()

    if weights is None:
        weights = DEFAULT_WEIGHTS

    num_edges = len(graph.edges)
    if num_edges == 0:
        return np.empty(0, dtype=np.float64)

    # Compute all heuristic scores
    heuristics: dict[str, np.ndarray] = {
        "stroke_width": score_stroke_width(graph, config),
        "color": score_color(graph),
        "dash_pattern": score_dash_pattern(graph),
        "geometric_regularity": score_geometric_regularity(graph),
        "hatching": score_hatching_detector(graph),
        "edge_length": score_edge_length(graph),
    }

    # Weighted average
    total_weight = 0.0
    composite = np.zeros(num_edges, dtype=np.float64)

    for name, score_array in heuristics.items():
        w = weights.get(name, 0.0)
        composite += w * score_array
        total_weight += w

    if total_weight > 0.0:
        composite /= total_weight

    return np.clip(composite, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def apply_filters(graph: RawGraph, config: ParserConfig | None = None) -> RawGraph:
    """Compute wall confidence scores and attach them to the graph.

    This is the primary entry point called by the pipeline. It populates
    ``graph.confidence_wall`` with per-edge heuristic scores.

    Args:
        graph: Raw spatial graph from ``build_raw_graph()``.
        config: Parser configuration. Uses defaults if None.

    Returns:
        The same ``RawGraph`` with ``confidence_wall`` populated.
    """
    graph.confidence_wall = compute_wall_confidence(graph, config)

    num_edges = len(graph.edges)
    if num_edges > 0:
        mean_conf = float(np.mean(graph.confidence_wall))
        high_conf = int(np.sum(graph.confidence_wall >= 0.7))
        logger.info(
            "Wall confidence: mean=%.3f, high-confidence edges=%d/%d (%.1f%%)",
            mean_conf,
            high_conf,
            num_edges,
            100.0 * high_conf / num_edges,
        )

    return graph
