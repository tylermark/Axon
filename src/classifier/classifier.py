"""Wall Classifier — combines rules into a classification pipeline.

Top-level classifier that chains thickness, position, adjacency, length,
and fire rating rules to produce a ClassifiedWallGraph from a FinalizedGraph.

Implements CL-005 (confidence scoring and human-review flagging).

Reference: AGENTS.md §Wall Classifier Agent, TASKS.md CL-005.
"""

from __future__ import annotations

import logging

import numpy as np

from docs.interfaces.classified_wall_graph import (
    ClassifiedWallGraph,
    FireRating,
    WallClassification,
)
from docs.interfaces.graph_to_serializer import FinalizedGraph, WallType
from src.classifier.rules import (
    detect_fire_rating,
    identify_perimeter_edges,
    score_by_adjacency,
    score_by_length,
    score_by_position,
    score_by_thickness,
)
from src.classifier.taxonomy import (
    DEFAULT_SIGNAL_WEIGHTS,
    HIGH_CONFIDENCE,
    LOW_CONFIDENCE,
    MEDIUM_CONFIDENCE,
    REVIEW_THRESHOLD,
    SignalWeights,
)

logger = logging.getLogger(__name__)


# All wall types the classifier considers (excludes UNKNOWN).
_ACTIVE_TYPES = [
    WallType.LOAD_BEARING,
    WallType.PARTITION,
    WallType.EXTERIOR,
    WallType.SHEAR,
    WallType.CURTAIN,
]


def _combine_scores(
    score_dicts: list[tuple[str, dict[WallType, float], float]],
) -> tuple[WallType, float, dict[str, float]]:
    """Combine multiple scored evidence sources into a final classification.

    Uses weighted averaging of per-type scores across all signal sources,
    then selects the type with highest combined score. Confidence is derived
    from the margin between the top two candidates.

    Args:
        score_dicts: List of (signal_name, type_scores, weight) tuples.
            type_scores maps WallType → evidence in [0, 1].
            weight is the relative importance of this signal.

    Returns:
        Tuple of (best_type, confidence, signal_contributions).
    """
    combined: dict[WallType, float] = {t: 0.0 for t in _ACTIVE_TYPES}
    total_weight = sum(w for _, _, w in score_dicts)
    signals: dict[str, float] = {}

    if total_weight < 1e-9:
        return WallType.UNKNOWN, 0.0, signals

    for signal_name, scores, weight in score_dicts:
        normalized_weight = weight / total_weight
        # Track which type this signal most supported.
        best_signal_type = max(scores, key=scores.get, default=WallType.UNKNOWN)
        signals[f"{signal_name}_score"] = scores.get(best_signal_type, 0.0)

        for wall_type in _ACTIVE_TYPES:
            combined[wall_type] += scores.get(wall_type, 0.0) * normalized_weight

    # Select best type.
    sorted_types = sorted(combined.items(), key=lambda x: x[1], reverse=True)
    best_type = sorted_types[0][0]
    best_score = sorted_types[0][1]
    second_score = sorted_types[1][1] if len(sorted_types) > 1 else 0.0

    # Confidence from margin: large margin → high confidence.
    margin = best_score - second_score
    if margin > 0.25:
        confidence = HIGH_CONFIDENCE
    elif margin > 0.10:
        confidence = MEDIUM_CONFIDENCE
    else:
        confidence = LOW_CONFIDENCE

    # Scale by absolute score — if best score is low, reduce confidence.
    if best_score < 0.3:
        confidence *= 0.7

    return best_type, confidence, signals


def classify_wall_graph(
    graph: FinalizedGraph,
    stroke_colors: np.ndarray | None = None,
    fill_colors: np.ndarray | None = None,
    weights: SignalWeights | None = None,
    review_threshold: float = REVIEW_THRESHOLD,
) -> ClassifiedWallGraph:
    """Classify all walls in a FinalizedGraph.

    Runs the full classification pipeline:
    1. Identify perimeter edges (convex hull heuristic)
    2. For each wall segment, compute evidence from:
       - Thickness (CL-002)
       - Position — perimeter vs interior (CL-003)
       - Adjacency context — junction degree (CL-003)
       - Length — relative to floor plan (CL-003)
       - Color — fire rating detection (CL-004)
    3. Combine evidence with weighted averaging (CL-005)
    4. Flag low-confidence walls for human review (CL-005)

    Args:
        graph: FinalizedGraph from Layer 1 pipeline.
        stroke_colors: Per-edge stroke colors from the original RawGraph,
            shape (E, 4) float64 RGBA. Used for fire rating detection.
            None if color data is unavailable.
        fill_colors: Per-edge fill colors, shape (E, 4) or None.
        weights: Signal weights for combining evidence. Uses defaults
            if None.
        review_threshold: Confidence threshold below which walls are
            flagged for human review.

    Returns:
        ClassifiedWallGraph with per-wall classifications.
    """
    if weights is None:
        weights = DEFAULT_SIGNAL_WEIGHTS

    perimeter_edges = identify_perimeter_edges(graph)
    logger.info(
        "Identified %d/%d walls as perimeter",
        len(perimeter_edges),
        len(graph.wall_segments),
    )

    classifications: list[WallClassification] = []
    flagged: list[int] = []
    type_counts: dict[str, int] = {}

    for seg in graph.wall_segments:
        is_perimeter = seg.edge_id in perimeter_edges

        # Gather evidence from each rule.
        thickness_scores = score_by_thickness(seg)
        position_scores = score_by_position(seg, is_perimeter)
        adjacency_scores = score_by_adjacency(seg, graph)
        length_scores = score_by_length(seg, graph)

        # Combine all signals.
        wall_type, confidence, signals = _combine_scores([
            ("thickness", thickness_scores, weights.thickness),
            ("position", position_scores, weights.position),
            ("adjacency", adjacency_scores, weights.adjacency),
            ("length", length_scores, weights.length),
        ])

        # Fire rating detection (CL-004).
        fire_rating, fire_confidence = detect_fire_rating(
            seg,
            stroke_colors=stroke_colors,
            fill_colors=fill_colors,
            edge_index=seg.edge_id,
        )

        # If fire-rated, boost confidence slightly — fire color is a strong signal.
        if fire_rating != FireRating.NONE:
            signals["color_score"] = fire_confidence

        classification = WallClassification(
            edge_id=seg.edge_id,
            wall_type=wall_type,
            fire_rating=fire_rating,
            confidence=confidence,
            signals=signals,
            is_perimeter=is_perimeter,
        )
        classifications.append(classification)

        # Track flagged walls.
        if confidence < review_threshold:
            flagged.append(seg.edge_id)

        # Update type counts.
        type_counts[wall_type.value] = type_counts.get(wall_type.value, 0) + 1

    logger.info(
        "Classification complete: %s, %d flagged for review",
        type_counts,
        len(flagged),
    )

    return ClassifiedWallGraph(
        graph=graph,
        classifications=classifications,
        review_threshold=review_threshold,
        walls_flagged_for_review=flagged,
        classification_summary=type_counts,
        perimeter_edge_ids=sorted(perimeter_edges),
    )