"""Wall Classifier — Phase 7 of the Axon pipeline.

Classifies wall segments from the Layer 1 FinalizedGraph into structural
types (load-bearing, partition, exterior, shear, curtain) with fire rating
detection and confidence scoring.
"""

from src.classifier.classifier import classify_wall_graph
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

__all__ = [
    "DEFAULT_SIGNAL_WEIGHTS",
    "HIGH_CONFIDENCE",
    "LOW_CONFIDENCE",
    "MEDIUM_CONFIDENCE",
    "REVIEW_THRESHOLD",
    "SignalWeights",
    "classify_wall_graph",
    "detect_fire_rating",
    "identify_perimeter_edges",
    "score_by_adjacency",
    "score_by_length",
    "score_by_position",
    "score_by_thickness",
]
