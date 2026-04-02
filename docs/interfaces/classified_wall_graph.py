"""Interface contract: Layer 1 Pipeline → Wall Classifier → DRL / BIM Transplant.

Defines the ClassifiedWallGraph dataclass — the output of the Wall Classifier
Agent (Phase 7) and input to the DRL Agent (Phase 8) and BIM Transplant Agent
(Phase 10).

Extends the FinalizedGraph from Layer 1 with per-wall structural classification,
fire rating, and classification confidence. Wall types are populated (no longer
UNKNOWN) and low-confidence classifications are flagged for human review.

Reference: AGENTS.md §Wall Classifier Agent, TASKS.md §Phase 7.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from docs.interfaces.graph_to_serializer import (
    FinalizedGraph,
    Opening,
    Room,
    WallSegment,
    WallType,
)


class FireRating(str, Enum):
    """Fire resistance rating per building code."""

    NONE = "none"
    """No fire rating required."""

    HOUR_1 = "1_hour"
    """1-hour fire-resistance rating."""

    HOUR_2 = "2_hour"
    """2-hour fire-resistance rating."""

    HOUR_3 = "3_hour"
    """3-hour fire-resistance rating."""

    HOUR_4 = "4_hour"
    """4-hour fire-resistance rating."""

    UNKNOWN = "unknown"
    """Fire rating not determined."""


@dataclass
class WallClassification:
    """Classification result for a single wall segment.

    Produced by the classifier for each WallSegment in the FinalizedGraph.
    Contains the assigned type, fire rating, confidence, and the signals
    that drove the decision — enabling human review of flagged walls.
    """

    edge_id: int
    """Index matching the WallSegment.edge_id in the source FinalizedGraph."""

    wall_type: WallType
    """Assigned structural classification."""

    fire_rating: FireRating
    """Assigned fire resistance rating."""

    confidence: float
    """Classification confidence in [0, 1].

    Below the review threshold (default 0.7), this wall is flagged for
    human review in the ClassifiedWallGraph.walls_flagged_for_review list.
    """

    signals: dict[str, float] = field(default_factory=dict)
    """Signals that contributed to this classification.

    Keys may include:
    - 'thickness_score': how strongly thickness indicates the assigned type
    - 'position_score': perimeter vs interior position signal
    - 'adjacency_score': structural context from neighboring walls
    - 'color_score': fill/stroke color signal (e.g., red → fire-rated)
    - 'label_score': text annotation signal if detected
    """

    is_perimeter: bool = False
    """True if this wall lies on the exterior boundary of the floor plan."""

    override: WallType | None = None
    """Human-provided override. When set, takes precedence over the
    classifier's wall_type assignment. None means no override applied."""


@dataclass
class ClassifiedWallGraph:
    """Structural graph with per-wall classification and fire rating.

    This is the primary output of the Wall Classifier Agent and the
    input to the DRL Agent (panelization/placement) and downstream
    modules. It wraps the Layer 1 FinalizedGraph and adds classification
    metadata without modifying the underlying geometry.

    Reference: AGENTS.md §Wall Classifier Agent, TASKS.md §Phase 7.
    """

    graph: FinalizedGraph
    """The source FinalizedGraph from Layer 1. Geometry is unchanged."""

    classifications: list[WallClassification]
    """Per-wall classification results, one per WallSegment.

    Ordered to match graph.wall_segments — classifications[i] corresponds
    to graph.wall_segments[i].
    """

    review_threshold: float = 0.7
    """Confidence threshold below which walls are flagged for review."""

    walls_flagged_for_review: list[int] = field(default_factory=list)
    """Edge IDs of walls with confidence below review_threshold.

    These walls should be presented to a human for verification before
    proceeding to DRL panelization.
    """

    classification_summary: dict[str, int] = field(default_factory=dict)
    """Count of walls by type, e.g. {'load_bearing': 12, 'partition': 8, ...}."""

    perimeter_edge_ids: list[int] = field(default_factory=list)
    """Edge IDs identified as part of the building perimeter."""

    metadata: dict[str, object] = field(default_factory=dict)
    """Additional metadata (classifier version, processing time, etc.)."""
