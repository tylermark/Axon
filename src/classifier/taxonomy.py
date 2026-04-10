"""Wall classification taxonomy — constants, thresholds, and signal types.

Defines the classification parameters used by the rule-based classifier
to assign wall types. Thresholds are derived from architectural drawing
conventions and Capsule Manufacturing's CFS product constraints.

Reference: AGENTS.md §Wall Classifier Agent, TASKS.md CL-001.
"""

from __future__ import annotations

from dataclasses import dataclass

from docs.interfaces.classified_wall_graph import FireRating
from docs.interfaces.graph_to_serializer import WallType

# ---------------------------------------------------------------------------
# Thickness thresholds (PDF user units, 72 units/inch)
# ---------------------------------------------------------------------------

# Typical architectural drawing conventions:
#   Exterior / load-bearing walls: drawn thicker (6-12+ pts)
#   Interior partitions: drawn thinner (2-4 pts)
#   Shear walls: similar to load-bearing, sometimes hatched
#
# These are heuristic ranges — actual values depend on the drawing standard
# and scale. The classifier uses these as soft priors, not hard rules.

THICKNESS_THIN_MAX = 4.0
"""Walls with thickness <= this are likely partitions."""

THICKNESS_MEDIUM_MIN = 4.0
THICKNESS_MEDIUM_MAX = 8.0
"""Walls in this range could be load-bearing or exterior."""

THICKNESS_THICK_MIN = 8.0
"""Walls with thickness >= this are likely exterior or load-bearing."""


# ---------------------------------------------------------------------------
# Color thresholds for fire rating detection
# ---------------------------------------------------------------------------

FIRE_RATING_RED_THRESHOLD = 0.6
"""Minimum red channel value (with low green/blue) to flag as fire-rated.
Many architectural conventions use red or magenta fill for fire walls."""

FIRE_RATING_GREEN_MAX = 0.3
"""Maximum green channel for fire-rated color detection."""

FIRE_RATING_BLUE_MAX = 0.3
"""Maximum blue channel for fire-rated color detection."""


# ---------------------------------------------------------------------------
# Confidence thresholds
# ---------------------------------------------------------------------------

REVIEW_THRESHOLD = 0.7
"""Walls with classification confidence below this are flagged for human review."""

HIGH_CONFIDENCE = 0.9
"""Confidence assigned when multiple strong signals agree."""

MEDIUM_CONFIDENCE = 0.75
"""Confidence assigned when one strong signal dominates."""

LOW_CONFIDENCE = 0.5
"""Confidence assigned when signals are ambiguous or conflicting."""


# ---------------------------------------------------------------------------
# Signal weights for combining classification evidence
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SignalWeights:
    """Relative importance of each classification signal.

    Used to compute a weighted confidence score when combining
    multiple evidence sources.
    """

    thickness: float = 0.30
    """Weight for thickness-based signal."""

    position: float = 0.30
    """Weight for perimeter/interior position signal."""

    adjacency: float = 0.15
    """Weight for structural adjacency context signal."""

    color: float = 0.15
    """Weight for stroke/fill color signal."""

    length: float = 0.10
    """Weight for wall length signal (long walls more likely structural)."""


DEFAULT_SIGNAL_WEIGHTS = SignalWeights()


# ---------------------------------------------------------------------------
# Wall type priors by position
# ---------------------------------------------------------------------------

PERIMETER_TYPE_PRIOR: dict[WallType, float] = {
    WallType.EXTERIOR: 0.6,
    WallType.LOAD_BEARING: 0.3,
    WallType.SHEAR: 0.05,
    WallType.PARTITION: 0.03,
    WallType.CURTAIN: 0.02,
}
"""Prior probabilities for perimeter walls."""

INTERIOR_TYPE_PRIOR: dict[WallType, float] = {
    WallType.PARTITION: 0.5,
    WallType.LOAD_BEARING: 0.25,
    WallType.SHEAR: 0.1,
    WallType.EXTERIOR: 0.0,
    WallType.CURTAIN: 0.0,
}
"""Prior probabilities for interior walls. Exterior/curtain impossible."""


# ---------------------------------------------------------------------------
# Fire rating inference from thickness
# ---------------------------------------------------------------------------

FIRE_RATING_THICKNESS_MAP: list[tuple[float, FireRating]] = [
    (10.0, FireRating.HOUR_2),
    (7.0, FireRating.HOUR_1),
]
"""Thick fire-rated walls often have additional gypsum layers.
If a wall is flagged as fire-rated by color AND exceeds these
thickness thresholds, a higher fire rating is inferred."""