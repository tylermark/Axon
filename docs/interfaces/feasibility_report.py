"""Interface contract: DRL Agent → Feasibility Agent → Pipeline Output.

Defines the FeasibilityReport dataclass — the output of the Feasibility Agent
(Phase 9, tasks FS-001 through FS-004) and input to the final pipeline output
(Phase 12).

The Feasibility Agent receives a PanelizationResult (containing PanelMap,
PlacementMap, ClassifiedWallGraph, and summary statistics) and produces a
comprehensive feasibility assessment:

    - Prefab coverage metrics (by wall length, area, cost)
    - Per-wall feasibility breakdown (panelizable or not, with reasons)
    - Per-room feasibility breakdown (pod-eligible or not, with reasons)
    - Blocker identification and categorization
    - Design modification suggestions that increase prefab percentage
    - Per-floor and whole-project scoring

This contract intentionally excludes cost estimates — that is the BOM Agent's
domain (see ``bill_of_materials.py``).

Reference: AGENTS.md §Feasibility Agent, TASKS.md §Phase 9.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from docs.interfaces.drl_output import PanelizationResult


# ══════════════════════════════════════════════════════════════════════════════
# 1. Enums
# ══════════════════════════════════════════════════════════════════════════════


class BlockerCategory(str, Enum):
    """Categories of blockers that prevent prefabrication.

    Used to classify why a wall or room cannot be handled by Capsule's
    standard product catalog.
    """

    GEOMETRY = "geometry"
    """Non-standard geometry (curved walls, acute angles, non-orthogonal
    intersections) that cannot be fabricated with standard CFS panels."""

    MACHINE_LIMITS = "machine_limits"
    """Wall dimensions exceed fabrication machine capabilities (max length,
    max gauge, max web depth, coil width range)."""

    CODE_CONSTRAINT = "code_constraint"
    """Building code requirement that cannot be met by available products
    (e.g., required fire rating not available for the needed gauge/depth)."""

    CLEARANCE = "clearance"
    """Insufficient clearance for pod placement or panel installation
    (room too small, obstructions, inadequate margins)."""

    OPENING_CONFLICT = "opening_conflict"
    """Door/window openings conflict with panel segmentation or pod
    placement (opening spans a splice point, pod blocks egress, etc.)."""

    PRODUCT_GAP = "product_gap"
    """No product in the Knowledge Graph catalog matches the required
    specification (missing SKU, discontinued product, etc.)."""


class SuggestionType(str, Enum):
    """Types of design modification suggestions."""

    WALL_STRAIGHTEN = "wall_straighten"
    """Straighten a non-orthogonal wall to enable standard panel usage."""

    WALL_EXTEND = "wall_extend"
    """Extend a wall to meet minimum panel length requirements."""

    WALL_SHORTEN = "wall_shorten"
    """Shorten a wall to fit within machine fabrication limits."""

    ROOM_RESIZE = "room_resize"
    """Resize a room to accommodate a pod product (widen, deepen, etc.)."""

    ROOM_REFUNCTION = "room_refunction"
    """Change room function assignment to match an available pod type."""

    WALL_RECLASSIFY = "wall_reclassify"
    """Reclassify a wall type to unlock additional panel options."""

    OPENING_RELOCATE = "opening_relocate"
    """Relocate a door or window to avoid splice-point conflicts."""

    CUSTOM = "custom"
    """Free-form suggestion that doesn't fit standard categories."""


# ══════════════════════════════════════════════════════════════════════════════
# 2. Blockers
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class Blocker:
    """A single issue that prevents prefabrication of a wall or room.

    Blockers are identified by the Feasibility Agent and categorized to
    help the user understand what must change — in the design or in the
    product catalog — to increase prefab coverage.
    """

    blocker_id: str
    """Unique identifier for this blocker, e.g. 'BLK-001'."""

    category: BlockerCategory
    """Classification of the blocker type."""

    description: str
    """Human-readable description of the issue.

    Example: 'Wall #12 is 396 inches long, exceeding the Howick 3.5
    max fabrication length of 360 inches.'
    """

    affected_edge_ids: list[int] = field(default_factory=list)
    """Edge IDs of walls affected by this blocker.

    Empty if the blocker applies to a room rather than specific walls.
    """

    affected_room_ids: list[int] = field(default_factory=list)
    """Room IDs affected by this blocker.

    Empty if the blocker applies to walls rather than rooms.
    """

    severity: float = 1.0
    """Severity score in [0, 1].

    1.0 = hard blocker (physically impossible to prefabricate).
    Lower values indicate soft blockers that reduce efficiency but
    don't completely prevent prefabrication.
    """


# ══════════════════════════════════════════════════════════════════════════════
# 3. Design Modification Suggestions
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class DesignSuggestion:
    """A recommended design change to increase prefab coverage.

    Each suggestion is tied to one or more blockers and quantifies
    the expected improvement if the change is adopted.
    """

    suggestion_id: str
    """Unique identifier, e.g. 'SUG-001'."""

    suggestion_type: SuggestionType
    """Category of the suggested modification."""

    description: str
    """Human-readable description of the recommended change.

    Example: 'Straighten wall #5 (currently 7 degrees off-axis) to
    enable standard panel coverage, adding approximately 12 panels.'
    """

    resolves_blocker_ids: list[str] = field(default_factory=list)
    """Blocker IDs that this suggestion would fully or partially resolve."""

    affected_edge_ids: list[int] = field(default_factory=list)
    """Edge IDs of walls affected by the suggested change."""

    affected_room_ids: list[int] = field(default_factory=list)
    """Room IDs affected by the suggested change."""

    estimated_coverage_gain_pct: float = 0.0
    """Estimated increase in prefab coverage percentage if adopted.

    Example: 3.5 means adopting this suggestion would raise coverage
    from, say, 78% to approximately 81.5%.
    """

    estimated_panels_gained: int = 0
    """Estimated number of additional panels that become placeable."""

    estimated_pods_gained: int = 0
    """Estimated number of additional pods that become placeable."""

    effort_level: str = ""
    """Qualitative effort indicator: 'low', 'medium', 'high'.

    - low: minor dimension adjustment, no structural impact
    - medium: requires architect review, possible code re-check
    - high: significant redesign, structural implications
    """


# ══════════════════════════════════════════════════════════════════════════════
# 4. Per-Wall Feasibility
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class WallFeasibility:
    """Feasibility assessment for a single wall segment.

    Summarizes whether this wall can be panelized and, if not, why.
    """

    edge_id: int
    """Edge index matching ``WallSegment.edge_id`` in the source graph."""

    is_panelizable: bool
    """True if the DRL Agent successfully assigned panels to this wall."""

    wall_length_inches: float
    """Total wall length in inches."""

    panelized_length_inches: float = 0.0
    """Length of this wall covered by panels, in inches.

    Equals ``wall_length_inches`` when fully panelized.
    """

    coverage_pct: float = 0.0
    """Percentage of this wall's length that is panelized.

    ``(panelized_length_inches / wall_length_inches) * 100``.
    """

    blocker_ids: list[str] = field(default_factory=list)
    """IDs of blockers affecting this wall (references ``Blocker.blocker_id``).

    Empty when ``is_panelizable`` is True.
    """

    rejection_reason: str = ""
    """Human-readable summary when ``is_panelizable`` is False.

    Copied from the DRL output's ``WallPanelization.rejection_reason``
    and enriched with feasibility-specific context.
    """


# ══════════════════════════════════════════════════════════════════════════════
# 5. Per-Room Feasibility
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class RoomFeasibility:
    """Feasibility assessment for a single room.

    Summarizes whether a pod was placed in this room and, if not, why.
    """

    room_id: int
    """Room identifier matching ``Room.room_id`` in the source graph."""

    room_label: str = ""
    """Semantic label (e.g., 'Bathroom', 'Kitchen')."""

    room_area_sqft: float = 0.0
    """Room area in square feet."""

    is_eligible: bool = True
    """Whether this room is eligible for pod placement.

    False for circulation, exterior, or room types without pod products.
    """

    has_pod: bool = False
    """True if a pod was successfully placed by the DRL Agent."""

    pod_sku: str = ""
    """SKU of the placed pod, or empty if no pod was placed."""

    blocker_ids: list[str] = field(default_factory=list)
    """IDs of blockers preventing pod placement in this room.

    Empty when ``has_pod`` is True or ``is_eligible`` is False.
    """

    rejection_reason: str = ""
    """Human-readable summary when eligible but no pod was placed."""


# ══════════════════════════════════════════════════════════════════════════════
# 6. Per-Floor Scoring
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class FloorScore:
    """Feasibility score for a single floor in a multi-floor project.

    When the input contains only one floor, the FloorScore and the
    whole-project score will be identical.
    """

    floor_id: str
    """Floor identifier (e.g., '1', '2', 'B1' for basement level 1)."""

    floor_label: str = ""
    """Human-readable floor name (e.g., 'Ground Floor', 'Level 2')."""

    wall_coverage_pct: float = 0.0
    """Percentage of total wall length on this floor that is panelized."""

    area_coverage_pct: float = 0.0
    """Percentage of total panelizable wall area on this floor that is
    covered by panels.

    Area = wall_length * wall_height for each panelized wall.
    """

    pod_placement_rate_pct: float = 0.0
    """Percentage of eligible rooms on this floor with a pod placed."""

    blocker_count: int = 0
    """Number of blockers on this floor."""

    total_wall_count: int = 0
    """Total number of wall segments on this floor."""

    panelized_wall_count: int = 0
    """Number of walls successfully panelized on this floor."""

    total_room_count: int = 0
    """Total number of interior rooms on this floor."""

    placed_room_count: int = 0
    """Number of rooms with a pod placed on this floor."""

    feasibility_score: float = 0.0
    """Composite feasibility score for this floor, in [0, 1].

    Weighted combination of wall coverage, pod placement rate, and
    blocker severity.  Higher is better.
    """


# ══════════════════════════════════════════════════════════════════════════════
# 7. Coverage Metrics
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class CoverageMetrics:
    """Aggregate prefab coverage metrics for the entire project.

    Three orthogonal views of coverage: by linear wall length, by wall
    area, and by estimated cost fraction.  Cost fractions are computed
    from KG unit pricing but the actual dollar amounts live in the
    BillOfMaterials contract.
    """

    by_wall_length_pct: float = 0.0
    """Percentage of total wall length covered by panels.

    ``(sum of panelized wall lengths / total wall length) * 100``.
    """

    by_area_pct: float = 0.0
    """Percentage of total panelizable wall area covered by panels.

    Area is computed as wall_length * wall_height for each wall.
    """

    by_cost_pct: float = 0.0
    """Estimated percentage of total wall cost addressable by prefab panels.

    Computed from KG ``Panel.unit_cost_per_foot`` pricing, but reported
    as a ratio — not a dollar amount.  Dollar estimates are the BOM
    Agent's responsibility.
    """

    total_wall_length_inches: float = 0.0
    """Total linear wall length across all floors, in inches."""

    panelized_wall_length_inches: float = 0.0
    """Total wall length covered by panels, in inches."""

    total_wall_area_sqft: float = 0.0
    """Total panelizable wall area, in square feet."""

    panelized_wall_area_sqft: float = 0.0
    """Wall area covered by panels, in square feet."""


# ══════════════════════════════════════════════════════════════════════════════
# 8. Summary Statistics
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class FeasibilitySummary:
    """High-level summary statistics for the feasibility report.

    Designed for dashboard display and quick decision-making.
    """

    total_wall_count: int = 0
    """Total number of wall segments in the project."""

    panelized_wall_count: int = 0
    """Number of walls successfully assigned panels."""

    unpanelized_wall_count: int = 0
    """Number of walls that could not be panelized."""

    total_room_count: int = 0
    """Total number of interior rooms."""

    eligible_room_count: int = 0
    """Number of rooms eligible for pod placement."""

    placed_room_count: int = 0
    """Number of rooms with a pod successfully placed."""

    total_blocker_count: int = 0
    """Total number of identified blockers."""

    hard_blocker_count: int = 0
    """Number of blockers with severity == 1.0 (hard blockers)."""

    soft_blocker_count: int = 0
    """Number of blockers with severity < 1.0 (soft/efficiency blockers)."""

    suggestion_count: int = 0
    """Total number of design modification suggestions."""

    max_coverage_gain_pct: float = 0.0
    """Maximum additional coverage achievable if all suggestions are adopted.

    This is an optimistic upper bound — suggestions may overlap.
    """

    spur_score: float = 0.0
    """SPUR score from the DRL output, carried forward for convenience."""


# ══════════════════════════════════════════════════════════════════════════════
# 9. Top-Level Report
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class FeasibilityReport:
    """Complete output of the Feasibility Agent (Phase 9).

    Wraps all feasibility analysis into a single structure consumed by
    the final pipeline output (Phase 12) and presented to the user.

    This report answers: "How much of this floor plan can Capsule
    Manufacturing prefabricate, what's blocking the rest, and what
    design changes would improve coverage?"

    **Consumers:**
    - Pipeline CLI (Phase 12) — renders the report for the user
    - BOM Agent — reads coverage data to scope the bill of materials
    - BIM Transplant Agent — reads per-wall feasibility to determine
      which walls get 3D panel models vs. conventional framing notes

    **Must NOT contain:** Dollar-value cost estimates. Cost estimation
    is the BOM Agent's exclusive domain (see ``bill_of_materials.py``).

    Reference: AGENTS.md §Feasibility Agent, TASKS.md §Phase 9.
    """

    source: PanelizationResult
    """The PanelizationResult that was analyzed.

    Retained so downstream consumers can access the underlying PanelMap,
    PlacementMap, and ClassifiedWallGraph without a separate reference.
    """

    # ── Coverage metrics ─────────────────────────────────────────────────

    coverage: CoverageMetrics
    """Aggregate prefab coverage metrics (by length, area, cost ratio)."""

    # ── Per-element breakdowns ───────────────────────────────────────────

    wall_feasibility: list[WallFeasibility]
    """Per-wall feasibility assessment, one per wall segment.

    Ordered by ``edge_id`` to match the source graph's wall_segments.
    Length matches ``source.source_graph.graph.wall_segments``.
    """

    room_feasibility: list[RoomFeasibility]
    """Per-room feasibility assessment, one per interior room.

    Ordered by ``room_id`` to match the source graph's rooms.
    """

    # ── Blockers and suggestions ─────────────────────────────────────────

    blockers: list[Blocker]
    """All identified blockers, ordered by severity (highest first)."""

    suggestions: list[DesignSuggestion]
    """Design modification suggestions, ordered by estimated coverage
    gain (highest first).

    Each suggestion references one or more blockers it would resolve.
    """

    # ── Scoring ──────────────────────────────────────────────────────────

    floor_scores: list[FloorScore]
    """Per-floor feasibility scores.

    For single-floor projects, this list has exactly one entry.
    For multi-floor projects, one entry per floor.
    """

    project_score: float = 0.0
    """Whole-project feasibility score, in [0, 1].

    Weighted average of per-floor scores (weighted by wall count).
    This is the single headline number for the feasibility assessment.
    Higher is better.
    """

    # ── Summary ──────────────────────────────────────────────────────────

    summary: FeasibilitySummary = field(default_factory=FeasibilitySummary)
    """High-level summary statistics for quick reference."""

    # ── Metadata ─────────────────────────────────────────────────────────

    metadata: dict[str, object] = field(default_factory=dict)
    """Additional metadata (processing time, agent version, etc.)."""
