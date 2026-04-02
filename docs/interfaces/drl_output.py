"""Interface contract: DRL Agent → Feasibility / BOM / BIM Transplant Agents.

Defines the PanelizationResult dataclass — the output of the DRL Agent
(Phase 8) and input to the Feasibility Agent (Phase 9), BOM Agent
(Phase 10), and BIM Transplant Agent (Phase 11).

The DRL agent receives a ClassifiedWallGraph and Knowledge Graph query
results, then produces two maps:

    PanelMap — per-wall panel assignments (SKU, cut lengths, splices)
    PlacementMap — per-room pod/product placements (SKU, position, orientation)

These are wrapped in a PanelizationResult that carries summary statistics
(SPUR score, waste %, coverage %) consumed by downstream agents.

Reference: AGENTS.md §DRL Agent, TASKS.md §Phase 8.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from docs.interfaces.classified_wall_graph import ClassifiedWallGraph


# ══════════════════════════════════════════════════════════════════════════════
# 1. Panel Assignment — per-wall
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class PanelAssignment:
    """A single panel placed along a wall segment.

    Represents one physical CFS panel cut to length and positioned at a
    specific offset along the parent wall.  Multiple PanelAssignments may
    cover a single wall when splicing is required.
    """

    panel_sku: str
    """SKU of the panel product from the Knowledge Graph catalog."""

    cut_length_inches: float
    """Actual cut length of this panel piece, in inches.

    Must fall within the panel's [min_length, max_length] range as
    defined in the KG schema (``Panel.min_length_inches`` /
    ``Panel.max_length_inches``).
    """

    position_along_wall: float
    """Start offset of this panel from the wall's start node, in inches.

    The panel occupies the interval
    [position_along_wall, position_along_wall + cut_length_inches]
    along the wall's linear axis.
    """

    panel_index: int
    """Zero-based index of this panel within the wall's assignment list.

    Used to identify splice joints: a splice exists between
    panel_index i and panel_index i+1.
    """


@dataclass
class WallPanelization:
    """Complete panel assignment for a single wall segment.

    Maps one wall (identified by ``edge_id``) to its ordered list of
    panel assignments, splice metadata, and per-wall waste.
    """

    edge_id: int
    """Edge index matching ``WallSegment.edge_id`` in the source
    ClassifiedWallGraph / FinalizedGraph."""

    wall_length_inches: float
    """Total wall length in inches (converted from PDF user units via
    the graph's ``scale_factor``)."""

    panels: list[PanelAssignment]
    """Ordered list of panel assignments along this wall.

    Panels are ordered from start node to end node.  Adjacent panels
    share a splice joint when ``requires_splice`` is True.
    """

    requires_splice: bool
    """True if this wall needs more than one panel (splice joints)."""

    splice_connection_skus: list[str] = field(default_factory=list)
    """SKUs of splice connection hardware at each joint.

    Length is ``len(panels) - 1`` when splicing is required, empty
    otherwise.  Each entry is the Connection SKU used at the joint
    between consecutive panels.
    """

    total_material_inches: float = 0.0
    """Sum of all panel cut lengths for this wall, in inches."""

    waste_inches: float = 0.0
    """Material waste for this wall: ``total_material_inches - wall_length_inches``.

    Always >= 0.  Waste arises from minimum-length constraints and
    rounding to standard cut increments.
    """

    waste_percentage: float = 0.0
    """Waste as a percentage of total material:
    ``(waste_inches / total_material_inches) * 100`` if material > 0,
    else 0.
    """

    is_panelizable: bool = True
    """False if the DRL agent could not find any valid panel assignment
    for this wall (e.g., wall is too short, no compatible panel type
    in the KG, or structural constraints cannot be satisfied)."""

    rejection_reason: str = ""
    """Human-readable explanation when ``is_panelizable`` is False."""


@dataclass
class PanelMap:
    """Panel assignments for all walls in the floor plan.

    One ``WallPanelization`` per wall segment in the source
    ClassifiedWallGraph.  Walls that cannot be panelized have
    ``is_panelizable=False`` with a ``rejection_reason``.
    """

    walls: list[WallPanelization]
    """Per-wall panelization results, ordered by ``edge_id``.

    Length matches the number of wall segments in the source graph.
    """

    panelized_wall_count: int = 0
    """Number of walls successfully assigned panels."""

    total_wall_count: int = 0
    """Total number of walls in the source graph."""

    unique_panel_skus: list[str] = field(default_factory=list)
    """Deduplicated list of all panel SKUs used across all walls.

    Useful for BOM aggregation and procurement planning.
    """

    unique_splice_skus: list[str] = field(default_factory=list)
    """Deduplicated list of all splice connection SKUs used."""


# ══════════════════════════════════════════════════════════════════════════════
# 2. Product Placement — per-room
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class ProductPlacement:
    """Placement of a prefabricated pod/product within a room.

    The DRL placer positions one pod per room (where applicable),
    choosing the SKU, position, and orientation that maximizes the
    SPUR reward while respecting KG clearance constraints.
    """

    pod_sku: str
    """SKU of the pod product from the Knowledge Graph catalog."""

    position: np.ndarray
    """Placement center position (x, y) in PDF user units, shape (2,) float64.

    Coordinates are in the same coordinate system as the source
    FinalizedGraph.nodes.
    """

    orientation_deg: float
    """Rotation of the pod about its center, in degrees [0, 360).

    0 = aligned with positive x-axis (no rotation).
    90 = rotated 90 degrees counter-clockwise.
    Only 0 and 90 are valid for axis-aligned rooms; arbitrary angles
    may be used for non-orthogonal geometries.
    """

    clearance_met: bool
    """True if all clearance requirements are satisfied.

    Clearance is defined by ``Pod.clearance_inches`` in the KG schema
    and must be maintained on all sides between the pod boundary and
    the room's bounding walls.
    """

    clearance_margins: dict[str, float] = field(default_factory=dict)
    """Actual clearance margins in inches on each side.

    Keys: 'north', 'south', 'east', 'west' (relative to the pod's
    local coordinate frame after rotation).  Values are the gap
    between the pod boundary and the nearest wall on that side.
    """

    confidence: float = 1.0
    """Placement confidence from the DRL policy, in [0, 1].

    Higher values indicate the policy is more certain this placement
    maximizes the reward.  Low confidence may trigger human review.
    """


@dataclass
class RoomPlacement:
    """Product placement result for a single room.

    Maps one room (identified by ``room_id``) to its pod placement,
    or records that the room is not eligible for pod placement.
    """

    room_id: int
    """Room identifier matching ``Room.room_id`` in the source
    FinalizedGraph."""

    room_label: str = ""
    """Semantic label of the room (e.g., 'Bathroom', 'Kitchen').

    Copied from ``Room.label`` in the source graph for convenience.
    """

    room_area_sqft: float = 0.0
    """Room area in square feet (converted from PDF user units)."""

    placement: ProductPlacement | None = None
    """Pod placement for this room, or None if no pod is placed.

    None when ``is_eligible`` is False or when no compatible pod
    exists in the KG for this room's function/dimensions.
    """

    is_eligible: bool = True
    """Whether this room is eligible for pod placement.

    False for rooms that are too small, exterior boundaries,
    circulation spaces, or room types without matching pod products.
    """

    rejection_reason: str = ""
    """Human-readable explanation when ``is_eligible`` is False or
    ``placement`` is None despite eligibility."""


@dataclass
class PlacementMap:
    """Pod/product placements for all rooms in the floor plan.

    One ``RoomPlacement`` per room in the source FinalizedGraph.
    Rooms without compatible pods have ``placement=None``.
    """

    rooms: list[RoomPlacement]
    """Per-room placement results, ordered by ``room_id``.

    Length matches the number of rooms in the source graph
    (excluding the exterior boundary room).
    """

    placed_room_count: int = 0
    """Number of rooms with a pod successfully placed."""

    eligible_room_count: int = 0
    """Number of rooms eligible for pod placement."""

    total_room_count: int = 0
    """Total number of interior rooms in the source graph."""

    unique_pod_skus: list[str] = field(default_factory=list)
    """Deduplicated list of all pod SKUs placed across all rooms."""


# ══════════════════════════════════════════════════════════════════════════════
# 3. Combined Result
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class PanelizationResult:
    """Complete output of the DRL Agent (Phase 8).

    Wraps the PanelMap and PlacementMap together with summary
    statistics consumed by downstream agents:

    - **Feasibility Agent** reads ``spur_score``, ``waste_percentage``,
      ``coverage_percentage``, and per-wall/per-room rejection reasons.
    - **BOM Agent** reads ``panel_map`` and ``placement_map`` to
      aggregate quantities and costs.
    - **BIM Transplant Agent** reads panel SKUs and pod placements to
      look up BIM families and assemble the 3D model.

    Reference: AGENTS.md §DRL Agent, TASKS.md §Phase 8.
    """

    source_graph: ClassifiedWallGraph
    """The ClassifiedWallGraph that was panelized.

    Retained for downstream agents that need access to wall geometry,
    classifications, and room boundaries alongside the DRL output.
    """

    panel_map: PanelMap
    """Panel assignments for all wall segments."""

    placement_map: PlacementMap
    """Pod/product placements for all rooms."""

    # ── Summary statistics ────────────────────────────────────────────────

    spur_score: float = 0.0
    """Standardized Prefab Utilization Ratio (SPUR), in [0, 1].

    Composite metric combining panel coverage, waste efficiency, and
    pod placement rate.  Defined as:

        SPUR = w1 * coverage_pct + w2 * (1 - waste_pct) + w3 * pod_placement_rate

    where w1 + w2 + w3 = 1 (default weights: 0.5, 0.3, 0.2).
    Higher is better.  This is the primary reward signal for the DRL
    agent during training.
    """

    coverage_percentage: float = 0.0
    """Percentage of total wall length successfully panelized.

    ``(sum of panelized wall lengths / total wall length) * 100``.
    100% means every wall has a valid panel assignment.
    """

    waste_percentage: float = 0.0
    """Aggregate material waste as a percentage of total material used.

    ``(total_waste_inches / total_material_inches) * 100``.
    Lower is better.
    """

    pod_placement_rate: float = 0.0
    """Percentage of eligible rooms with a pod successfully placed.

    ``(placed_room_count / eligible_room_count) * 100`` if
    eligible > 0, else 0.
    """

    total_panel_count: int = 0
    """Total number of individual panel pieces across all walls."""

    total_splice_count: int = 0
    """Total number of splice joints across all walls."""

    total_material_cost: float = 0.0
    """Estimated total material cost in USD (panels + splices + pods).

    Computed from KG unit costs.  Does not include labor, overhead,
    or shipping.
    """

    # ── DRL policy metadata ───────────────────────────────────────────────

    policy_version: str = ""
    """Version identifier of the DRL policy checkpoint used."""

    episode_reward: float = 0.0
    """Total episode reward from the DRL rollout that produced this result.

    Useful for comparing policy performance across floor plans.
    """

    inference_steps: int = 0
    """Number of DRL environment steps taken during inference."""

    metadata: dict[str, object] = field(default_factory=dict)
    """Additional metadata (processing time, GPU used, etc.)."""
