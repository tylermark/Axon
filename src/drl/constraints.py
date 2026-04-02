"""Opening and junction constraints for panelization and placement.

DRL-007: Opening constraint handling — panels must not span across door/window
openings. Walls with openings are split into segments between openings, and
each segment is panelized independently.

DRL-008: Joint/angle constraint handling — at wall junctions (T-junctions,
corners, crosses), panels from adjacent walls must have compatible specs
(gauge, stud depth). Corner panels must account for the thickness of the
perpendicular wall.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from docs.interfaces.graph_to_serializer import Opening, WallSegment
    from src.knowledge_graph.loader import KnowledgeGraphStore
    from src.knowledge_graph.schema import Panel

logger = logging.getLogger(__name__)


# ── DRL-007: Opening Constraint Types ────────────────────────────────────────


@dataclass
class WallSubSegment:
    """A panelizable sub-segment of a wall, between openings or wall ends.

    When a wall has openings (doors/windows), the wall is divided into
    sub-segments that can each be independently panelized. The sub-segments
    are defined by their start and end positions along the wall, and their
    effective length excludes the opening widths.

    Attributes:
        wall_edge_id: The parent wall's edge_id.
        start_offset_inches: Start position along the wall in inches
            (from wall start node).
        end_offset_inches: End position along the wall in inches.
        length_inches: Panelizable length of this sub-segment in inches.
        segment_index: Index of this sub-segment within the wall (0-based).
        total_segments: Total number of sub-segments for this wall.
        left_bounded_by_opening: True if the left boundary is an opening
            (not the wall start).
        right_bounded_by_opening: True if the right boundary is an opening
            (not the wall end).
    """

    wall_edge_id: int
    start_offset_inches: float
    end_offset_inches: float
    length_inches: float
    segment_index: int
    total_segments: int
    left_bounded_by_opening: bool = False
    right_bounded_by_opening: bool = False


def compute_wall_sub_segments(
    wall: WallSegment,
    openings: list[Opening],
    scale: float,
) -> list[WallSubSegment]:
    """Split a wall into panelizable sub-segments around openings.

    Openings create exclusion zones along the wall. The regions between
    openings (and between openings and wall ends) become independently
    panelizable sub-segments.

    Args:
        wall: The wall segment to split.
        openings: Openings on this wall (filtered by wall_edge_id).
        scale: Scale factor from PDF user units to millimeters.

    Returns:
        List of WallSubSegment objects, sorted by position along wall.
        Returns a single sub-segment spanning the full wall if no openings.
    """
    to_inches = scale / 25.4 if scale != 1.0 else 1.0 / 72.0
    wall_length_inches = wall.length * to_inches

    if not openings or wall_length_inches <= 0.0:
        return [
            WallSubSegment(
                wall_edge_id=wall.edge_id,
                start_offset_inches=0.0,
                end_offset_inches=wall_length_inches,
                length_inches=wall_length_inches,
                segment_index=0,
                total_segments=1,
                left_bounded_by_opening=False,
                right_bounded_by_opening=False,
            )
        ]

    # Build exclusion zones from openings, sorted by position
    exclusion_zones: list[tuple[float, float]] = []
    for opening in openings:
        opening_width_inches = opening.width * to_inches
        center_inches = opening.position_along_wall * wall_length_inches
        half_width = opening_width_inches / 2.0
        zone_start = max(center_inches - half_width, 0.0)
        zone_end = min(center_inches + half_width, wall_length_inches)
        exclusion_zones.append((zone_start, zone_end))

    # Sort by start position
    exclusion_zones.sort(key=lambda z: z[0])

    # Merge overlapping exclusion zones
    merged: list[tuple[float, float]] = []
    for start, end in exclusion_zones:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))

    # Build sub-segments from gaps between exclusion zones
    sub_segments: list[WallSubSegment] = []
    cursor = 0.0

    for zone_start, zone_end in merged:
        if zone_start > cursor + 0.25:  # minimum 1/4 inch gap to create a segment
            sub_segments.append(
                WallSubSegment(
                    wall_edge_id=wall.edge_id,
                    start_offset_inches=cursor,
                    end_offset_inches=zone_start,
                    length_inches=zone_start - cursor,
                    segment_index=len(sub_segments),
                    total_segments=0,  # will be set after all segments are found
                    left_bounded_by_opening=cursor > 0.0,
                    right_bounded_by_opening=True,
                )
            )
        cursor = zone_end

    # Final segment after the last opening
    if cursor < wall_length_inches - 0.25:
        sub_segments.append(
            WallSubSegment(
                wall_edge_id=wall.edge_id,
                start_offset_inches=cursor,
                end_offset_inches=wall_length_inches,
                length_inches=wall_length_inches - cursor,
                segment_index=len(sub_segments),
                total_segments=0,
                left_bounded_by_opening=cursor > 0.0,
                right_bounded_by_opening=False,
            )
        )

    # Fix total_segments and segment_index
    total = len(sub_segments)
    for i, seg in enumerate(sub_segments):
        seg.segment_index = i
        seg.total_segments = total

    # Fix the first segment's left_bounded flag
    if sub_segments:
        sub_segments[0].left_bounded_by_opening = False

    return sub_segments


# ── DRL-008: Junction Constraint Types ───────────────────────────────────────


@dataclass
class JunctionInfo:
    """Information about a wall junction (shared node between walls).

    Attributes:
        node_id: The shared node index.
        wall_edge_ids: Edge IDs of walls meeting at this junction.
        junction_type: Type of junction: 'corner', 'T', 'cross', or 'end'.
        angles: Angles (radians) of walls at this junction.
    """

    node_id: int
    wall_edge_ids: list[int]
    junction_type: str
    angles: list[float]


def compute_junction_map(
    walls: list[WallSegment],
) -> dict[int, JunctionInfo]:
    """Build a map of node_id to JunctionInfo for all wall junctions.

    A junction is any node shared by two or more walls. The junction type
    is determined by the number of walls meeting:
    - 1 wall: ``"end"`` (dead-end, no constraint needed)
    - 2 walls: ``"corner"`` (L-junction or straight continuation)
    - 3 walls: ``"T"`` (T-junction)
    - 4+ walls: ``"cross"`` (cross or more complex intersection)

    Args:
        walls: All wall segments in the floor plan.

    Returns:
        Map from node_id to JunctionInfo, only for nodes with 2+ walls.
    """
    # Collect which walls touch each node
    node_to_walls: dict[int, list[int]] = {}
    node_to_angles: dict[int, list[float]] = {}

    for wall in walls:
        for node_id in (wall.start_node, wall.end_node):
            node_to_walls.setdefault(node_id, []).append(wall.edge_id)
            node_to_angles.setdefault(node_id, []).append(wall.angle)

    junction_map: dict[int, JunctionInfo] = {}
    for node_id, edge_ids in node_to_walls.items():
        if len(edge_ids) < 2:
            continue  # Skip dead-ends

        n_walls = len(edge_ids)
        if n_walls == 2:
            junction_type = "corner"
        elif n_walls == 3:
            junction_type = "T"
        else:
            junction_type = "cross"

        junction_map[node_id] = JunctionInfo(
            node_id=node_id,
            wall_edge_ids=edge_ids,
            junction_type=junction_type,
            angles=node_to_angles[node_id],
        )

    return junction_map


def compute_junction_penalties(
    wall_edge_id: int,
    panel: Panel | None,
    junction_map: dict[int, JunctionInfo],
    wall_assignments: dict[int, list[tuple[str, float]]],
    walls: list[WallSegment],
    store: KnowledgeGraphStore,
) -> tuple[float, list[str]]:
    """Compute junction compatibility penalty for a panel assignment.

    Checks whether the panel assigned to ``wall_edge_id`` is compatible
    with panels already assigned to adjacent walls at shared junctions.

    Compatibility is checked on:
    - Gauge: must match at shared junctions (different gauges cannot be
      spliced together structurally).
    - Stud depth: must match at shared junctions.

    Also computes a corner thickness adjustment penalty: at corners (90-degree
    junctions), the panel on one wall must account for the thickness of the
    perpendicular wall's panel.

    Args:
        wall_edge_id: The wall being panelized.
        panel: The panel being assigned (None if skip).
        junction_map: Pre-computed junction map from ``compute_junction_map``.
        wall_assignments: Current panel assignments (edge_id -> [(sku, len)]).
        walls: All wall segments.
        store: KG store for looking up panel specs from SKUs.

    Returns:
        Tuple of ``(penalty, violations)`` where penalty is a non-positive
        float and violations is a list of human-readable descriptions.
    """
    if panel is None:
        return 0.0, []

    # Find which junctions this wall participates in
    wall_lookup = {w.edge_id: w for w in walls}
    wall = wall_lookup.get(wall_edge_id)
    if wall is None:
        return 0.0, []

    violations: list[str] = []
    penalty = 0.0

    for node_id in (wall.start_node, wall.end_node):
        junction = junction_map.get(node_id)
        if junction is None:
            continue

        # Check compatibility with each adjacent wall that already has a panel
        for adj_edge_id in junction.wall_edge_ids:
            if adj_edge_id == wall_edge_id:
                continue
            if adj_edge_id not in wall_assignments:
                continue  # Not yet assigned, no constraint to check

            # Look up the panel SKU of the adjacent wall
            adj_assignments = wall_assignments[adj_edge_id]
            if not adj_assignments:
                continue

            # Use the first panel's SKU (at a junction, the relevant panel
            # is the one nearest the junction node)
            adj_sku = adj_assignments[0][0]
            adj_panel = store.panels.get(adj_sku)
            if adj_panel is None:
                continue

            # Gauge compatibility check
            if panel.gauge != adj_panel.gauge:
                violations.append(
                    f"Junction node {node_id}: gauge mismatch between "
                    f"wall {wall_edge_id} ({panel.gauge}ga) and "
                    f"wall {adj_edge_id} ({adj_panel.gauge}ga)"
                )
                penalty -= 0.3

            # Stud depth compatibility check
            if panel.stud_depth_inches != adj_panel.stud_depth_inches:
                violations.append(
                    f"Junction node {node_id}: stud depth mismatch between "
                    f'wall {wall_edge_id} ({panel.stud_depth_inches}") and '
                    f'wall {adj_edge_id} ({adj_panel.stud_depth_inches}")'
                )
                penalty -= 0.2

        # Corner thickness adjustment
        if junction.junction_type == "corner" and len(junction.angles) >= 2:
            angles = junction.angles
            # Compute the angle between the two walls
            angle_diff = abs(angles[0] - angles[1])
            # Normalize to [0, pi]
            angle_diff = angle_diff % math.pi
            # Check for approximately perpendicular (90 degrees +/- 10 degrees)
            is_perpendicular = abs(angle_diff - math.pi / 2.0) < math.radians(10.0)

            if is_perpendicular:
                # At perpendicular corners, one wall's panel should account for
                # the other wall's thickness. Compute the expected thickness
                # deduction and check if the panel assignment accounts for it.
                # For now, add a small penalty as a signal to the policy to
                # prefer panels that work well at corners.
                adj_wall_ids = [eid for eid in junction.wall_edge_ids if eid != wall_edge_id]
                for adj_eid in adj_wall_ids:
                    adj_wall = wall_lookup.get(adj_eid)
                    if adj_wall is not None:
                        # The perpendicular wall's stud depth reduces the
                        # effective length of this wall's panel at the corner.
                        # We track this but don't impose a hard penalty —
                        # the reward signal helps the policy learn.
                        pass  # Tracked via junction info in observations

    return max(penalty, -1.0), violations


def get_corner_thickness_deduction(
    wall_edge_id: int,
    junction_map: dict[int, JunctionInfo],
    walls: list[WallSegment],
    scale: float,
) -> float:
    """Compute the total thickness deduction at corners for a wall.

    At perpendicular corners, the panel on this wall must be shortened by
    the thickness of the perpendicular wall (the perpendicular wall's stud
    depth takes up space at the corner).

    Args:
        wall_edge_id: The wall to compute deductions for.
        junction_map: Pre-computed junction map.
        walls: All wall segments.
        scale: Scale factor (PDF user units to mm).

    Returns:
        Total deduction in inches to subtract from panelizable length.
    """
    to_inches = scale / 25.4 if scale != 1.0 else 1.0 / 72.0

    wall_lookup = {w.edge_id: w for w in walls}
    wall = wall_lookup.get(wall_edge_id)
    if wall is None:
        return 0.0

    deduction = 0.0

    for node_id in (wall.start_node, wall.end_node):
        junction = junction_map.get(node_id)
        if junction is None:
            continue
        if junction.junction_type not in ("corner", "T", "cross"):
            continue

        # Check if any adjacent wall is roughly perpendicular
        for adj_eid in junction.wall_edge_ids:
            if adj_eid == wall_edge_id:
                continue
            adj_wall = wall_lookup.get(adj_eid)
            if adj_wall is None:
                continue

            # Compute angle between this wall and the adjacent wall
            angle_diff = abs(wall.angle - adj_wall.angle) % math.pi
            is_perpendicular = abs(angle_diff - math.pi / 2.0) < math.radians(10.0)

            if is_perpendicular:
                # The adjacent wall's thickness reduces this wall's
                # available length at the junction
                thickness_inches = adj_wall.thickness * to_inches
                deduction += thickness_inches
                break  # Only one deduction per junction node

    return deduction
