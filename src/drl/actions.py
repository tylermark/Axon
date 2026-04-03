"""Action space definitions for panelization and product placement.

DRL-002: Panelization actions — for each wall, select a panel SKU from
KG-filtered candidates.  The action is a discrete index into the current
candidate list (populated by querying the KG for panels matching the wall's
classification, fire rating, and fabrication constraints).

DRL-003: Placement actions — for each room, select a pod SKU from
KG-filtered candidates plus an orientation (normal or rotated 90 deg).

Both action types use ``gymnasium.spaces.Discrete`` — the agent picks an
integer index into the candidate list presented in the observation.
A special ``SKIP`` action (index 0) allows the agent to leave a wall
unpanelized or a room without a pod (valid when no candidate fits or
the wall/room is unsuitable for prefab).

Action encoding:
    Panelization: action ∈ {0, 1, ..., MAX_CANDIDATES}
        0 = SKIP (no panel for this wall)
        1..MAX_CANDIDATES = index into panel_candidates (1-indexed)
    Placement: action ∈ {0, 1, ..., 2 * MAX_CANDIDATES}
        0 = SKIP (no pod for this room)
        odd  indices (1, 3, 5, ...) = pod candidate at index (a-1)//2, normal orientation
        even indices (2, 4, 6, ...) = pod candidate at index (a-2)//2, rotated 90 deg
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from docs.interfaces.classified_wall_graph import WallClassification
from docs.interfaces.graph_to_serializer import Opening, WallSegment
from src.drl.state import (
    MAX_CANDIDATES,
    fire_rating_to_hours,
    wall_type_to_panel_type,
)
from src.knowledge_graph.query import (
    PanelRecommendation,
    get_panels_for_wall_segment,
    get_valid_pods,
    validate_panel_fabrication,
)
from src.knowledge_graph.schema import Panel, Pod

if TYPE_CHECKING:
    from src.knowledge_graph.loader import KnowledgeGraphStore

# ── Action space sizes ──────────────────────────────────────────────────────

PANELIZATION_ACTION_SIZE: int = MAX_CANDIDATES + 1
"""Discrete action space size for panelization: SKIP + MAX_CANDIDATES panels."""

PLACEMENT_ACTION_SIZE: int = 2 * MAX_CANDIDATES + 1
"""Discrete action space size for placement: SKIP + MAX_CANDIDATES pods x 2 orientations."""


# ── Decoded action dataclasses ──────────────────────────────────────────────


@dataclass
class PanelAction:
    """Decoded panelization action for a single wall segment.

    Attributes:
        wall_edge_id: The edge_id of the wall being panelized.
        skip: If True, no panel is assigned to this wall.
        panel: The selected Panel from the KG catalog.
        recommendation: The full PanelRecommendation with cut lengths,
            quantity, waste, and splice info.
        panel_assignments: List of ``(sku, cut_length)`` tuples for
            the wall — derived from the recommendation.
    """

    wall_edge_id: int
    skip: bool = False
    panel: Panel | None = None
    recommendation: PanelRecommendation | None = None
    panel_assignments: list[tuple[str, float]] = field(default_factory=list)


@dataclass
class PlacementAction:
    """Decoded placement action for a single room.

    Attributes:
        room_id: The room_id being assigned.
        skip: If True, no pod is placed in this room.
        pod: The selected Pod from the KG catalog.
        rotated: If True, the pod is placed rotated 90 degrees.
        position_x: X coordinate of pod center in inches (computed
            as room centroid — pod is centered by default).
        position_y: Y coordinate of pod center in inches.
    """

    room_id: int
    skip: bool = False
    pod: Pod | None = None
    rotated: bool = False
    position_x: float = 0.0
    position_y: float = 0.0


# ── Candidate retrieval ─────────────────────────────────────────────────────


def get_panel_candidates(
    store: KnowledgeGraphStore,
    wall: WallSegment,
    classification: WallClassification,
    openings_on_wall: list[Opening],
    scale: float,
    effective_length_inches: float | None = None,
) -> list[PanelRecommendation]:
    """Query the KG for valid panel recommendations for a wall segment.

    Converts wall geometry to inches, maps wall classification to KG
    PanelType, and calls the KG query API.  Openings reduce the
    panelizable length — panels cannot span openings.

    The returned list is capped at ``MAX_CANDIDATES`` entries, sorted
    by the KG's recommendation score (descending).

    Args:
        store: Knowledge Graph store.
        wall: Wall segment geometry.
        classification: Classification for this wall.
        openings_on_wall: Openings detected on this wall.
        scale: Scale factor from PDF user units to millimeters.
        effective_length_inches: Pre-computed effective wall length in
            inches (after openings, corner deductions, etc.).  When
            provided, this overrides the internally computed length so
            that panel recommendations match the actual panelizable
            length the reward function evaluates against.

    Returns:
        Up to MAX_CANDIDATES PanelRecommendation objects.
    """
    if effective_length_inches is not None:
        panelizable_length = effective_length_inches
    else:
        to_inches = scale / 25.4 if scale != 1.0 else 1.0 / 72.0
        wall_length_inches = wall.length * to_inches

        # Subtract opening widths — panels must avoid openings
        opening_width_total = sum(o.width * to_inches for o in openings_on_wall)
        panelizable_length = max(wall_length_inches - opening_width_total, 0.0)

    if panelizable_length <= 0.0:
        return []

    panel_type = wall_type_to_panel_type(
        classification.wall_type,
        classification.fire_rating,
    )
    fire_hours = fire_rating_to_hours(classification.fire_rating)

    recommendations = get_panels_for_wall_segment(
        store,
        wall_length_inches=panelizable_length,
        wall_type=panel_type,
        fire_rating_hours=fire_hours,
    )

    # Filter out recommendations that fail fabrication validation
    valid_recs: list[PanelRecommendation] = []
    for rec in recommendations:
        validation = validate_panel_fabrication(
            store,
            panel_sku=rec.panel.sku,
            required_length_inches=max(rec.cut_lengths_inches) if rec.cut_lengths_inches else 0.0,
            required_quantity=rec.quantity,
        )
        if validation.is_valid:
            valid_recs.append(rec)
        if len(valid_recs) >= MAX_CANDIDATES:
            break

    return valid_recs[:MAX_CANDIDATES]


def get_pod_candidates(
    store: KnowledgeGraphStore,
    room_width_inches: float,
    room_depth_inches: float,
    room_label: str,
) -> list[Pod]:
    """Query the KG for valid pods that fit in a room.

    Maps room label to the pod_type filter.  If no label is set,
    queries without a function filter (returns all fitting pods).

    Args:
        store: Knowledge Graph store.
        room_width_inches: Room bounding box width in inches.
        room_depth_inches: Room bounding box depth in inches.
        room_label: Semantic room label (e.g., 'Bathroom').

    Returns:
        Up to MAX_CANDIDATES Pod objects.
    """
    # Normalize label to lowercase; empty string means no filter
    room_function = room_label.lower().strip() if room_label else None

    pods = get_valid_pods(
        store,
        room_width_inches=room_width_inches,
        room_depth_inches=room_depth_inches,
        room_function=room_function,
    )

    # If function-filtered query returns nothing, try without function filter
    if not pods and room_function is not None:
        pods = get_valid_pods(
            store,
            room_width_inches=room_width_inches,
            room_depth_inches=room_depth_inches,
        )

    return pods[:MAX_CANDIDATES]


# ── Action masking ──────────────────────────────────────────────────────────


def compute_panel_action_mask(
    num_candidates: int,
) -> np.ndarray:
    """Compute a binary mask over the panelization action space.

    SKIP (action 0) is always valid.
    Actions 1..num_candidates are valid (mapped to candidate indices 0..n-1).
    Actions beyond num_candidates are masked out.

    Args:
        num_candidates: Number of valid panel candidates for the current wall.

    Returns:
        Binary mask of shape ``(PANELIZATION_ACTION_SIZE,)``, dtype float32.
    """
    mask = np.zeros(PANELIZATION_ACTION_SIZE, dtype=np.float32)
    mask[0] = 1.0  # SKIP is always valid
    for i in range(min(num_candidates, MAX_CANDIDATES)):
        mask[i + 1] = 1.0
    return mask


def compute_placement_action_mask(
    num_candidates: int,
    room_width_inches: float,
    room_depth_inches: float,
    pods: list[Pod],
) -> np.ndarray:
    """Compute a binary mask over the placement action space.

    SKIP (action 0) is always valid.
    For each pod candidate, normal orientation is valid if the pod fits
    in ``(width, depth)``. Rotated orientation is valid if the pod fits
    in ``(depth, width)``.

    Args:
        num_candidates: Number of valid pod candidates.
        room_width_inches: Room width in inches.
        room_depth_inches: Room depth in inches.
        pods: Pod candidates from the KG.

    Returns:
        Binary mask of shape ``(PLACEMENT_ACTION_SIZE,)``, dtype float32.
    """
    mask = np.zeros(PLACEMENT_ACTION_SIZE, dtype=np.float32)
    mask[0] = 1.0  # SKIP is always valid

    for i in range(min(num_candidates, MAX_CANDIDATES)):
        pod = pods[i]
        # Normal orientation: pod.width along room width, pod.depth along room depth
        if (
            room_width_inches >= pod.min_room_width_inches
            and room_depth_inches >= pod.min_room_depth_inches
        ):
            mask[2 * i + 1] = 1.0  # normal

        # Rotated 90 degrees: swap pod dimensions against room
        if (
            room_width_inches >= pod.min_room_depth_inches
            and room_depth_inches >= pod.min_room_width_inches
        ):
            mask[2 * i + 2] = 1.0  # rotated

    return mask


# ── Action decoding ─────────────────────────────────────────────────────────


def decode_panel_action(
    action: int,
    wall: WallSegment,
    candidates: list[PanelRecommendation],
) -> PanelAction:
    """Decode a discrete panelization action into a PanelAction.

    Args:
        action: Integer action from the policy, in [0, PANELIZATION_ACTION_SIZE).
        wall: The wall segment being panelized.
        candidates: Panel recommendations from ``get_panel_candidates``.

    Returns:
        Decoded PanelAction with panel assignment details.

    Raises:
        ValueError: If action index is out of range or points to a
            non-existent candidate.
    """
    if action < 0 or action >= PANELIZATION_ACTION_SIZE:
        raise ValueError(
            f"Panel action {action} out of range [0, {PANELIZATION_ACTION_SIZE})"
        )

    if action == 0:
        return PanelAction(wall_edge_id=wall.edge_id, skip=True)

    candidate_idx = action - 1
    if candidate_idx >= len(candidates):
        raise ValueError(
            f"Panel action {action} refers to candidate index {candidate_idx} "
            f"but only {len(candidates)} candidates available"
        )

    rec = candidates[candidate_idx]
    assignments = [(rec.panel.sku, cl) for cl in rec.cut_lengths_inches]

    return PanelAction(
        wall_edge_id=wall.edge_id,
        skip=False,
        panel=rec.panel,
        recommendation=rec,
        panel_assignments=assignments,
    )


def decode_placement_action(
    action: int,
    room_id: int,
    candidates: list[Pod],
    room_centroid: tuple[float, float],
) -> PlacementAction:
    """Decode a discrete placement action into a PlacementAction.

    Pod is centered at the room centroid. Orientation is encoded in the
    action index parity: odd = normal, even = rotated 90 degrees.

    Args:
        action: Integer action from the policy, in [0, PLACEMENT_ACTION_SIZE).
        room_id: ID of the room being assigned.
        candidates: Pod candidates from ``get_pod_candidates``.
        room_centroid: ``(x, y)`` centroid of the room in inches.

    Returns:
        Decoded PlacementAction with pod and orientation.

    Raises:
        ValueError: If action index is out of range or points to a
            non-existent candidate.
    """
    if action < 0 or action >= PLACEMENT_ACTION_SIZE:
        raise ValueError(
            f"Placement action {action} out of range [0, {PLACEMENT_ACTION_SIZE})"
        )

    if action == 0:
        return PlacementAction(room_id=room_id, skip=True)

    # Odd actions = normal orientation, even = rotated
    if action % 2 == 1:
        candidate_idx = (action - 1) // 2
        rotated = False
    else:
        candidate_idx = (action - 2) // 2
        rotated = True

    if candidate_idx >= len(candidates):
        raise ValueError(
            f"Placement action {action} refers to candidate index {candidate_idx} "
            f"but only {len(candidates)} candidates available"
        )

    pod = candidates[candidate_idx]

    return PlacementAction(
        room_id=room_id,
        skip=False,
        pod=pod,
        rotated=rotated,
        position_x=room_centroid[0],
        position_y=room_centroid[1],
    )
