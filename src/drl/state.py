"""State representation for the panelization/placement DRL environment.

Encodes the ClassifiedWallGraph, current panel assignments, and pod placements
into fixed-size observation tensors suitable for neural network policies.

The observation is a ``gymnasium.spaces.Dict`` with:
- ``wall_features``: per-wall geometric and classification features
- ``room_features``: per-room geometric and label features
- ``assignment_mask``: binary mask of which walls/rooms are already assigned
- ``current_target``: features of the wall/room currently being decided
- ``candidate_features``: features of valid KG candidates for the current target
- ``candidate_mask``: binary mask of which candidate slots are populated

All coordinates are in inches (converted from PDF user units at 72 units/inch).
"""

from __future__ import annotations

import math

import numpy as np

from docs.interfaces.classified_wall_graph import (
    ClassifiedWallGraph,
    FireRating,
    WallClassification,
)
from docs.interfaces.graph_to_serializer import (
    Opening,
    Room,
    WallSegment,
    WallType,
)
from src.knowledge_graph.schema import Panel, PanelType, Pod

# ── Constants ───────────────────────────────────────────────────────────────

PDF_UNITS_PER_INCH: float = 72.0
"""PDF user units per inch (standard PostScript)."""

MAX_WALLS: int = 256
"""Maximum number of wall segments supported in one floor plan."""

MAX_ROOMS: int = 64
"""Maximum number of rooms supported in one floor plan."""

MAX_CANDIDATES: int = 32
"""Maximum number of KG candidates presented per decision step."""

WALL_FEATURE_DIM: int = 12
"""Feature vector dimension for a single wall segment."""

ROOM_FEATURE_DIM: int = 8
"""Feature vector dimension for a single room."""

CANDIDATE_PANEL_DIM: int = 10
"""Feature vector dimension for a panel candidate."""

CANDIDATE_POD_DIM: int = 10
"""Feature vector dimension for a pod candidate."""

# ── WallType/FireRating → numeric encoding ──────────────────────────────────

_WALL_TYPE_TO_INT: dict[WallType, int] = {
    WallType.LOAD_BEARING: 0,
    WallType.PARTITION: 1,
    WallType.EXTERIOR: 2,
    WallType.SHEAR: 3,
    WallType.CURTAIN: 4,
    WallType.UNKNOWN: 5,
}

_FIRE_RATING_TO_HOURS: dict[FireRating, float] = {
    FireRating.NONE: 0.0,
    FireRating.HOUR_1: 1.0,
    FireRating.HOUR_2: 2.0,
    FireRating.HOUR_3: 3.0,
    FireRating.HOUR_4: 4.0,
    FireRating.UNKNOWN: 0.0,
}

_WALL_TYPE_TO_PANEL_TYPE: dict[WallType, PanelType] = {
    WallType.LOAD_BEARING: PanelType.LOAD_BEARING,
    WallType.PARTITION: PanelType.PARTITION,
    WallType.SHEAR: PanelType.SHEAR,
    WallType.EXTERIOR: PanelType.ENVELOPE,
    WallType.CURTAIN: PanelType.ENVELOPE,
    WallType.UNKNOWN: PanelType.PARTITION,
}


def wall_type_to_panel_type(
    wall_type: WallType,
    fire_rating: FireRating,
) -> PanelType:
    """Map a WallType + FireRating to the appropriate KG PanelType.

    Fire-rated walls (rating > 0, non-UNKNOWN) use PanelType.FIRE_RATED
    regardless of wall type — the fire rating takes precedence.

    Args:
        wall_type: Structural wall classification from the classifier.
        fire_rating: Fire resistance rating from the classifier.

    Returns:
        The PanelType to query the KG with.
    """
    if fire_rating not in (FireRating.NONE, FireRating.UNKNOWN):
        return PanelType.FIRE_RATED
    return _WALL_TYPE_TO_PANEL_TYPE.get(wall_type, PanelType.PARTITION)


def fire_rating_to_hours(fire_rating: FireRating) -> float:
    """Convert FireRating enum to hours as a float."""
    return _FIRE_RATING_TO_HOURS.get(fire_rating, 0.0)


# ── Wall feature encoding ──────────────────────────────────────────────────


def encode_wall_segment(
    wall: WallSegment,
    classification: WallClassification,
    openings_on_wall: list[Opening],
    scale: float,
) -> np.ndarray:
    """Encode a single wall segment into a fixed-size feature vector.

    Features (WALL_FEATURE_DIM = 12):
        [0] length_inches          — wall length in inches
        [1] thickness_inches       — wall thickness in inches
        [2] angle_normalized       — angle / pi, in [0, 1)
        [3] wall_type_encoded      — integer encoding of WallType
        [4] fire_rating_hours      — fire rating in hours (0, 1, 2, 3, 4)
        [5] confidence             — classifier confidence [0, 1]
        [6] is_perimeter           — 1.0 if perimeter wall, else 0.0
        [7] num_openings           — count of openings on this wall
        [8] opening_coverage_ratio — fraction of wall length occluded by openings
        [9] start_x_inches         — start x coordinate in inches
        [10] start_y_inches        — start y coordinate in inches
        [11] cos_angle             — cosine of wall angle (for direction)

    Args:
        wall: The wall segment geometry.
        classification: Classification result for this wall.
        openings_on_wall: Openings detected on this wall.
        scale: Scale factor from PDF user units to millimeters.

    Returns:
        Feature vector of shape ``(WALL_FEATURE_DIM,)``, dtype float32.
    """
    # Convert from PDF user units to inches
    # PDF user units → mm (via scale), then mm → inches (/ 25.4)
    to_inches = scale / 25.4 if scale != 1.0 else 1.0 / PDF_UNITS_PER_INCH
    length_inches = wall.length * to_inches
    thickness_inches = wall.thickness * to_inches

    # Opening coverage: sum of opening widths / wall length
    opening_width_total = sum(o.width * to_inches for o in openings_on_wall)
    opening_coverage = min(opening_width_total / length_inches, 1.0) if length_inches > 0 else 0.0

    start_x = float(wall.start_coord[0]) * to_inches
    start_y = float(wall.start_coord[1]) * to_inches

    features = np.array(
        [
            length_inches,
            thickness_inches,
            wall.angle / math.pi,  # normalize to [0, 1)
            float(_WALL_TYPE_TO_INT.get(classification.wall_type, 5)),
            fire_rating_to_hours(classification.fire_rating),
            classification.confidence,
            1.0 if classification.is_perimeter else 0.0,
            float(len(openings_on_wall)),
            opening_coverage,
            start_x,
            start_y,
            math.cos(wall.angle),
        ],
        dtype=np.float32,
    )
    return features


# ── Room feature encoding ──────────────────────────────────────────────────

# Common room function labels → integer codes for encoding
_ROOM_LABEL_TO_INT: dict[str, int] = {
    "": 0,
    "bedroom": 1,
    "bathroom": 2,
    "kitchen": 3,
    "living": 4,
    "dining": 5,
    "office": 6,
    "laundry": 7,
    "closet": 8,
    "hallway": 9,
    "garage": 10,
    "mechanical": 11,
    "storage": 12,
    "entry": 13,
}


def encode_room(
    room: Room,
    nodes: np.ndarray,
    scale: float,
) -> np.ndarray:
    """Encode a single room into a fixed-size feature vector.

    Computes a bounding box from the room's boundary nodes to derive
    width and depth (needed for pod fitting queries).

    Features (ROOM_FEATURE_DIM = 8):
        [0] area_sq_inches       — room area in square inches
        [1] width_inches         — bounding box width in inches
        [2] depth_inches         — bounding box depth in inches
        [3] aspect_ratio         — width / depth (clamped to [0.1, 10])
        [4] num_boundary_edges   — number of wall segments bounding room
        [5] label_encoded        — integer encoding of room label
        [6] centroid_x_inches    — room centroid x coordinate
        [7] centroid_y_inches    — room centroid y coordinate

    Args:
        room: Room geometry and label.
        nodes: FinalizedGraph.nodes array, shape (N, 2) in PDF user units.
        scale: Scale factor from PDF user units to millimeters.

    Returns:
        Feature vector of shape ``(ROOM_FEATURE_DIM,)``, dtype float32.
    """
    to_inches = scale / 25.4 if scale != 1.0 else 1.0 / PDF_UNITS_PER_INCH

    # Get boundary node coordinates
    if room.boundary_nodes and all(0 <= idx < len(nodes) for idx in room.boundary_nodes):
        boundary_coords = nodes[room.boundary_nodes]
        min_xy = boundary_coords.min(axis=0) * to_inches
        max_xy = boundary_coords.max(axis=0) * to_inches
        width = max_xy[0] - min_xy[0]
        depth = max_xy[1] - min_xy[1]
        centroid = (min_xy + max_xy) / 2.0
    else:
        width = 0.0
        depth = 0.0
        centroid = np.array([0.0, 0.0])

    area_sq_inches = room.area * (to_inches ** 2)

    # Ensure non-zero for aspect ratio (1 inch floor — 0.01" is nonsensical)
    width_for_aspect = max(width, 1.0)
    depth_for_aspect = max(depth, 1.0)
    aspect = np.clip(width_for_aspect / depth_for_aspect, 0.1, 10.0)

    label_code = _ROOM_LABEL_TO_INT.get(room.label.lower().strip(), 0)

    features = np.array(
        [
            area_sq_inches,
            width,
            depth,
            float(aspect),
            float(len(room.boundary_edges)),
            float(label_code),
            float(centroid[0]),
            float(centroid[1]),
        ],
        dtype=np.float32,
    )
    return features


def get_room_dims_inches(
    room: Room,
    nodes: np.ndarray,
    scale: float,
) -> tuple[float, float]:
    """Compute room bounding box width and depth in inches.

    Args:
        room: Room with boundary nodes.
        nodes: FinalizedGraph.nodes array.
        scale: Scale factor from PDF user units to millimeters.

    Returns:
        ``(width_inches, depth_inches)`` tuple.
    """
    to_inches = scale / 25.4 if scale != 1.0 else 1.0 / PDF_UNITS_PER_INCH

    if not room.boundary_nodes or not all(
        0 <= idx < len(nodes) for idx in room.boundary_nodes
    ):
        return (0.0, 0.0)

    boundary_coords = nodes[room.boundary_nodes]
    min_xy = boundary_coords.min(axis=0) * to_inches
    max_xy = boundary_coords.max(axis=0) * to_inches
    width = max(float(max_xy[0] - min_xy[0]), 0.0)
    depth = max(float(max_xy[1] - min_xy[1]), 0.0)
    return (width, depth)


# ── Panel/Pod candidate encoding ───────────────────────────────────────────


def encode_panel_candidate(panel: Panel) -> np.ndarray:
    """Encode a Panel from the KG into a candidate feature vector.

    Features (CANDIDATE_PANEL_DIM = 10):
        [0] min_length_inches
        [1] max_length_inches
        [2] gauge
        [3] stud_depth_inches
        [4] stud_spacing_inches
        [5] fire_rating_hours
        [6] load_capacity_plf
        [7] weight_per_foot_lbs
        [8] unit_cost_per_foot
        [9] panel_type_encoded (same int encoding as PanelType ordinal)

    Args:
        panel: Panel entity from the KG.

    Returns:
        Feature vector of shape ``(CANDIDATE_PANEL_DIM,)``, dtype float32.
    """
    panel_type_int = list(PanelType).index(panel.panel_type)
    return np.array(
        [
            panel.min_length_inches,
            panel.max_length_inches,
            float(panel.gauge),
            panel.stud_depth_inches,
            panel.stud_spacing_inches,
            panel.fire_rating_hours,
            panel.load_capacity_plf,
            panel.weight_per_foot_lbs,
            panel.unit_cost_per_foot,
            float(panel_type_int),
        ],
        dtype=np.float32,
    )


def encode_pod_candidate(pod: Pod) -> np.ndarray:
    """Encode a Pod from the KG into a candidate feature vector.

    Features (CANDIDATE_POD_DIM = 10):
        [0] width_inches
        [1] depth_inches
        [2] height_inches
        [3] min_room_width_inches
        [4] min_room_depth_inches
        [5] clearance_inches
        [6] num_included_trades
        [7] weight_lbs
        [8] unit_cost (in thousands, for normalization)
        [9] lead_time_days

    Args:
        pod: Pod entity from the KG.

    Returns:
        Feature vector of shape ``(CANDIDATE_POD_DIM,)``, dtype float32.
    """
    return np.array(
        [
            pod.width_inches,
            pod.depth_inches,
            pod.height_inches,
            pod.min_room_width_inches,
            pod.min_room_depth_inches,
            pod.clearance_inches,
            float(len(pod.included_trades)),
            pod.weight_lbs,
            pod.unit_cost / 1000.0,  # normalize to thousands
            float(pod.lead_time_days),
        ],
        dtype=np.float32,
    )


# ── Full observation encoding ──────────────────────────────────────────────


def encode_observation(
    classified_graph: ClassifiedWallGraph,
    wall_assignments: dict[int, list[tuple[str, float]]],
    room_assignments: dict[int, str],
    current_wall_idx: int | None,
    current_room_idx: int | None,
    panel_candidates: list[Panel],
    pod_candidates: list[Pod],
    phase: str,
) -> dict[str, np.ndarray]:
    """Build the full observation dictionary for the DRL policy.

    This is the primary observation function called by ``PanelizationEnv.step``
    and ``PanelizationEnv.reset``.

    Args:
        classified_graph: The input classified wall graph.
        wall_assignments: Current panel assignments — maps edge_id to
            list of ``(panel_sku, cut_length)`` tuples.
        room_assignments: Current pod assignments — maps room_id to pod SKU.
        current_wall_idx: Index into wall_segments of the wall being decided
            (None if not in panelization phase).
        current_room_idx: Index into rooms of the room being decided
            (None if not in placement phase).
        panel_candidates: Valid panels from KG for the current wall.
        pod_candidates: Valid pods from KG for the current room.
        phase: ``"panelization"`` or ``"placement"``.

    Returns:
        Dictionary of numpy arrays matching the observation space.
    """
    graph = classified_graph.graph
    walls = graph.wall_segments
    rooms = [r for r in graph.rooms if not r.is_exterior]
    classifications = classified_graph.classifications
    openings = graph.openings
    scale = graph.scale_factor

    n_walls = min(len(walls), MAX_WALLS)
    n_rooms = min(len(rooms), MAX_ROOMS)

    # Build opening lookup: edge_id → list of openings
    opening_map: dict[int, list[Opening]] = {}
    for opening in openings:
        opening_map.setdefault(opening.wall_edge_id, []).append(opening)

    # ── Wall features (MAX_WALLS x WALL_FEATURE_DIM) ──
    wall_features = np.zeros((MAX_WALLS, WALL_FEATURE_DIM), dtype=np.float32)
    for i in range(n_walls):
        wall = walls[i]
        cls = classifications[i]
        wall_openings = opening_map.get(wall.edge_id, [])
        wall_features[i] = encode_wall_segment(wall, cls, wall_openings, scale)

    # ── Room features (MAX_ROOMS x ROOM_FEATURE_DIM) ──
    room_features = np.zeros((MAX_ROOMS, ROOM_FEATURE_DIM), dtype=np.float32)
    for i in range(n_rooms):
        room_features[i] = encode_room(rooms[i], graph.nodes, scale)

    # ── Assignment masks ──
    wall_assigned = np.zeros(MAX_WALLS, dtype=np.float32)
    for i in range(n_walls):
        if walls[i].edge_id in wall_assignments:
            wall_assigned[i] = 1.0

    room_assigned = np.zeros(MAX_ROOMS, dtype=np.float32)
    for i in range(n_rooms):
        if rooms[i].room_id in room_assignments:
            room_assigned[i] = 1.0

    # ── Current target features ──
    current_target = np.zeros(max(WALL_FEATURE_DIM, ROOM_FEATURE_DIM), dtype=np.float32)
    if phase == "panelization" and current_wall_idx is not None and current_wall_idx < n_walls:
        wall = walls[current_wall_idx]
        cls = classifications[current_wall_idx]
        wall_openings = opening_map.get(wall.edge_id, [])
        current_target[:WALL_FEATURE_DIM] = encode_wall_segment(
            wall, cls, wall_openings, scale,
        )
    elif phase == "placement" and current_room_idx is not None and current_room_idx < n_rooms:
        current_target[:ROOM_FEATURE_DIM] = encode_room(
            rooms[current_room_idx], graph.nodes, scale,
        )

    # ── Candidate features ──
    # Always use the max of panel/pod dims so the tensor shape is constant
    # across phases (matches the observation_space defined in env.py).
    candidate_dim = max(CANDIDATE_PANEL_DIM, CANDIDATE_POD_DIM)
    candidate_features = np.zeros((MAX_CANDIDATES, candidate_dim), dtype=np.float32)
    candidate_mask = np.zeros(MAX_CANDIDATES, dtype=np.float32)

    if phase == "panelization":
        for i, panel in enumerate(panel_candidates[:MAX_CANDIDATES]):
            encoded = encode_panel_candidate(panel)
            candidate_features[i, :len(encoded)] = encoded
            candidate_mask[i] = 1.0
    else:
        for i, pod in enumerate(pod_candidates[:MAX_CANDIDATES]):
            encoded = encode_pod_candidate(pod)
            candidate_features[i, :len(encoded)] = encoded
            candidate_mask[i] = 1.0

    # ── Phase encoding ──
    phase_vec = np.array(
        [1.0, 0.0] if phase == "panelization" else [0.0, 1.0],
        dtype=np.float32,
    )

    # ── Progress ──
    # Fraction of walls/rooms already assigned
    progress = np.array(
        [
            float(np.sum(wall_assigned)) / max(n_walls, 1),
            float(np.sum(room_assigned)) / max(n_rooms, 1),
        ],
        dtype=np.float32,
    )

    return {
        "wall_features": wall_features,
        "room_features": room_features,
        "wall_assigned": wall_assigned,
        "room_assigned": room_assigned,
        "current_target": current_target,
        "candidate_features": candidate_features,
        "candidate_mask": candidate_mask,
        "phase": phase_vec,
        "progress": progress,
    }