"""Gymnasium-compatible environment for wall panelization and pod placement.

DRL-001: Implements the ``PanelizationEnv`` which wraps a ClassifiedWallGraph
and a Knowledge Graph store, exposing the standard Gymnasium interface
(``reset``, ``step``, ``observation_space``, ``action_space``).

Episode structure:
    1. **Panelization phase**: iterate over walls sequentially. For each wall,
       the agent selects a panel from KG-filtered candidates (or skips).
    2. **Placement phase**: iterate over rooms sequentially. For each room,
       the agent selects a pod from KG-filtered candidates with an orientation
       (or skips).
    3. Episode terminates after all walls and rooms have been processed.

The environment is deterministic given a fixed ClassifiedWallGraph — randomness
only enters through initial state sampling during training (via different PDFs).
"""

from __future__ import annotations

import logging
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from docs.interfaces.classified_wall_graph import ClassifiedWallGraph, WallClassification
from docs.interfaces.graph_to_serializer import Opening, Room, WallSegment
from src.drl.actions import (
    PANELIZATION_ACTION_SIZE,
    PLACEMENT_ACTION_SIZE,
    PanelAction,
    PlacementAction,
    compute_panel_action_mask,
    compute_placement_action_mask,
    decode_panel_action,
    decode_placement_action,
    get_panel_candidates,
    get_pod_candidates,
)
from src.drl.constraints import (
    JunctionInfo,
    WallSubSegment,
    compute_junction_map,
    compute_junction_penalties,
    compute_wall_sub_segments,
    get_corner_thickness_deduction,
)
from src.drl.reward import RewardBreakdown, RewardWeights, compute_reward
from src.drl.state import (
    CANDIDATE_PANEL_DIM,
    CANDIDATE_POD_DIM,
    MAX_CANDIDATES,
    MAX_ROOMS,
    MAX_WALLS,
    ROOM_FEATURE_DIM,
    WALL_FEATURE_DIM,
    encode_observation,
    get_room_dims_inches,
    wall_type_to_panel_type,
)
from src.knowledge_graph.loader import KnowledgeGraphStore
from src.knowledge_graph.query import PanelRecommendation
from src.knowledge_graph.schema import Panel, PanelType, Pod

logger = logging.getLogger(__name__)

# Maximum candidate feature dim (union of panel and pod dims)
_CANDIDATE_DIM: int = max(CANDIDATE_PANEL_DIM, CANDIDATE_POD_DIM)
_TARGET_DIM: int = max(WALL_FEATURE_DIM, ROOM_FEATURE_DIM)


class PanelizationEnv(gym.Env):
    """Gymnasium environment for CFS wall panelization and pod placement.

    The environment processes a classified wall graph through two sequential
    phases:

    **Phase 1 — Panelization:** For each wall segment, the agent selects a
    panel configuration from KG-validated candidates. The action is a
    discrete index (0 = skip, 1..N = candidate selection).

    **Phase 2 — Placement:** For each room, the agent selects a pod and
    orientation from KG-validated candidates. The action encodes both
    pod index and orientation (normal or rotated 90 degrees).

    The action space uses the larger of the two phase sizes
    (``max(PANELIZATION_ACTION_SIZE, PLACEMENT_ACTION_SIZE)``), with
    invalid actions masked out via ``action_masks()``.

    Attributes:
        classified_graph: The input ClassifiedWallGraph.
        store: Knowledge Graph store for deterministic catalog lookups.
        reward_weights: Configurable reward component weights.
        phase: Current phase — ``"panelization"`` or ``"placement"``.
        current_wall_idx: Index of wall being decided in panelization phase.
        current_room_idx: Index of room being decided in placement phase.
        wall_assignments: Maps edge_id to list of (sku, cut_length) tuples.
        room_assignments: Maps room_id to pod SKU.
        wall_remaining_inches: Maps edge_id to uncovered length remaining
            on each wall (DRL-005).
        current_sub_segment_idx: Index of the current sub-segment within
            the current wall when openings split it (DRL-007).
        room_remaining_area: Maps room_id to remaining area available for
            pods in square inches (DRL-006).
        room_pod_placements: Maps room_id to list of placed pod tuples
            ``(sku, x, y, rotated)`` (DRL-006).
        junction_map: Pre-computed wall junction information (DRL-008).
    """

    metadata: dict[str, Any] = {"render_modes": ["human"]}

    def __init__(
        self,
        classified_graph: ClassifiedWallGraph,
        store: KnowledgeGraphStore,
        reward_weights: RewardWeights | None = None,
    ) -> None:
        """Initialize the panelization environment.

        Args:
            classified_graph: Classified wall graph from the classifier agent.
            store: Loaded Knowledge Graph store with product catalog.
            reward_weights: Optional custom reward weights.
        """
        super().__init__()

        self.classified_graph = classified_graph
        self.store = store
        self.reward_weights = reward_weights or RewardWeights()

        # Extract walls and non-exterior rooms
        self._walls: list[WallSegment] = list(classified_graph.graph.wall_segments)
        self._classifications: list[WallClassification] = list(classified_graph.classifications)
        self._rooms: list[Room] = [r for r in classified_graph.graph.rooms if not r.is_exterior]
        self._openings: list[Opening] = list(classified_graph.graph.openings)
        self._scale: float = classified_graph.graph.scale_factor
        self._nodes: np.ndarray = classified_graph.graph.nodes

        # Precompute opening map: edge_id → list of openings
        self._opening_map: dict[int, list[Opening]] = {}
        for opening in self._openings:
            self._opening_map.setdefault(opening.wall_edge_id, []).append(opening)

        # DRL-007: Precompute sub-segments for walls with openings
        self._sub_segment_map: dict[int, list[WallSubSegment]] = {}
        for wall in self._walls:
            wall_openings = self._opening_map.get(wall.edge_id, [])
            self._sub_segment_map[wall.edge_id] = compute_wall_sub_segments(
                wall,
                wall_openings,
                self._scale,
            )

        # DRL-008: Precompute junction map
        self._junction_map: dict[int, JunctionInfo] = compute_junction_map(self._walls)

        self._n_walls: int = len(self._walls)
        self._n_rooms: int = len(self._rooms)

        # ── Action space ──
        # Use the max of both phase action sizes so the space is fixed
        self._action_size: int = max(PANELIZATION_ACTION_SIZE, PLACEMENT_ACTION_SIZE)
        self.action_space: spaces.Discrete = spaces.Discrete(self._action_size)

        # ── Observation space ──
        self.observation_space: spaces.Dict = spaces.Dict(
            {
                "wall_features": spaces.Box(
                    low=-1e6,
                    high=1e6,
                    shape=(MAX_WALLS, WALL_FEATURE_DIM),
                    dtype=np.float32,
                ),
                "room_features": spaces.Box(
                    low=-1e6,
                    high=1e6,
                    shape=(MAX_ROOMS, ROOM_FEATURE_DIM),
                    dtype=np.float32,
                ),
                "wall_assigned": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(MAX_WALLS,),
                    dtype=np.float32,
                ),
                "room_assigned": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(MAX_ROOMS,),
                    dtype=np.float32,
                ),
                "current_target": spaces.Box(
                    low=-1e6,
                    high=1e6,
                    shape=(_TARGET_DIM,),
                    dtype=np.float32,
                ),
                "candidate_features": spaces.Box(
                    low=-1e6,
                    high=1e6,
                    shape=(MAX_CANDIDATES, _CANDIDATE_DIM),
                    dtype=np.float32,
                ),
                "candidate_mask": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(MAX_CANDIDATES,),
                    dtype=np.float32,
                ),
                "phase": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(2,),
                    dtype=np.float32,
                ),
                "progress": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(2,),
                    dtype=np.float32,
                ),
            }
        )

        # ── Mutable state (reset in reset()) ──
        self.phase: str = "panelization"
        self.current_wall_idx: int = 0
        self.current_room_idx: int = 0
        self.wall_assignments: dict[int, list[tuple[str, float]]] = {}
        self.room_assignments: dict[int, str] = {}

        # DRL-005: Multi-panel state — remaining uncovered length per wall
        self.wall_remaining_inches: dict[int, float] = {}
        # DRL-007: Current sub-segment index within the current wall
        self.current_sub_segment_idx: int = 0

        # DRL-006: Multi-pod state — remaining area and placements per room
        self.room_remaining_area: dict[int, float] = {}
        self.room_pod_placements: dict[int, list[tuple[str, float, float, bool]]] = {}

        # Cached candidates for the current decision point
        self._panel_candidates: list[PanelRecommendation] = []
        self._pod_candidates: list[Pod] = []
        self._action_mask: np.ndarray = np.zeros(self._action_size, dtype=np.float32)

        # Episode metrics
        self._episode_rewards: list[RewardBreakdown] = []
        self._step_count: int = 0

    # ── Gymnasium API ──────────────────────────────────────────────────────

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        """Reset the environment to the beginning of a new episode.

        Optionally accepts a new ClassifiedWallGraph via ``options``:
            ``options={"classified_graph": new_graph}``

        Args:
            seed: Random seed (unused — environment is deterministic).
            options: Optional dict with ``"classified_graph"`` and/or
                ``"store"`` to replace the current inputs.

        Returns:
            Tuple of ``(observation, info)``.
        """
        super().reset(seed=seed)

        # Allow replacing the graph/store between episodes
        if options is not None:
            if "classified_graph" in options:
                self.classified_graph = options["classified_graph"]
                self._walls = list(self.classified_graph.graph.wall_segments)
                self._classifications = list(self.classified_graph.classifications)
                self._rooms = [r for r in self.classified_graph.graph.rooms if not r.is_exterior]
                self._openings = list(self.classified_graph.graph.openings)
                self._scale = self.classified_graph.graph.scale_factor
                self._nodes = self.classified_graph.graph.nodes
                self._opening_map = {}
                for opening in self._openings:
                    self._opening_map.setdefault(opening.wall_edge_id, []).append(opening)
                # DRL-007: Recompute sub-segments
                self._sub_segment_map = {}
                for wall in self._walls:
                    wall_openings = self._opening_map.get(wall.edge_id, [])
                    self._sub_segment_map[wall.edge_id] = compute_wall_sub_segments(
                        wall,
                        wall_openings,
                        self._scale,
                    )
                # DRL-008: Recompute junction map
                self._junction_map = compute_junction_map(self._walls)
                self._n_walls = len(self._walls)
                self._n_rooms = len(self._rooms)
            if "store" in options:
                self.store = options["store"]

        # Reset state
        self.phase = "panelization"
        self.current_wall_idx = 0
        self.current_room_idx = 0
        self.wall_assignments = {}
        self.room_assignments = {}
        # DRL-005: Reset multi-panel state
        self.wall_remaining_inches = {}
        self.current_sub_segment_idx = 0
        # DRL-006: Reset multi-pod state
        self.room_remaining_area = {}
        self.room_pod_placements = {}
        self._episode_rewards = []
        self._step_count = 0

        # Handle edge case: no walls → skip to placement
        if self._n_walls == 0:
            self.phase = "placement"

        # Populate candidates for the first decision
        self._refresh_candidates()

        obs = self._get_observation()
        info = self._get_info()
        return obs, info

    def step(
        self,
        action: int,
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        """Execute one action in the environment.

        Processes the action for the current wall (panelization) or room
        (placement), computes the reward, advances to the next decision
        point, and returns the new observation.

        Args:
            action: Discrete action index from the policy.

        Returns:
            Tuple of ``(observation, reward, terminated, truncated, info)``.
            ``terminated`` is True when all walls and rooms have been processed.
            ``truncated`` is always False (no time limit).
        """
        self._step_count += 1

        # ── Execute action ──
        panel_action: PanelAction | None = None
        placement_action: PlacementAction | None = None
        wall_length_inches: float = 0.0
        wall_panel_type: PanelType | None = None
        room_width_inches: float = 0.0
        room_depth_inches: float = 0.0
        junction_penalty: float = 0.0
        junction_violations: list[str] = []

        if self.phase == "panelization":
            # Clamp action to panelization range
            clamped_action = min(action, PANELIZATION_ACTION_SIZE - 1)
            wall = self._walls[self.current_wall_idx]
            classification = self._classifications[self.current_wall_idx]

            panel_action = decode_panel_action(
                clamped_action,
                wall,
                self._panel_candidates,
            )

            # ── DRL-005: Multi-panel wall handling ──
            # Compute the effective panelizable length for this step.
            # If this wall has sub-segments (DRL-007), use the current
            # sub-segment's length; otherwise use the full wall minus openings.
            to_inches = self._scale / 25.4 if self._scale != 1.0 else 1.0 / 72.0

            sub_segments = self._sub_segment_map.get(wall.edge_id, [])
            if sub_segments and self.current_sub_segment_idx < len(sub_segments):
                current_sub = sub_segments[self.current_sub_segment_idx]
                effective_length = current_sub.length_inches
            else:
                full_length = wall.length * to_inches
                opening_width = sum(
                    o.width * to_inches for o in self._opening_map.get(wall.edge_id, [])
                )
                effective_length = max(full_length - opening_width, 0.0)

            # Apply corner thickness deduction (DRL-008)
            corner_deduction = get_corner_thickness_deduction(
                wall.edge_id,
                self._junction_map,
                self._walls,
                self._scale,
            )
            effective_length = max(effective_length - corner_deduction, 0.0)

            # Track remaining uncovered length
            if wall.edge_id not in self.wall_remaining_inches:
                self.wall_remaining_inches[wall.edge_id] = effective_length

            # Record assignment and update remaining length
            if panel_action.skip:
                # Skip means the agent chose not to panelize this wall/segment.
                # Set remaining to 0 so _advance moves to the next wall.
                self.wall_remaining_inches[wall.edge_id] = 0.0
            elif panel_action.panel_assignments:
                # Accumulate panel assignments for multi-panel walls
                existing = self.wall_assignments.get(wall.edge_id, [])
                existing.extend(panel_action.panel_assignments)
                self.wall_assignments[wall.edge_id] = existing

                # Compute how much length these panels cover
                covered = sum(cl for _, cl in panel_action.panel_assignments)
                self.wall_remaining_inches[wall.edge_id] = max(
                    self.wall_remaining_inches[wall.edge_id] - covered,
                    0.0,
                )

                # Track splice connections between panels
                if (
                    len(existing) > 1
                    and panel_action.recommendation is not None
                    and panel_action.recommendation.requires_splice
                ):
                    logger.debug(
                        "Wall %d: splice required between panels (total panels: %d)",
                        wall.edge_id,
                        len(existing),
                    )

            wall_length_inches = effective_length
            wall_panel_type = wall_type_to_panel_type(
                classification.wall_type,
                classification.fire_rating,
            )

            # ── DRL-008: Junction penalty ──
            junction_penalty, junction_violations = compute_junction_penalties(
                wall.edge_id,
                panel_action.panel if not panel_action.skip else None,
                self._junction_map,
                self.wall_assignments,
                self._walls,
                self.store,
            )

        elif self.phase == "placement":
            # Clamp action to placement range
            clamped_action = min(action, PLACEMENT_ACTION_SIZE - 1)
            room = self._rooms[self.current_room_idx]

            # Compute room dimensions and centroid for pod positioning
            room_w, room_d = get_room_dims_inches(room, self._nodes, self._scale)
            room_width_inches = room_w
            room_depth_inches = room_d

            to_inches = self._scale / 25.4 if self._scale != 1.0 else 1.0 / 72.0

            # ── DRL-006: Pod position calculation ──
            # Compute placement coordinates within the room. If multiple pods
            # are being placed, offset from the base centroid.
            if room.boundary_nodes:
                boundary_coords = self._nodes[room.boundary_nodes]
                min_xy = boundary_coords.min(axis=0) * to_inches
                max_xy = boundary_coords.max(axis=0) * to_inches
                centroid = (min_xy + max_xy) / 2.0
                room_centroid = (float(centroid[0]), float(centroid[1]))
            else:
                room_centroid = (0.0, 0.0)
                min_xy = np.array([0.0, 0.0])
                max_xy = np.array([0.0, 0.0])

            # Initialize remaining area tracking
            if room.room_id not in self.room_remaining_area:
                self.room_remaining_area[room.room_id] = room_w * room_d
            if room.room_id not in self.room_pod_placements:
                self.room_pod_placements[room.room_id] = []

            # Adjust centroid for multi-pod placement — offset along the
            # longer room axis based on how many pods are already placed
            existing_placements = self.room_pod_placements[room.room_id]
            if existing_placements:
                n_placed = len(existing_placements)
                # Stack pods along the longer axis
                if room_w >= room_d:
                    # Stack along x-axis
                    slot_width = room_w / (n_placed + 2)
                    x_offset = min_xy[0] + slot_width * (n_placed + 1)
                    room_centroid = (float(x_offset), room_centroid[1])
                else:
                    # Stack along y-axis
                    slot_depth = room_d / (n_placed + 2)
                    y_offset = min_xy[1] + slot_depth * (n_placed + 1)
                    room_centroid = (room_centroid[0], float(y_offset))

            placement_action = decode_placement_action(
                clamped_action,
                room.room_id,
                self._pod_candidates,
                room_centroid,
            )

            # Record assignment and update remaining area
            if placement_action.skip:
                # Skip means the agent chose to stop placing in this room.
                # Set remaining to 0 so _advance moves to the next room.
                self.room_remaining_area[room.room_id] = 0.0
            elif placement_action.pod is not None:
                pod = placement_action.pod
                # First pod assigned records the room assignment SKU
                if room.room_id not in self.room_assignments:
                    self.room_assignments[room.room_id] = pod.sku

                # Track the placement with position
                self.room_pod_placements[room.room_id].append(
                    (
                        pod.sku,
                        placement_action.position_x,
                        placement_action.position_y,
                        placement_action.rotated,
                    )
                )

                # Update remaining area (use clearance-inclusive footprint)
                # Include clearance area
                clearance_area = (pod.width_inches + 2 * pod.clearance_inches) * (
                    pod.depth_inches + 2 * pod.clearance_inches
                )
                self.room_remaining_area[room.room_id] = max(
                    self.room_remaining_area[room.room_id] - clearance_area,
                    0.0,
                )

        # ── Advance to next decision point ──
        self._advance()

        # ── Check termination ──
        terminated = self._is_done()

        # ── Compute reward ──
        reward_breakdown = compute_reward(
            panel_action=panel_action,
            placement_action=placement_action,
            wall_length_inches=wall_length_inches,
            wall_panel_type=wall_panel_type,
            room_width_inches=room_width_inches,
            room_depth_inches=room_depth_inches,
            store=self.store,
            walls_assigned=len(self.wall_assignments),
            total_walls=self._n_walls,
            rooms_assigned=len(self.room_assignments),
            total_rooms=self._n_rooms,
            is_terminal=terminated,
            weights=self.reward_weights,
        )

        # ── DRL-008: Apply junction penalty to reward ──
        if junction_penalty < 0.0:
            reward_breakdown.violation += junction_penalty
            reward_breakdown.violations.extend(junction_violations)
            reward_breakdown.total += self.reward_weights.violation * junction_penalty
            reward_breakdown.info["junction_penalty"] = junction_penalty

        self._episode_rewards.append(reward_breakdown)
        reward = reward_breakdown.total

        # ── Refresh candidates for the next step ──
        if not terminated:
            self._refresh_candidates()

        obs = self._get_observation()
        info = self._get_info()
        info["reward_breakdown"] = reward_breakdown

        return obs, reward, terminated, False, info

    def action_masks(self) -> np.ndarray:
        """Return the current action mask.

        Used by action-masking-aware policies (e.g., MaskablePPO from
        SB3-contrib). Invalid actions have mask value 0.

        Returns:
            Binary mask of shape ``(action_size,)``, dtype float32.
        """
        return self._action_mask.copy()

    # ── Internal helpers ───────────────────────────────────────────────────

    def _advance(self) -> None:
        """Advance to the next decision point.

        DRL-005/DRL-007: If the current wall still has uncovered sub-segments
        or remaining length that needs more panels, stay on the same wall
        and advance to the next sub-segment. Otherwise move to the next wall.

        DRL-006: If the current room still has space for more pods and
        valid candidates exist, stay on the same room. Otherwise advance.

        Moves to the next wall in panelization phase, or transitions to
        placement phase when all walls are done, then advances through rooms.
        """
        if self.phase == "panelization":
            if self.current_wall_idx < self._n_walls:
                wall = self._walls[self.current_wall_idx]
                sub_segments = self._sub_segment_map.get(wall.edge_id, [])

                # DRL-007: Check if there are more sub-segments to panelize
                if self.current_sub_segment_idx < len(sub_segments) - 1:
                    self.current_sub_segment_idx += 1
                    # Re-initialize remaining length for the new sub-segment
                    new_sub = sub_segments[self.current_sub_segment_idx]
                    self.wall_remaining_inches[wall.edge_id] = new_sub.length_inches
                    return

                # DRL-005: Check if the wall still has significant uncovered length
                remaining = self.wall_remaining_inches.get(wall.edge_id, 0.0)
                if remaining > 6.0:  # More than 6 inches remaining needs another panel
                    # Stay on same wall for another panel step
                    # Reset sub-segment to 0 (for the remaining portion)
                    self.current_sub_segment_idx = 0
                    return

            # Move to next wall
            self.current_wall_idx += 1
            self.current_sub_segment_idx = 0
            if self.current_wall_idx >= self._n_walls:
                self.phase = "placement"
                self.current_room_idx = 0
        elif self.phase == "placement":
            if self.current_room_idx < self._n_rooms:
                room = self._rooms[self.current_room_idx]
                remaining_area = self.room_remaining_area.get(room.room_id, 0.0)

                # DRL-006: Check if more pods can fit in this room.
                # Only stay if there is meaningful remaining area and the last
                # action was not a skip (skip means the agent chose to stop
                # placing in this room).
                placements = self.room_pod_placements.get(room.room_id, [])
                if placements and remaining_area > 0.0:
                    # Check if any pod candidate could still fit in the
                    # remaining effective dimensions. Use a conservative
                    # estimate: sqrt(remaining_area) as both width and depth.
                    effective_dim = remaining_area**0.5
                    if effective_dim > 24.0:  # At least 2 feet remaining
                        # Stay on same room for another pod placement
                        return

            # Move to next room
            self.current_room_idx += 1

    def _is_done(self) -> bool:
        """Check if the episode is complete."""
        if self.phase == "panelization":
            return False
        return self.current_room_idx >= self._n_rooms

    def _refresh_candidates(self) -> None:
        """Query the KG for candidates at the current decision point.

        DRL-005: When a wall has remaining uncovered length, queries the KG
        for panels that fit the remaining length (not the full wall).

        DRL-007: When a wall has sub-segments due to openings, queries the
        KG for panels that fit the current sub-segment's length.

        DRL-006: When a room has remaining area after a pod placement,
        queries the KG for pods that fit the reduced available space.

        Populates ``_panel_candidates`` or ``_pod_candidates`` and updates
        ``_action_mask`` accordingly.
        """
        # Reset action mask
        self._action_mask = np.zeros(self._action_size, dtype=np.float32)
        self._panel_candidates = []
        self._pod_candidates = []

        if self.phase == "panelization" and self.current_wall_idx < self._n_walls:
            wall = self._walls[self.current_wall_idx]
            classification = self._classifications[self.current_wall_idx]

            # DRL-005/DRL-007: Determine the effective length for candidate query.
            # If there is remaining uncovered length (multi-panel), use that.
            # Otherwise, if there are sub-segments, use the current sub-segment.
            remaining = self.wall_remaining_inches.get(wall.edge_id)
            sub_segments = self._sub_segment_map.get(wall.edge_id, [])

            # Compute the effective length that the reward function will
            # evaluate against — candidates must be generated for this
            # exact length so recommendations don't overshoot.
            to_inches = self._scale / 25.4 if self._scale != 1.0 else 1.0 / 72.0

            if remaining is not None and remaining > 0.0:
                eff_length = remaining
            elif sub_segments and self.current_sub_segment_idx < len(sub_segments):
                eff_length = sub_segments[self.current_sub_segment_idx].length_inches
            else:
                full_length = wall.length * to_inches
                opening_width = sum(
                    o.width * to_inches
                    for o in self._opening_map.get(wall.edge_id, [])
                )
                eff_length = max(full_length - opening_width, 0.0)

            # Apply corner thickness deduction (DRL-008)
            corner_deduction = get_corner_thickness_deduction(
                wall.edge_id,
                self._junction_map,
                self._walls,
                self._scale,
            )
            eff_length = max(eff_length - corner_deduction, 0.0)

            self._panel_candidates = get_panel_candidates(
                self.store,
                wall,
                classification,
                [],  # openings already accounted for in eff_length
                self._scale,
                effective_length_inches=eff_length,
            )

            # Build action mask (panelization range)
            panel_mask = compute_panel_action_mask(len(self._panel_candidates))
            # Pad to full action size
            self._action_mask[:PANELIZATION_ACTION_SIZE] = panel_mask

        elif self.phase == "placement" and self.current_room_idx < self._n_rooms:
            room = self._rooms[self.current_room_idx]
            room_w, room_d = get_room_dims_inches(room, self._nodes, self._scale)

            # DRL-006: If pods have already been placed, reduce effective
            # room dimensions for subsequent pod queries.
            placements = self.room_pod_placements.get(room.room_id, [])
            if placements:
                remaining_area = self.room_remaining_area.get(
                    room.room_id,
                    room_w * room_d,
                )
                # Estimate remaining dimensions — assume remaining area is
                # roughly rectangular along the longer axis.
                if room_w >= room_d:
                    # Pods stacked along x-axis: reduce effective width
                    effective_w = remaining_area / max(room_d, 0.01)
                    room_w = max(effective_w, 0.0)
                else:
                    # Pods stacked along y-axis: reduce effective depth
                    effective_d = remaining_area / max(room_w, 0.01)
                    room_d = max(effective_d, 0.0)

            self._pod_candidates = get_pod_candidates(
                self.store,
                room_width_inches=room_w,
                room_depth_inches=room_d,
                room_label=room.label,
            )

            # Build action mask (placement range)
            placement_mask = compute_placement_action_mask(
                len(self._pod_candidates),
                room_w,
                room_d,
                self._pod_candidates,
            )
            # Pad to full action size
            self._action_mask[:PLACEMENT_ACTION_SIZE] = placement_mask

    def _get_observation(self) -> dict[str, np.ndarray]:
        """Build the current observation dictionary."""
        # Extract Panel objects from PanelRecommendation for state encoding
        panels_for_encoding: list[Panel] = [rec.panel for rec in self._panel_candidates]

        return encode_observation(
            classified_graph=self.classified_graph,
            wall_assignments=self.wall_assignments,
            room_assignments=self.room_assignments,
            current_wall_idx=(self.current_wall_idx if self.phase == "panelization" else None),
            current_room_idx=(self.current_room_idx if self.phase == "placement" else None),
            panel_candidates=panels_for_encoding,
            pod_candidates=self._pod_candidates,
            phase=self.phase,
        )

    def _get_info(self) -> dict[str, Any]:
        """Build the info dictionary for the current step."""
        info: dict[str, Any] = {
            "phase": self.phase,
            "step": self._step_count,
            "walls_assigned": len(self.wall_assignments),
            "rooms_assigned": len(self.room_assignments),
            "total_walls": self._n_walls,
            "total_rooms": self._n_rooms,
            "num_panel_candidates": len(self._panel_candidates),
            "num_pod_candidates": len(self._pod_candidates),
        }
        if self.phase == "panelization" and self.current_wall_idx < self._n_walls:
            wall = self._walls[self.current_wall_idx]
            info["current_wall_edge_id"] = wall.edge_id
            # DRL-005: Include remaining wall length
            info["wall_remaining_inches"] = self.wall_remaining_inches.get(
                wall.edge_id,
                0.0,
            )
            # DRL-007: Include sub-segment info
            sub_segments = self._sub_segment_map.get(wall.edge_id, [])
            info["num_sub_segments"] = len(sub_segments)
            info["current_sub_segment_idx"] = self.current_sub_segment_idx
            # DRL-008: Include junction info
            junction_nodes = []
            for node_id in (wall.start_node, wall.end_node):
                if node_id in self._junction_map:
                    junction_nodes.append(node_id)
            info["junction_nodes"] = junction_nodes
        if self.phase == "placement" and self.current_room_idx < self._n_rooms:
            room = self._rooms[self.current_room_idx]
            info["current_room_id"] = room.room_id
            # DRL-006: Include placement info
            info["room_remaining_area"] = self.room_remaining_area.get(
                room.room_id,
                0.0,
            )
            info["pods_placed_in_room"] = len(
                self.room_pod_placements.get(room.room_id, []),
            )

        return info

    # ── Result extraction ──────────────────────────────────────────────────

    def get_results(self) -> dict[str, Any]:
        """Extract the final panelization and placement results.

        Call after the episode terminates to get the complete assignment
        maps and episode statistics.

        Returns:
            Dictionary with:
            - ``wall_assignments``: dict[edge_id, list[(sku, cut_length)]]
            - ``room_assignments``: dict[room_id, pod_sku]
            - ``walls_covered``: number of walls with panel assignments
            - ``rooms_covered``: number of rooms with pod assignments
            - ``total_walls``: total number of walls
            - ``total_rooms``: total number of rooms
            - ``wall_coverage_pct``: percentage of walls covered
            - ``room_coverage_pct``: percentage of rooms covered
            - ``total_reward``: cumulative episode reward
            - ``total_violations``: count of violations across all steps
            - ``step_rewards``: list of RewardBreakdown objects
        """
        total_reward = sum(r.total for r in self._episode_rewards)
        total_violations = sum(len(r.violations) for r in self._episode_rewards)

        # DRL-005: Count multi-panel walls (walls with >1 panel)
        multi_panel_walls = sum(
            1 for assignments in self.wall_assignments.values() if len(assignments) > 1
        )

        # DRL-006: Count multi-pod rooms
        multi_pod_rooms = sum(
            1 for placements in self.room_pod_placements.values() if len(placements) > 1
        )

        # DRL-008: Count junction violations
        junction_violations = sum(
            1 for r in self._episode_rewards if r.info.get("junction_penalty", 0.0) < 0.0
        )

        return {
            "wall_assignments": dict(self.wall_assignments),
            "room_assignments": dict(self.room_assignments),
            "walls_covered": len(self.wall_assignments),
            "rooms_covered": len(self.room_assignments),
            "total_walls": self._n_walls,
            "total_rooms": self._n_rooms,
            "wall_coverage_pct": (
                100.0 * len(self.wall_assignments) / self._n_walls if self._n_walls > 0 else 0.0
            ),
            "room_coverage_pct": (
                100.0 * len(self.room_assignments) / self._n_rooms if self._n_rooms > 0 else 0.0
            ),
            "total_reward": total_reward,
            "total_violations": total_violations,
            "step_rewards": list(self._episode_rewards),
            # DRL-005: Multi-panel info
            "multi_panel_walls": multi_panel_walls,
            "wall_remaining_inches": dict(self.wall_remaining_inches),
            # DRL-006: Multi-pod info
            "multi_pod_rooms": multi_pod_rooms,
            "room_pod_placements": dict(self.room_pod_placements),
            # DRL-008: Junction info
            "junction_violations": junction_violations,
        }
