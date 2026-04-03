"""Reward function for the panelization/placement DRL environment.

DRL-004: Computes reward after each action based on four components:

1. **SPUR (Standard Panel Utilization Rate)** — positive reward for using
   standard-length panels that minimize cutting and waste.

2. **Waste penalty** — negative reward proportional to material waste
   (cut-offs, over-length panels).

3. **Violation penalty** — strong negative reward for structural gaps,
   spatial overlaps, opening obstructions, type mismatches, or fabrication
   constraint failures.

4. **Coverage bonus** — positive reward for completing wall/room coverage
   and maintaining high catalog match rate.

The overall reward is a weighted sum:

    R = w_spur * R_spur + w_waste * R_waste + w_violation * R_violation
        + w_coverage * R_coverage

At episode end, a terminal bonus is added:

    R_terminal = w_wall_completion * wall_coverage + w_room_completion * room_coverage

with configurable weights.  Default weights are tuned so that:
- A perfect standard-panel assignment yields R ~ +1.0
- A wall skip yields R ~ -0.5 (penalized — agent must learn to panelize)
- A violation yields R ~ -0.5 to -1.5 (depending on severity)
- Full wall+room coverage yields terminal bonus of +5.0
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from src.drl.actions import PanelAction, PlacementAction
from src.knowledge_graph.query import validate_wall_panelization
from src.knowledge_graph.schema import PanelType

if TYPE_CHECKING:
    from src.knowledge_graph.loader import KnowledgeGraphStore

logger = logging.getLogger(__name__)


# ── Reward weights (defaults) ──────────────────────────────────────────────


@dataclass
class RewardWeights:
    """Configurable weights for each reward component.

    Attributes:
        spur: Weight for SPUR (standard panel utilization). Default 1.0.
        waste: Weight for waste penalty (negative). Default 0.5.
        violation: Weight for violation penalty (negative). Default 1.5.
        coverage: Weight for coverage bonus. Default 1.0.
        completion_bonus: One-time bonus at episode end for full coverage.
            Default 5.0.
        wall_skip_penalty: Penalty applied when the agent skips a wall
            instead of panelizing it. Default 0.5.
        wall_completion_bonus: Terminal bonus scaled by wall coverage.
            Default 3.0.
        room_completion_bonus: Terminal bonus scaled by room coverage.
            Default 2.0.
    """

    spur: float = 1.0
    waste: float = 0.5
    violation: float = 1.5
    coverage: float = 1.0
    completion_bonus: float = 5.0
    wall_skip_penalty: float = 0.5
    wall_completion_bonus: float = 3.0
    room_completion_bonus: float = 2.0


# ── Reward breakdown ───────────────────────────────────────────────────────


@dataclass
class RewardBreakdown:
    """Detailed breakdown of a single-step reward.

    Returned by ``compute_reward`` for logging and debugging.
    """

    total: float = 0.0

    spur: float = 0.0
    """SPUR component: higher when using standard-length panels."""

    waste: float = 0.0
    """Waste penalty component (non-positive)."""

    violation: float = 0.0
    """Violation penalty component (non-positive)."""

    coverage: float = 0.0
    """Coverage bonus component (non-negative)."""

    violations: list[str] = field(default_factory=list)
    """Human-readable list of violations detected."""

    info: dict[str, float] = field(default_factory=dict)
    """Additional metrics for logging."""


# ── SPUR computation ───────────────────────────────────────────────────────

# Standard panel lengths in inches (common CFS industry lengths)
_STANDARD_LENGTHS: list[float] = [
    48.0,    # 4 ft
    72.0,    # 6 ft
    96.0,    # 8 ft
    120.0,   # 10 ft
    144.0,   # 12 ft
    192.0,   # 16 ft
    240.0,   # 20 ft
]

_STANDARD_TOLERANCE: float = 0.5  # 1/2 inch tolerance for "standard" match


def _compute_spur(panel_action: PanelAction) -> float:
    """Compute SPUR (Standard Panel Utilization Rate) for a panel action.

    SPUR measures how well the chosen panels align with standard industry
    lengths. A panel cut to exactly a standard length gets SPUR = 1.0.
    Custom cuts reduce SPUR proportionally.

    Returns:
        SPUR score in [0.0, 1.0].  0.0 if action is SKIP.
    """
    if panel_action.skip or panel_action.recommendation is None:
        return 0.0

    rec = panel_action.recommendation
    if not rec.cut_lengths_inches:
        return 0.0

    total_spur = 0.0
    for cut_length in rec.cut_lengths_inches:
        # Check how close this cut is to any standard length
        best_match_dist = min(
            abs(cut_length - std) for std in _STANDARD_LENGTHS
        )
        if best_match_dist <= _STANDARD_TOLERANCE:
            # Exact standard match
            total_spur += 1.0
        else:
            # Partial credit: decay based on distance from nearest standard
            # A panel 6 inches off standard gets ~0.5, 12 inches off gets ~0.25
            total_spur += max(0.0, 1.0 - best_match_dist / 24.0)

    # Average across all pieces
    return total_spur / len(rec.cut_lengths_inches)


# ── Waste computation ──────────────────────────────────────────────────────


def _compute_waste_penalty(panel_action: PanelAction) -> float:
    """Compute waste penalty for a panel action.

    Waste is the material that is cut off and discarded. Expressed as
    a fraction of total material consumed.

    Returns:
        Penalty in [-1.0, 0.0].  0.0 if no waste or SKIP.
    """
    if panel_action.skip or panel_action.recommendation is None:
        return 0.0

    rec = panel_action.recommendation
    total_material = sum(rec.cut_lengths_inches) if rec.cut_lengths_inches else 0.0
    if total_material <= 0.0:
        return 0.0

    waste_fraction = rec.waste_inches / total_material
    # Square the waste fraction to penalize high waste more severely
    return -min(waste_fraction ** 0.5, 1.0)


# ── Violation detection ────────────────────────────────────────────────────


def _detect_panel_violations(
    panel_action: PanelAction,
    wall_length_inches: float,
    wall_panel_type: PanelType,
    store: KnowledgeGraphStore,
) -> list[str]:
    """Detect violations in a panelization action.

    Checks:
    - Panel type matches wall type
    - Total panel coverage matches wall length (no gaps/overlaps)
    - Fabrication constraints are satisfied (via KG validation)
    - No single piece exceeds fabrication machine limits

    Returns:
        List of violation description strings. Empty if no violations.
    """
    if panel_action.skip:
        return []

    violations: list[str] = []
    panel = panel_action.panel
    rec = panel_action.recommendation

    if panel is None or rec is None:
        violations.append("Panel action has no panel or recommendation")
        return violations

    # 1. Type mismatch
    if panel.panel_type != wall_panel_type:
        violations.append(
            f"Panel type '{panel.panel_type}' does not match required "
            f"wall type '{wall_panel_type}'"
        )

    # 2. Coverage check: total cut lengths vs wall length
    total_coverage = sum(rec.cut_lengths_inches) if rec.cut_lengths_inches else 0.0
    coverage_diff = total_coverage - wall_length_inches
    if abs(coverage_diff) > 0.5:  # 1/2 inch tolerance
        if coverage_diff < 0:
            violations.append(
                f"Gap: panels cover {total_coverage:.1f}\" but wall is "
                f"{wall_length_inches:.1f}\" (gap of {-coverage_diff:.1f}\")"
            )
        else:
            violations.append(
                f"Overlap: panels cover {total_coverage:.1f}\" but wall is "
                f"{wall_length_inches:.1f}\" (overlap of {coverage_diff:.1f}\")"
            )

    # 3. KG fabrication validation
    if panel_action.panel_assignments:
        validation = validate_wall_panelization(
            store,
            wall_length_inches=wall_length_inches,
            wall_type=wall_panel_type,
            panel_assignments=panel_action.panel_assignments,
        )
        if not validation.is_valid:
            violations.extend(
                f"KG validation: {err}" for err in validation.errors
            )

    return violations


def _detect_pod_violations(
    placement_action: PlacementAction,
    room_width_inches: float,
    room_depth_inches: float,
) -> list[str]:
    """Detect violations in a pod placement action.

    Checks:
    - Pod physically fits in the room (with clearances)
    - Pod does not exceed room dimensions in the chosen orientation

    Returns:
        List of violation description strings.
    """
    if placement_action.skip:
        return []

    violations: list[str] = []
    pod = placement_action.pod
    if pod is None:
        violations.append("Placement action has no pod")
        return violations

    if placement_action.rotated:
        pod_w = pod.depth_inches + 2 * pod.clearance_inches
        pod_d = pod.width_inches + 2 * pod.clearance_inches
    else:
        pod_w = pod.width_inches + 2 * pod.clearance_inches
        pod_d = pod.depth_inches + 2 * pod.clearance_inches

    if pod_w > room_width_inches:
        violations.append(
            f"Pod width ({pod_w:.1f}\" with clearance) exceeds "
            f"room width ({room_width_inches:.1f}\")"
        )
    if pod_d > room_depth_inches:
        violations.append(
            f"Pod depth ({pod_d:.1f}\" with clearance) exceeds "
            f"room depth ({room_depth_inches:.1f}\")"
        )

    return violations


# ── Coverage computation ───────────────────────────────────────────────────


def _compute_coverage_bonus(
    walls_assigned: int,
    total_walls: int,
    rooms_assigned: int,
    total_rooms: int,
    is_terminal: bool,
) -> float:
    """Compute incremental coverage bonus.

    Awards a small positive reward for each new assignment, and a
    large bonus at episode end if all walls and rooms are covered.

    Returns:
        Coverage bonus in [0.0, 1.0] per step,
        plus completion bonus at terminal state.
    """
    if total_walls == 0 and total_rooms == 0:
        return 0.0

    wall_coverage = walls_assigned / max(total_walls, 1)
    room_coverage = rooms_assigned / max(total_rooms, 1)

    # Incremental: small reward per assignment
    incremental = (wall_coverage + room_coverage) / 2.0

    return incremental


# ── Main reward function ───────────────────────────────────────────────────


def compute_reward(
    panel_action: PanelAction | None,
    placement_action: PlacementAction | None,
    wall_length_inches: float,
    wall_panel_type: PanelType | None,
    room_width_inches: float,
    room_depth_inches: float,
    store: KnowledgeGraphStore,
    walls_assigned: int,
    total_walls: int,
    rooms_assigned: int,
    total_rooms: int,
    is_terminal: bool,
    weights: RewardWeights | None = None,
) -> RewardBreakdown:
    """Compute the total reward for a single step.

    Called after each ``env.step()`` to evaluate the action taken.
    Exactly one of ``panel_action`` or ``placement_action`` should be
    non-None (depending on whether the current phase is panelization
    or placement).

    Args:
        panel_action: Decoded panel action (panelization phase), or None.
        placement_action: Decoded placement action (placement phase), or None.
        wall_length_inches: Length of the current wall in inches (0 if placement).
        wall_panel_type: Expected PanelType for the wall (None if placement).
        room_width_inches: Room width in inches (0 if panelization).
        room_depth_inches: Room depth in inches (0 if panelization).
        store: Knowledge Graph store for validation queries.
        walls_assigned: Number of walls assigned so far (including current).
        total_walls: Total number of walls in the floor plan.
        rooms_assigned: Number of rooms assigned so far (including current).
        total_rooms: Total number of rooms in the floor plan.
        is_terminal: Whether this is the last step in the episode.
        weights: Reward component weights. Uses defaults if None.

    Returns:
        RewardBreakdown with total reward and per-component scores.
    """
    if weights is None:
        weights = RewardWeights()

    breakdown = RewardBreakdown()

    # ── Panelization reward ──
    if panel_action is not None:
        if panel_action.skip:
            # Penalize skipping walls — the agent must learn to panelize
            breakdown.violation = -weights.wall_skip_penalty
            breakdown.info["wall_skipped"] = 1.0
        else:
            # SPUR
            breakdown.spur = _compute_spur(panel_action)

            # Waste
            breakdown.waste = _compute_waste_penalty(panel_action)

            # Violations
            if wall_panel_type is not None:
                violations = _detect_panel_violations(
                    panel_action, wall_length_inches, wall_panel_type, store,
                )
                breakdown.violations = violations
                if violations:
                    # Scale violation penalty by number of violations
                    breakdown.violation = -min(float(len(violations)), 3.0) / 3.0

            # Info metrics
            if panel_action.recommendation is not None:
                breakdown.info["waste_percentage"] = panel_action.recommendation.waste_percentage
                breakdown.info["num_panels"] = float(panel_action.recommendation.quantity)
                breakdown.info["requires_splice"] = float(panel_action.recommendation.requires_splice)

    # ── Placement reward ──
    if placement_action is not None:
        if not placement_action.skip and placement_action.pod is not None:
            # SPUR analog for placement: space utilization
            pod = placement_action.pod
            if placement_action.rotated:
                pod_area = pod.depth_inches * pod.width_inches
            else:
                pod_area = pod.width_inches * pod.depth_inches
            room_area = room_width_inches * room_depth_inches
            if room_area > 0:
                utilization = pod_area / room_area
                breakdown.spur = min(utilization, 1.0)
            else:
                breakdown.spur = 0.0

            # Waste analog: unused room space
            breakdown.waste = -(1.0 - breakdown.spur) * 0.3

            # Violations
            violations = _detect_pod_violations(
                placement_action, room_width_inches, room_depth_inches,
            )
            breakdown.violations = violations
            if violations:
                breakdown.violation = -min(float(len(violations)), 3.0) / 3.0

            breakdown.info["pod_utilization"] = breakdown.spur
        else:
            # Skip: neutral
            breakdown.spur = 0.0
            breakdown.waste = 0.0

    # ── Coverage ──
    breakdown.coverage = _compute_coverage_bonus(
        walls_assigned, total_walls, rooms_assigned, total_rooms, is_terminal,
    )

    # ── Weighted total ──
    breakdown.total = (
        weights.spur * breakdown.spur
        + weights.waste * breakdown.waste
        + weights.violation * breakdown.violation
        + weights.coverage * breakdown.coverage
    )

    # ── Terminal completion bonus ──
    if is_terminal:
        wall_coverage = walls_assigned / max(total_walls, 1)
        room_coverage = rooms_assigned / max(total_rooms, 1)

        # Separate bonuses so skipping walls can't be subsidized by rooms
        wall_bonus = weights.wall_completion_bonus * wall_coverage
        room_bonus = weights.room_completion_bonus * room_coverage
        completion = wall_bonus + room_bonus
        breakdown.total += completion
        breakdown.info["completion_bonus"] = completion
        breakdown.info["wall_completion_bonus"] = wall_bonus
        breakdown.info["room_completion_bonus"] = room_bonus
        breakdown.info["wall_coverage"] = wall_coverage
        breakdown.info["room_coverage"] = room_coverage

    # ── Log violations at WARNING level ──
    if breakdown.violations:
        logger.warning(
            "DRL step violations: %s",
            "; ".join(breakdown.violations),
        )

    return breakdown
