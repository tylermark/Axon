"""FS-004: Feasibility report generation.

Orchestrates the calculator, blocker identification, and suggestion
generation into a single ``FeasibilityReport`` output that answers:

    "How much of this floor plan can Capsule Manufacturing prefabricate,
     what's blocking the rest, and what design changes would improve
     coverage?"

This module is the public entry point for the Feasibility Agent.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from docs.interfaces.feasibility_report import (
    FeasibilityReport,
    FeasibilitySummary,
    FloorScore,
    RoomFeasibility,
    WallFeasibility,
)
from src.feasibility.blockers import identify_blockers
from src.feasibility.calculator import calculate_coverage
from src.feasibility.suggestions import generate_suggestions

if TYPE_CHECKING:
    from docs.interfaces.drl_output import PanelizationResult
    from src.knowledge_graph.loader import KnowledgeGraphStore

logger = logging.getLogger(__name__)

# ── Scoring weights ─────────────────────────────────────────────────────
# Composite feasibility score = w_wall * wall_coverage + w_pod * pod_rate
#                                - w_blocker * normalized_blocker_severity
_W_WALL: float = 0.55
_W_POD: float = 0.25
_W_BLOCKER: float = 0.20


def _build_wall_feasibility(
    result: PanelizationResult,
    blocker_edge_map: dict[int, list[str]],
) -> list[WallFeasibility]:
    """Construct per-wall feasibility entries.

    Args:
        result: DRL panelization result.
        blocker_edge_map: Mapping from edge_id to list of blocker IDs.

    Returns:
        Ordered list of ``WallFeasibility``, one per wall segment.
    """
    entries: list[WallFeasibility] = []
    for wp in result.panel_map.walls:
        panelized_len = wp.wall_length_inches if wp.is_panelizable else 0.0
        coverage = (
            (panelized_len / wp.wall_length_inches * 100.0) if wp.wall_length_inches > 0 else 0.0
        )
        entries.append(
            WallFeasibility(
                edge_id=wp.edge_id,
                is_panelizable=wp.is_panelizable,
                wall_length_inches=round(wp.wall_length_inches, 2),
                panelized_length_inches=round(panelized_len, 2),
                coverage_pct=round(coverage, 2),
                blocker_ids=blocker_edge_map.get(wp.edge_id, []),
                rejection_reason=wp.rejection_reason if not wp.is_panelizable else "",
            )
        )
    return entries


def _build_room_feasibility(
    result: PanelizationResult,
    blocker_room_map: dict[int, list[str]],
) -> list[RoomFeasibility]:
    """Construct per-room feasibility entries.

    Args:
        result: DRL panelization result.
        blocker_room_map: Mapping from room_id to list of blocker IDs.

    Returns:
        Ordered list of ``RoomFeasibility``, one per room.
    """
    entries: list[RoomFeasibility] = []
    for rp in result.placement_map.rooms:
        has_pod = rp.placement is not None
        pod_sku = rp.placement.pod_sku if has_pod else ""
        entries.append(
            RoomFeasibility(
                room_id=rp.room_id,
                room_label=rp.room_label,
                room_area_sqft=round(rp.room_area_sqft, 2),
                is_eligible=rp.is_eligible,
                has_pod=has_pod,
                pod_sku=pod_sku,
                blocker_ids=blocker_room_map.get(rp.room_id, []),
                rejection_reason=rp.rejection_reason if not has_pod and rp.is_eligible else "",
            )
        )
    return entries


def _build_floor_score(
    wall_feas: list[WallFeasibility],
    room_feas: list[RoomFeasibility],
    blockers_count: int,
    coverage_pct: float,
    area_coverage_pct: float,
) -> FloorScore:
    """Compute a single FloorScore (single-floor assumption).

    Args:
        wall_feas: Per-wall feasibility list.
        room_feas: Per-room feasibility list.
        blockers_count: Total blocker count for this floor.
        coverage_pct: Wall length coverage percentage.
        area_coverage_pct: Wall area coverage percentage.

    Returns:
        A ``FloorScore`` instance.
    """
    total_walls = len(wall_feas)
    panelized_walls = sum(1 for w in wall_feas if w.is_panelizable)

    total_rooms = len(room_feas)
    eligible_rooms = sum(1 for r in room_feas if r.is_eligible)
    placed_rooms = sum(1 for r in room_feas if r.has_pod)

    pod_rate = (placed_rooms / eligible_rooms * 100.0) if eligible_rooms > 0 else 0.0

    # Composite score: blend wall coverage, pod rate, and blocker penalty.
    wall_factor = coverage_pct / 100.0
    pod_factor = pod_rate / 100.0
    # Normalize blocker severity: cap at total_walls + total_rooms.
    max_possible = max(total_walls + total_rooms, 1)
    blocker_factor = min(blockers_count / max_possible, 1.0)

    score = _W_WALL * wall_factor + _W_POD * pod_factor - _W_BLOCKER * blocker_factor
    score = max(0.0, min(score, 1.0))

    return FloorScore(
        floor_id="1",
        floor_label="Ground Floor",
        wall_coverage_pct=round(coverage_pct, 2),
        area_coverage_pct=round(area_coverage_pct, 2),
        pod_placement_rate_pct=round(pod_rate, 2),
        blocker_count=blockers_count,
        total_wall_count=total_walls,
        panelized_wall_count=panelized_walls,
        total_room_count=total_rooms,
        placed_room_count=placed_rooms,
        feasibility_score=round(score, 4),
    )


def _build_summary(
    wall_feas: list[WallFeasibility],
    room_feas: list[RoomFeasibility],
    blockers: list,
    suggestions: list,
    spur_score: float,
) -> FeasibilitySummary:
    """Build high-level summary statistics.

    Args:
        wall_feas: Per-wall feasibility list.
        room_feas: Per-room feasibility list.
        blockers: List of Blocker instances.
        suggestions: List of DesignSuggestion instances.
        spur_score: SPUR score from DRL output.

    Returns:
        A ``FeasibilitySummary`` instance.
    """
    total_walls = len(wall_feas)
    panelized_walls = sum(1 for w in wall_feas if w.is_panelizable)
    unpanelized_walls = total_walls - panelized_walls

    total_rooms = len(room_feas)
    eligible_rooms = sum(1 for r in room_feas if r.is_eligible)
    placed_rooms = sum(1 for r in room_feas if r.has_pod)

    hard_blockers = sum(1 for b in blockers if b.severity >= 1.0)
    soft_blockers = sum(1 for b in blockers if b.severity < 1.0)

    # Max coverage gain: sum all suggestion gains (optimistic upper bound).
    max_gain = sum(s.estimated_coverage_gain_pct for s in suggestions)

    return FeasibilitySummary(
        total_wall_count=total_walls,
        panelized_wall_count=panelized_walls,
        unpanelized_wall_count=unpanelized_walls,
        total_room_count=total_rooms,
        eligible_room_count=eligible_rooms,
        placed_room_count=placed_rooms,
        total_blocker_count=len(blockers),
        hard_blocker_count=hard_blockers,
        soft_blocker_count=soft_blockers,
        suggestion_count=len(suggestions),
        max_coverage_gain_pct=round(max_gain, 2),
        spur_score=round(spur_score, 4),
    )


def generate_feasibility_report(
    result: PanelizationResult,
    store: KnowledgeGraphStore,
) -> FeasibilityReport:
    """Generate a complete feasibility report from a panelization result.

    Orchestrates all four sub-tasks:

    1. **Coverage calculation** (FS-001) — prefab coverage by length,
       area, and cost ratio.
    2. **Blocker identification** (FS-002) — categorized issues preventing
       prefabrication.
    3. **Suggestion generation** (FS-003) — actionable design changes
       with estimated gains.
    4. **Report assembly** (FS-004) — per-wall/room breakdowns, floor
       scores, project score, and summary dashboard.

    Args:
        result: The ``PanelizationResult`` from the DRL Agent.
        store: KG store for product lookups, fabrication limits, etc.

    Returns:
        A fully populated ``FeasibilityReport``.
    """
    logger.info("Generating feasibility report...")

    # Step 1: Coverage metrics.
    coverage = calculate_coverage(result, store)
    logger.info(
        "Coverage: %.1f%% by length, %.1f%% by area, %.1f%% by cost",
        coverage.by_wall_length_pct,
        coverage.by_area_pct,
        coverage.by_cost_pct,
    )

    # Step 2: Blocker identification.
    blockers = identify_blockers(result, store)
    logger.info("Identified %d blockers", len(blockers))

    # Step 3: Design suggestions.
    suggestions = generate_suggestions(blockers, result, store)
    logger.info("Generated %d suggestions", len(suggestions))

    # Step 4: Assemble report.

    # Build blocker-to-element maps for cross-referencing.
    blocker_edge_map: dict[int, list[str]] = {}
    blocker_room_map: dict[int, list[str]] = {}
    for b in blockers:
        for eid in b.affected_edge_ids:
            blocker_edge_map.setdefault(eid, []).append(b.blocker_id)
        for rid in b.affected_room_ids:
            blocker_room_map.setdefault(rid, []).append(b.blocker_id)

    # Per-wall feasibility.
    wall_feas = _build_wall_feasibility(result, blocker_edge_map)

    # Per-room feasibility.
    room_feas = _build_room_feasibility(result, blocker_room_map)

    # Floor score (single floor for now).
    floor_score = _build_floor_score(
        wall_feas,
        room_feas,
        len(blockers),
        coverage.by_wall_length_pct,
        coverage.by_area_pct,
    )

    # Project score = floor score (single floor case).
    project_score = floor_score.feasibility_score

    # Summary dashboard stats.
    summary = _build_summary(
        wall_feas,
        room_feas,
        blockers,
        suggestions,
        result.spur_score,
    )

    report = FeasibilityReport(
        source=result,
        coverage=coverage,
        wall_feasibility=wall_feas,
        room_feasibility=room_feas,
        blockers=blockers,
        suggestions=suggestions,
        floor_scores=[floor_score],
        project_score=round(project_score, 4),
        summary=summary,
    )

    logger.info(
        "Feasibility report complete: project_score=%.4f, "
        "%d walls (%d panelized), %d rooms (%d with pods), "
        "%d blockers, %d suggestions",
        report.project_score,
        summary.total_wall_count,
        summary.panelized_wall_count,
        summary.total_room_count,
        summary.placed_room_count,
        summary.total_blocker_count,
        summary.suggestion_count,
    )

    return report
