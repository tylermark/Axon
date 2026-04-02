"""FS-003: Design modification suggestions.

Analyzes identified blockers and the panelization result to produce
actionable design change recommendations that would increase prefab
coverage. Each suggestion is tied to one or more blockers and includes
estimated coverage gains.
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING

from docs.interfaces.feasibility_report import (
    Blocker,
    BlockerCategory,
    DesignSuggestion,
    SuggestionType,
)
from src.knowledge_graph.query import get_fabrication_limits, get_valid_pods

if TYPE_CHECKING:
    from docs.interfaces.drl_output import PanelizationResult
    from src.knowledge_graph.loader import KnowledgeGraphStore

logger = logging.getLogger(__name__)


def _estimate_wall_coverage_gain(
    wall_length_inches: float,
    total_wall_length: float,
) -> float:
    """Estimate coverage gain from converting one wall to panelizable.

    Args:
        wall_length_inches: Length of the wall that would become panelizable.
        total_wall_length: Total wall length across all walls.

    Returns:
        Estimated coverage gain in percentage points.
    """
    if total_wall_length <= 0:
        return 0.0
    return round(wall_length_inches / total_wall_length * 100.0, 2)


def _estimate_panels_for_wall(wall_length_inches: float, max_panel_length: float) -> int:
    """Estimate number of panels needed for a wall.

    Args:
        wall_length_inches: Wall length in inches.
        max_panel_length: Maximum single-panel length from the KG.

    Returns:
        Estimated panel count (at least 1).
    """
    if max_panel_length <= 0:
        return 1
    return max(1, math.ceil(wall_length_inches / max_panel_length))


def generate_suggestions(
    blockers: list[Blocker],
    result: PanelizationResult,
    store: KnowledgeGraphStore,
) -> list[DesignSuggestion]:
    """Generate design modification suggestions from identified blockers.

    Produces targeted suggestions for each blocker category:

    - **MACHINE_LIMITS** (wall too long): ``WALL_SHORTEN`` with target
      length equal to the machine's max fabrication length.
    - **MACHINE_LIMITS** (wall too short): ``WALL_EXTEND`` to minimum
      panel length.
    - **GEOMETRY** (non-orthogonal): ``WALL_STRAIGHTEN`` to enable
      standard panel usage.
    - **CLEARANCE** (room too small for pod): ``ROOM_RESIZE`` with
      specific dimension recommendations from pod min requirements.
    - **PRODUCT_GAP** (no matching product): ``WALL_RECLASSIFY`` or
      ``ROOM_REFUNCTION`` to match available catalog entries.
    - **CODE_CONSTRAINT**: ``WALL_RECLASSIFY`` to access panels with
      the required rating.
    - **OPENING_CONFLICT**: ``OPENING_RELOCATE`` to avoid splice
      conflicts.

    Args:
        blockers: Identified blockers from ``identify_blockers``.
        result: The ``PanelizationResult`` from the DRL Agent.
        store: KG store for product and limit lookups.

    Returns:
        List of ``DesignSuggestion`` instances sorted by estimated
        coverage gain (highest first).
    """
    suggestions: list[DesignSuggestion] = []
    suggestion_counter = 0

    # Pre-compute totals for gain estimation.
    total_wall_length = sum(wp.wall_length_inches for wp in result.panel_map.walls)
    fab_limits = get_fabrication_limits(store)
    max_fab_length = fab_limits.get("max_length_inches", 0.0)

    # Build wall lookup for cross-referencing.
    walls_by_edge: dict[int, object] = {}
    for wp in result.panel_map.walls:
        walls_by_edge[wp.edge_id] = wp

    # Build room lookup.
    rooms_by_id: dict[int, object] = {}
    for rp in result.placement_map.rooms:
        rooms_by_id[rp.room_id] = rp

    # Also index source rooms for dimension data.
    source_rooms_by_id: dict[int, object] = {}
    for room in result.source_graph.graph.rooms:
        source_rooms_by_id[room.room_id] = room

    for blocker in blockers:
        if blocker.category == BlockerCategory.MACHINE_LIMITS:
            suggestion_counter = _suggest_machine_limits(
                blocker,
                walls_by_edge,
                total_wall_length,
                max_fab_length,
                fab_limits,
                suggestion_counter,
                suggestions,
            )

        elif blocker.category == BlockerCategory.GEOMETRY:
            suggestion_counter = _suggest_geometry(
                blocker,
                walls_by_edge,
                total_wall_length,
                max_fab_length,
                suggestion_counter,
                suggestions,
            )

        elif blocker.category == BlockerCategory.CLEARANCE:
            suggestion_counter = _suggest_clearance(
                blocker,
                rooms_by_id,
                source_rooms_by_id,
                store,
                suggestion_counter,
                suggestions,
            )

        elif blocker.category == BlockerCategory.PRODUCT_GAP:
            suggestion_counter = _suggest_product_gap(
                blocker,
                walls_by_edge,
                rooms_by_id,
                total_wall_length,
                max_fab_length,
                suggestion_counter,
                suggestions,
            )

        elif blocker.category == BlockerCategory.CODE_CONSTRAINT:
            suggestion_counter = _suggest_code_constraint(
                blocker,
                walls_by_edge,
                total_wall_length,
                max_fab_length,
                suggestion_counter,
                suggestions,
            )

        elif blocker.category == BlockerCategory.OPENING_CONFLICT:
            suggestion_counter = _suggest_opening_conflict(
                blocker,
                suggestion_counter,
                suggestions,
            )

    # Sort by estimated coverage gain descending.
    suggestions.sort(key=lambda s: -s.estimated_coverage_gain_pct)
    return suggestions


# ── Per-category suggestion generators ──────────────────────────────────


def _suggest_machine_limits(
    blocker: Blocker,
    walls_by_edge: dict[int, object],
    total_wall_length: float,
    max_fab_length: float,
    fab_limits: dict[str, float],
    counter: int,
    suggestions: list[DesignSuggestion],
) -> int:
    """Generate suggestions for MACHINE_LIMITS blockers."""
    desc_lower = blocker.description.lower()

    for edge_id in blocker.affected_edge_ids:
        wp = walls_by_edge.get(edge_id)
        if wp is None:
            continue

        if "too short" in desc_lower or "below minimum" in desc_lower:
            # Wall too short: suggest extending.
            counter += 1
            suggestions.append(
                DesignSuggestion(
                    suggestion_id=f"SUG-{counter:03d}",
                    suggestion_type=SuggestionType.WALL_EXTEND,
                    description=(
                        f"Extend wall #{edge_id} to meet minimum panel length. "
                        f"Current length: {wp.wall_length_inches:.1f} in."
                    ),
                    resolves_blocker_ids=[blocker.blocker_id],
                    affected_edge_ids=[edge_id],
                    estimated_coverage_gain_pct=_estimate_wall_coverage_gain(
                        wp.wall_length_inches, total_wall_length
                    ),
                    estimated_panels_gained=1,
                    effort_level="low",
                )
            )
        else:
            # Wall too long: suggest shortening to max fab length.
            counter += 1
            target_length = max_fab_length if max_fab_length > 0 else wp.wall_length_inches
            overshoot = wp.wall_length_inches - target_length
            suggestions.append(
                DesignSuggestion(
                    suggestion_id=f"SUG-{counter:03d}",
                    suggestion_type=SuggestionType.WALL_SHORTEN,
                    description=(
                        f"Shorten wall #{edge_id} from "
                        f"{wp.wall_length_inches:.1f} in to "
                        f"{target_length:.1f} in (reduce by "
                        f"{overshoot:.1f} in) to fit within machine "
                        f"fabrication limits."
                    ),
                    resolves_blocker_ids=[blocker.blocker_id],
                    affected_edge_ids=[edge_id],
                    estimated_coverage_gain_pct=_estimate_wall_coverage_gain(
                        wp.wall_length_inches, total_wall_length
                    ),
                    estimated_panels_gained=_estimate_panels_for_wall(
                        target_length, max_fab_length
                    ),
                    effort_level="medium",
                )
            )

    return counter


def _suggest_geometry(
    blocker: Blocker,
    walls_by_edge: dict[int, object],
    total_wall_length: float,
    max_fab_length: float,
    counter: int,
    suggestions: list[DesignSuggestion],
) -> int:
    """Generate suggestions for GEOMETRY blockers."""
    for edge_id in blocker.affected_edge_ids:
        wp = walls_by_edge.get(edge_id)
        if wp is None:
            continue

        counter += 1
        panels_gained = _estimate_panels_for_wall(wp.wall_length_inches, max_fab_length)
        suggestions.append(
            DesignSuggestion(
                suggestion_id=f"SUG-{counter:03d}",
                suggestion_type=SuggestionType.WALL_STRAIGHTEN,
                description=(
                    f"Straighten wall #{edge_id} "
                    f"({wp.wall_length_inches:.1f} in) to enable "
                    f"standard panel coverage, adding approximately "
                    f"{panels_gained} panel(s)."
                ),
                resolves_blocker_ids=[blocker.blocker_id],
                affected_edge_ids=[edge_id],
                estimated_coverage_gain_pct=_estimate_wall_coverage_gain(
                    wp.wall_length_inches, total_wall_length
                ),
                estimated_panels_gained=panels_gained,
                effort_level="high",
            )
        )

    return counter


def _suggest_clearance(
    blocker: Blocker,
    rooms_by_id: dict[int, object],
    source_rooms_by_id: dict[int, object],
    store: KnowledgeGraphStore,
    counter: int,
    suggestions: list[DesignSuggestion],
) -> int:
    """Generate suggestions for CLEARANCE blockers."""
    for room_id in blocker.affected_room_ids:
        rp = rooms_by_id.get(room_id)
        if rp is None:
            continue

        # Try to find a pod that would fit with a resize.
        room_label = rp.room_label if hasattr(rp, "room_label") else ""
        pods = get_valid_pods(
            store,
            room_width_inches=9999.0,  # unrestricted search
            room_depth_inches=9999.0,
            room_function=room_label.lower() if room_label else None,
        )

        if pods:
            smallest_pod = min(
                pods,
                key=lambda p: p.min_room_width_inches * p.min_room_depth_inches,
            )
            counter += 1
            suggestions.append(
                DesignSuggestion(
                    suggestion_id=f"SUG-{counter:03d}",
                    suggestion_type=SuggestionType.ROOM_RESIZE,
                    description=(
                        f"Resize room #{room_id} "
                        f"({room_label or 'unlabeled'}) to at least "
                        f"{smallest_pod.min_room_width_inches:.0f} in x "
                        f"{smallest_pod.min_room_depth_inches:.0f} in "
                        f"to accommodate {smallest_pod.name} "
                        f"({smallest_pod.sku})."
                    ),
                    resolves_blocker_ids=[blocker.blocker_id],
                    affected_room_ids=[room_id],
                    estimated_pods_gained=1,
                    effort_level="medium",
                )
            )
        else:
            # No pods available for this room type: suggest refunction.
            counter += 1
            suggestions.append(
                DesignSuggestion(
                    suggestion_id=f"SUG-{counter:03d}",
                    suggestion_type=SuggestionType.ROOM_REFUNCTION,
                    description=(
                        f"Consider changing room #{room_id} "
                        f"({room_label or 'unlabeled'}) function to "
                        f"match an available pod type in the catalog."
                    ),
                    resolves_blocker_ids=[blocker.blocker_id],
                    affected_room_ids=[room_id],
                    estimated_pods_gained=1,
                    effort_level="medium",
                )
            )

    return counter


def _suggest_product_gap(
    blocker: Blocker,
    walls_by_edge: dict[int, object],
    rooms_by_id: dict[int, object],
    total_wall_length: float,
    max_fab_length: float,
    counter: int,
    suggestions: list[DesignSuggestion],
) -> int:
    """Generate suggestions for PRODUCT_GAP blockers."""
    # Wall-related product gaps: suggest reclassifying wall type.
    for edge_id in blocker.affected_edge_ids:
        wp = walls_by_edge.get(edge_id)
        if wp is None:
            continue

        counter += 1
        suggestions.append(
            DesignSuggestion(
                suggestion_id=f"SUG-{counter:03d}",
                suggestion_type=SuggestionType.WALL_RECLASSIFY,
                description=(
                    f"Reclassify wall #{edge_id} "
                    f"({wp.wall_length_inches:.1f} in) to a type with "
                    f"available panel products in the catalog "
                    f"(e.g., partition or load-bearing)."
                ),
                resolves_blocker_ids=[blocker.blocker_id],
                affected_edge_ids=[edge_id],
                estimated_coverage_gain_pct=_estimate_wall_coverage_gain(
                    wp.wall_length_inches, total_wall_length
                ),
                estimated_panels_gained=_estimate_panels_for_wall(
                    wp.wall_length_inches, max_fab_length
                ),
                effort_level="low",
            )
        )

    # Room-related product gaps: suggest refunctioning.
    for room_id in blocker.affected_room_ids:
        rp = rooms_by_id.get(room_id)
        if rp is None:
            continue

        counter += 1
        room_label = rp.room_label if hasattr(rp, "room_label") else ""
        suggestions.append(
            DesignSuggestion(
                suggestion_id=f"SUG-{counter:03d}",
                suggestion_type=SuggestionType.ROOM_REFUNCTION,
                description=(
                    f"Change function of room #{room_id} "
                    f"({room_label or 'unlabeled'}) to match an "
                    f"available pod type (e.g., bathroom, kitchen)."
                ),
                resolves_blocker_ids=[blocker.blocker_id],
                affected_room_ids=[room_id],
                estimated_pods_gained=1,
                effort_level="low",
            )
        )

    return counter


def _suggest_code_constraint(
    blocker: Blocker,
    walls_by_edge: dict[int, object],
    total_wall_length: float,
    max_fab_length: float,
    counter: int,
    suggestions: list[DesignSuggestion],
) -> int:
    """Generate suggestions for CODE_CONSTRAINT blockers."""
    for edge_id in blocker.affected_edge_ids:
        wp = walls_by_edge.get(edge_id)
        if wp is None:
            continue

        counter += 1
        suggestions.append(
            DesignSuggestion(
                suggestion_id=f"SUG-{counter:03d}",
                suggestion_type=SuggestionType.WALL_RECLASSIFY,
                description=(
                    f"Reclassify wall #{edge_id} to a type that can "
                    f"achieve the required fire rating or code "
                    f"compliance with available panel products."
                ),
                resolves_blocker_ids=[blocker.blocker_id],
                affected_edge_ids=[edge_id],
                estimated_coverage_gain_pct=_estimate_wall_coverage_gain(
                    wp.wall_length_inches, total_wall_length
                ),
                estimated_panels_gained=_estimate_panels_for_wall(
                    wp.wall_length_inches, max_fab_length
                ),
                effort_level="high",
            )
        )

    return counter


def _suggest_opening_conflict(
    blocker: Blocker,
    counter: int,
    suggestions: list[DesignSuggestion],
) -> int:
    """Generate suggestions for OPENING_CONFLICT blockers."""
    for edge_id in blocker.affected_edge_ids:
        counter += 1
        suggestions.append(
            DesignSuggestion(
                suggestion_id=f"SUG-{counter:03d}",
                suggestion_type=SuggestionType.OPENING_RELOCATE,
                description=(
                    f"Relocate opening on wall #{edge_id} to avoid "
                    f"panel splice-point conflicts. Move the opening "
                    f"at least 6 inches from any panel joint."
                ),
                resolves_blocker_ids=[blocker.blocker_id],
                affected_edge_ids=[edge_id],
                estimated_coverage_gain_pct=0.0,  # Opening fixes don't change length coverage.
                estimated_panels_gained=0,
                effort_level="low",
            )
        )

    return counter
