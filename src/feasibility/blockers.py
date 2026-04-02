"""FS-002: Blocker identification and categorization.

Scans PanelizationResult for walls and rooms that could not be
prefabricated, classifies the root cause into a ``BlockerCategory``,
and assigns a severity score.

Hard blockers (severity 1.0) represent physically impossible cases.
Soft blockers (severity 0.3-0.8) represent efficiency losses or
cases that could be resolved with design modifications.
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING

from docs.interfaces.feasibility_report import Blocker, BlockerCategory
from src.knowledge_graph.query import get_fabrication_limits, get_machine_for_spec
from src.knowledge_graph.schema import PanelType

if TYPE_CHECKING:
    from docs.interfaces.drl_output import PanelizationResult
    from docs.interfaces.graph_to_serializer import WallType
    from src.knowledge_graph.loader import KnowledgeGraphStore

logger = logging.getLogger(__name__)

# Orthogonality tolerance: walls with angles outside multiples of 90 deg
# by more than this threshold (in radians) are considered non-orthogonal.
_ORTHO_TOLERANCE_RAD: float = math.radians(2.0)

# Keyword patterns in rejection reasons that map to categories.
_REASON_CATEGORY_MAP: list[tuple[list[str], BlockerCategory, float]] = [
    # (keyword list, category, default severity)
    (["length", "too long", "exceeds", "max length"], BlockerCategory.MACHINE_LIMITS, 1.0),
    (["too short", "min length", "below minimum"], BlockerCategory.MACHINE_LIMITS, 0.8),
    (["gauge", "depth", "web depth"], BlockerCategory.MACHINE_LIMITS, 0.9),
    (["angle", "non-orthogonal", "curved", "arc", "radius"], BlockerCategory.GEOMETRY, 0.7),
    (["fire", "rating", "fire-rated", "fire rating"], BlockerCategory.CODE_CONSTRAINT, 1.0),
    (["code", "compliance", "building code"], BlockerCategory.CODE_CONSTRAINT, 1.0),
    (["opening", "door", "window", "splice"], BlockerCategory.OPENING_CONFLICT, 0.6),
    (["clearance", "too small", "insufficient"], BlockerCategory.CLEARANCE, 0.8),
    (["no panel", "no product", "no compatible", "not found"], BlockerCategory.PRODUCT_GAP, 1.0),
]


def _classify_rejection(reason: str) -> tuple[BlockerCategory, float]:
    """Map a rejection reason string to a blocker category and severity.

    Args:
        reason: Free-text rejection reason from the DRL output.

    Returns:
        Tuple of ``(BlockerCategory, severity)``.
    """
    reason_lower = reason.lower()
    for keywords, category, severity in _REASON_CATEGORY_MAP:
        if any(kw in reason_lower for kw in keywords):
            return category, severity
    # Default: product gap is the safest catch-all for unrecognized reasons.
    return BlockerCategory.PRODUCT_GAP, 0.8


def _is_non_orthogonal(angle_rad: float) -> bool:
    """Check if a wall angle deviates from orthogonal alignment.

    Args:
        angle_rad: Wall angle in radians [0, pi).

    Returns:
        True if the wall is not aligned to 0, 90, or 180 degrees
        within tolerance.
    """
    for target in (0.0, math.pi / 2, math.pi):
        if abs(angle_rad - target) <= _ORTHO_TOLERANCE_RAD:
            return False
    return True


def _wall_type_to_panel_type(wall_type: WallType) -> PanelType:
    """Map graph WallType to KG PanelType for fabrication checks.

    Args:
        wall_type: The wall's structural classification.

    Returns:
        Corresponding KG panel type.
    """
    mapping: dict[str, PanelType] = {
        "load_bearing": PanelType.LOAD_BEARING,
        "partition": PanelType.PARTITION,
        "shear": PanelType.SHEAR,
        "exterior": PanelType.ENVELOPE,
        "curtain": PanelType.ENVELOPE,
        "fire_rated": PanelType.FIRE_RATED,
        "unknown": PanelType.PARTITION,
    }
    return mapping.get(wall_type.value, PanelType.PARTITION)


def identify_blockers(
    result: PanelizationResult,
    store: KnowledgeGraphStore,
) -> list[Blocker]:
    """Identify and categorize all blockers preventing prefabrication.

    Scans every wall and room in the panelization result, checks
    rejection reasons, validates against machine fabrication limits,
    and flags non-orthogonal geometry.

    Args:
        result: The ``PanelizationResult`` from the DRL Agent.
        store: KG store for fabrication limit lookups.

    Returns:
        List of ``Blocker`` instances sorted by severity (highest first).
    """
    blockers: list[Blocker] = []
    blocker_counter = 0

    # Pre-fetch fabrication limits.
    fab_limits = get_fabrication_limits(store)
    max_fab_length = fab_limits.get("max_length_inches", 0.0)

    # Index classifications by edge_id for fast lookup.
    classifications_by_edge: dict[int, object] = {}
    for cls in result.source_graph.classifications:
        classifications_by_edge[cls.edge_id] = cls

    # Index wall segments by edge_id.
    segments_by_edge: dict[int, object] = {}
    for seg in result.source_graph.graph.wall_segments:
        segments_by_edge[seg.edge_id] = seg

    # ── Wall blockers ───────────────────────────────────────────────────

    for wp in result.panel_map.walls:
        seg = segments_by_edge.get(wp.edge_id)
        cls = classifications_by_edge.get(wp.edge_id)

        # Check 1: Rejected wall from DRL output.
        if not wp.is_panelizable and wp.rejection_reason:
            blocker_counter += 1
            category, severity = _classify_rejection(wp.rejection_reason)
            blockers.append(
                Blocker(
                    blocker_id=f"BLK-{blocker_counter:03d}",
                    category=category,
                    description=(
                        f"Wall #{wp.edge_id} ({wp.wall_length_inches:.1f} in): "
                        f"{wp.rejection_reason}"
                    ),
                    affected_edge_ids=[wp.edge_id],
                    severity=severity,
                )
            )
            continue  # Already captured — skip duplicate checks.

        if not wp.is_panelizable:
            # No rejection reason given — check structural causes below.
            pass

        # Check 2: Machine limits — wall longer than any machine can produce.
        if max_fab_length > 0 and wp.wall_length_inches > max_fab_length:
            blocker_counter += 1
            overshoot = wp.wall_length_inches - max_fab_length
            blockers.append(
                Blocker(
                    blocker_id=f"BLK-{blocker_counter:03d}",
                    category=BlockerCategory.MACHINE_LIMITS,
                    description=(
                        f"Wall #{wp.edge_id} is {wp.wall_length_inches:.1f} in long, "
                        f"exceeding max fabrication length of {max_fab_length:.1f} in "
                        f"by {overshoot:.1f} in."
                    ),
                    affected_edge_ids=[wp.edge_id],
                    severity=1.0,
                )
            )

        # Check 3: Non-orthogonal geometry.
        if seg is not None and _is_non_orthogonal(seg.angle):
            blocker_counter += 1
            angle_deg = math.degrees(seg.angle)
            blockers.append(
                Blocker(
                    blocker_id=f"BLK-{blocker_counter:03d}",
                    category=BlockerCategory.GEOMETRY,
                    description=(
                        f"Wall #{wp.edge_id} is non-orthogonal "
                        f"({angle_deg:.1f} deg). Standard CFS panels "
                        f"require orthogonal alignment."
                    ),
                    affected_edge_ids=[wp.edge_id],
                    severity=0.7,
                )
            )

        # Check 4: No machines can handle the specific gauge + depth combo.
        if cls is not None and seg is not None and wp.is_panelizable and wp.panels:
            # Verify at least one machine can produce the assigned panels.
            first_sku = wp.panels[0].panel_sku
            panel = store.panels.get(first_sku)
            if panel is not None:
                machines = get_machine_for_spec(
                    store,
                    gauge=panel.gauge,
                    stud_depth_inches=panel.stud_depth_inches,
                    length_inches=wp.wall_length_inches,
                )
                if not machines:
                    blocker_counter += 1
                    blockers.append(
                        Blocker(
                            blocker_id=f"BLK-{blocker_counter:03d}",
                            category=BlockerCategory.MACHINE_LIMITS,
                            description=(
                                f"Wall #{wp.edge_id}: no machine can produce "
                                f'{panel.gauge}ga / {panel.stud_depth_inches}" depth '
                                f"panel at {wp.wall_length_inches:.1f} in length."
                            ),
                            affected_edge_ids=[wp.edge_id],
                            severity=0.9,
                        )
                    )

    # ── Room blockers ───────────────────────────────────────────────────

    for rp in result.placement_map.rooms:
        if rp.is_eligible and rp.placement is None:
            blocker_counter += 1
            if rp.rejection_reason:
                category, severity = _classify_rejection(rp.rejection_reason)
            else:
                # Default: clearance or product gap.
                category = BlockerCategory.CLEARANCE
                severity = 0.8

            blockers.append(
                Blocker(
                    blocker_id=f"BLK-{blocker_counter:03d}",
                    category=category,
                    description=(
                        f"Room #{rp.room_id} ({rp.room_label or 'unlabeled'}, "
                        f"{rp.room_area_sqft:.1f} sqft): "
                        f"{rp.rejection_reason or 'No compatible pod found.'}"
                    ),
                    affected_room_ids=[rp.room_id],
                    severity=severity,
                )
            )

    # Sort by severity descending (hardest blockers first).
    blockers.sort(key=lambda b: -b.severity)
    return blockers
