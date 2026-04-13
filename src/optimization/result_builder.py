"""Build PanelizationResult from raw wall/room assignment dicts.

Extracted and generalized from ``src/pipeline/full_pipeline._build_panelization_result``
to work with both the OR solver output and the DRL env output. Both paths
produce the same intermediate format:
- ``wall_assignments: dict[int, list[tuple[str, float]]]`` (edge_id -> [(sku, cut_len)])
- ``room_assignments: dict[int, str]`` (room_id -> pod_sku)

This builder converts those into the typed ``PanelizationResult`` dataclass
consumed by feasibility, BOM, and transplant agents.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from docs.interfaces.drl_output import (
    PanelAssignment,
    PanelizationResult,
    PanelMap,
    PlacementMap,
    ProductPlacement,
    RoomPlacement,
    WallPanelization,
)

if TYPE_CHECKING:
    from docs.interfaces.classified_wall_graph import ClassifiedWallGraph

logger = logging.getLogger(__name__)


def build_panelization_result(
    classified_graph: ClassifiedWallGraph,
    wall_assignments: dict[int, list[tuple[str, float]]],
    room_assignments: dict[int, str],
    room_orientations: dict[int, bool] | None = None,
    wall_splice_skus: dict[int, list[str]] | None = None,
    total_material_cost: float = 0.0,
    solver_name: str = "or_cpsat",
    solve_time_seconds: float = 0.0,
) -> PanelizationResult:
    """Build a PanelizationResult from raw assignment dictionaries.

    Args:
        classified_graph: The classified wall graph that was panelized.
        wall_assignments: Per-wall panel assignments: edge_id -> [(sku, cut_length_inches)].
        room_assignments: Per-room pod assignments: room_id -> pod_sku.
        room_orientations: Per-room pod orientations: room_id -> rotated. Optional.
        wall_splice_skus: Per-wall splice connection SKUs: edge_id -> [splice_sku, ...].
            Length should be len(panels) - 1 for each wall. Optional.
        total_material_cost: Total material cost in USD from solver. Optional.
        solver_name: Name of the solver that produced these assignments.
        solve_time_seconds: Total solve time for metadata.

    Returns:
        A fully populated PanelizationResult.
    """
    if room_orientations is None:
        room_orientations = {}
    if wall_splice_skus is None:
        wall_splice_skus = {}

    graph = classified_graph.graph
    scale = graph.scale_factor
    to_inches = scale / 25.4 if scale != 1.0 else 1.0 / 72.0

    # ── Build PanelMap ────────────────────────────────────────────────────
    wall_panelizations: list[WallPanelization] = []
    total_material_inches = 0.0
    all_panel_skus: set[str] = set()
    all_splice_skus: set[str] = set()
    total_panel_count = 0
    total_splice_count = 0

    for seg in graph.wall_segments:
        wall_length_inches = seg.length * to_inches
        assignments = wall_assignments.get(seg.edge_id, [])
        is_panelizable = bool(assignments)

        panel_list: list[PanelAssignment] = []
        position = 0.0
        for idx, (sku, cut_length) in enumerate(assignments):
            panel_list.append(
                PanelAssignment(
                    panel_sku=sku,
                    cut_length_inches=cut_length,
                    position_along_wall=position,
                    panel_index=idx,
                )
            )
            position += cut_length
            all_panel_skus.add(sku)

        mat_total = sum(pa.cut_length_inches for pa in panel_list)
        waste = max(0.0, mat_total - wall_length_inches)
        waste_pct = (waste / mat_total * 100.0) if mat_total > 0 else 0.0
        requires_splice = len(panel_list) > 1
        splice_skus = wall_splice_skus.get(seg.edge_id, [])
        for s_sku in splice_skus:
            all_splice_skus.add(s_sku)

        wp = WallPanelization(
            edge_id=seg.edge_id,
            wall_length_inches=round(wall_length_inches, 4),
            panels=panel_list,
            requires_splice=requires_splice,
            splice_connection_skus=splice_skus,
            total_material_inches=round(mat_total, 4),
            waste_inches=round(waste, 4),
            waste_percentage=round(waste_pct, 4),
            is_panelizable=is_panelizable,
            rejection_reason="" if is_panelizable else f"No valid panel assignment from {solver_name}.",
        )
        wall_panelizations.append(wp)

        if is_panelizable:
            total_material_inches += mat_total
            total_panel_count += len(panel_list)
            total_splice_count += max(0, len(panel_list) - 1)

    panelized_count = sum(1 for wp in wall_panelizations if wp.is_panelizable)

    panel_map = PanelMap(
        walls=wall_panelizations,
        panelized_wall_count=panelized_count,
        total_wall_count=len(wall_panelizations),
        unique_panel_skus=sorted(all_panel_skus),
        unique_splice_skus=sorted(all_splice_skus),
    )

    # ── Build PlacementMap ────────────────────────────────────────────────
    room_placements: list[RoomPlacement] = []
    all_pod_skus: set[str] = set()

    for room in graph.rooms:
        if room.is_exterior:
            continue
        pod_sku = room_assignments.get(room.room_id)
        has_placement = pod_sku is not None
        placement = None
        if has_placement and pod_sku:
            # Position at room centroid
            if room.boundary_nodes and graph.nodes.shape[0] > 0:
                node_coords = graph.nodes[room.boundary_nodes]
                centroid = node_coords.mean(axis=0)
            else:
                centroid = np.zeros(2, dtype=np.float64)

            rotated = room_orientations.get(room.room_id, False)
            placement = ProductPlacement(
                pod_sku=pod_sku,
                position=centroid,
                orientation_deg=90.0 if rotated else 0.0,
                clearance_met=True,
                clearance_margins={},
                confidence=1.0,
            )
            all_pod_skus.add(pod_sku)

        room_placements.append(
            RoomPlacement(
                room_id=room.room_id,
                room_label=room.label,
                room_area_sqft=round(room.area * (to_inches ** 2) / 144.0, 4),
                placement=placement,
                is_eligible=True,
                rejection_reason="" if has_placement else f"No compatible pod found by {solver_name}.",
            )
        )

    placed_count = sum(1 for rp in room_placements if rp.placement is not None)
    eligible_count = len(room_placements)

    placement_map = PlacementMap(
        rooms=room_placements,
        placed_room_count=placed_count,
        eligible_room_count=eligible_count,
        total_room_count=eligible_count,
        unique_pod_skus=sorted(all_pod_skus),
    )

    # ── Summary statistics ────────────────────────────────────────────────
    total_wall_length = sum(
        seg.length * to_inches for seg in graph.wall_segments
    )
    panelized_length = sum(
        wp.wall_length_inches for wp in wall_panelizations if wp.is_panelizable
    )
    coverage_pct = (
        (panelized_length / total_wall_length * 100.0) if total_wall_length > 0 else 0.0
    )

    total_waste = sum(wp.waste_inches for wp in wall_panelizations)
    waste_pct_total = (
        (total_waste / total_material_inches * 100.0) if total_material_inches > 0 else 0.0
    )
    pod_rate = (placed_count / eligible_count * 100.0) if eligible_count > 0 else 0.0

    # SPUR: weighted combination of coverage, efficiency, and pod placement
    w1, w2, w3 = 0.5, 0.3, 0.2
    spur = (
        w1 * (coverage_pct / 100.0)
        + w2 * (1.0 - waste_pct_total / 100.0)
        + w3 * (pod_rate / 100.0)
    )
    spur = max(0.0, min(spur, 1.0))

    return PanelizationResult(
        source_graph=classified_graph,
        panel_map=panel_map,
        placement_map=placement_map,
        spur_score=round(spur, 4),
        coverage_percentage=round(coverage_pct, 4),
        waste_percentage=round(waste_pct_total, 4),
        pod_placement_rate=round(pod_rate, 4),
        total_panel_count=total_panel_count,
        total_splice_count=total_splice_count,
        total_material_cost=round(total_material_cost, 2),
        policy_version=solver_name,
        episode_reward=0.0,
        inference_steps=len(wall_panelizations) + len(room_placements),
        metadata={"solve_time_seconds": solve_time_seconds},
    )
