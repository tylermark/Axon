"""BM-002: Pod component takeoff from room placements.

Derives pod assembly line items from the PlacementMap, looking up each
placed pod's unit cost in the Knowledge Graph.

Reference: TASKS.md Phase 9 BM-002.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from docs.interfaces.bill_of_materials import BOMLineItem, LineItemCategory

if TYPE_CHECKING:
    from docs.interfaces.drl_output import PanelizationResult
    from src.knowledge_graph.loader import KnowledgeGraphStore

logger = logging.getLogger(__name__)


def compute_pod_takeoff(
    result: PanelizationResult,
    store: KnowledgeGraphStore,
    *,
    start_item_id: int = 500,
) -> list[BOMLineItem]:
    """Compute pod assembly takeoff from room placements.

    For each room with a placed pod, creates a POD_ASSEMBLY line item
    with quantity=1 and unit_cost sourced from ``Pod.unit_cost`` in the KG.

    Args:
        result: The DRL agent's panelization output.
        store: The Knowledge Graph store for pod lookups.
        start_item_id: Starting line-item counter to avoid collisions
            with CFS takeoff item IDs.

    Returns:
        BOM line items for all placed pods.
    """
    items: list[BOMLineItem] = []
    # Aggregate by pod SKU: collect room IDs per unique pod
    pod_rooms: dict[str, list[int]] = {}

    for room in result.placement_map.rooms:
        if room.placement is None:
            continue

        pod_sku = room.placement.pod_sku
        if pod_sku not in pod_rooms:
            pod_rooms[pod_sku] = []
        pod_rooms[pod_sku].append(room.room_id)

    item_counter = start_item_id
    for pod_sku, room_ids in sorted(pod_rooms.items()):
        pod = store.pods.get(pod_sku)
        if pod is None:
            logger.warning(
                "Pod SKU '%s' not found in KG for rooms %s, skipping",
                pod_sku,
                room_ids,
            )
            continue

        quantity = len(room_ids)
        item_counter += 1
        items.append(
            BOMLineItem(
                item_id=f"LI-{item_counter:03d}",
                category=LineItemCategory.POD_ASSEMBLY,
                sku=pod_sku,
                description=f"{pod.name} ({pod.pod_type})",
                quantity=float(quantity),
                unit="ea",
                unit_cost_usd=pod.unit_cost,
                extended_cost_usd=round(quantity * pod.unit_cost, 2),
                source_room_ids=sorted(room_ids),
            )
        )

    logger.info(
        "Pod takeoff: %d line items from %d placed rooms",
        len(items),
        sum(1 for r in result.placement_map.rooms if r.placement is not None),
    )
    return items
