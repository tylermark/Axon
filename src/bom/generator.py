"""Top-level BOM orchestrator: PanelizationResult + KG -> BillOfMaterials.

Coordinates all takeoff modules (CFS, pod), costing, labor estimation,
and metadata assembly into a single BillOfMaterials output.

Reference: TASKS.md Phase 9, all BM tasks.
"""

from __future__ import annotations

import logging
import time
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from docs.interfaces.bill_of_materials import (
    BillOfMaterials,
    ExportMetadata,
    LineItemCategory,
)
from src.bom.cfs_takeoff import compute_cfs_takeoff
from src.bom.costing import (
    compute_labor_estimates,
    compute_material_summary,
    compute_project_cost,
)
from src.bom.pod_takeoff import compute_pod_takeoff

if TYPE_CHECKING:
    from docs.interfaces.drl_output import PanelizationResult
    from src.knowledge_graph.loader import KnowledgeGraphStore

logger = logging.getLogger(__name__)

_BOM_AGENT_VERSION: str = "1.0.0"


def generate_bom(
    result: PanelizationResult,
    store: KnowledgeGraphStore,
    *,
    contingency_pct: float = 10.0,
) -> BillOfMaterials:
    """Generate a complete BillOfMaterials from panelization output.

    Orchestrates:
    1. CFS quantity takeoff (BM-001)
    2. Pod component takeoff (BM-002)
    3. Material cost summary (BM-003)
    4. Labor hour estimation (BM-004)
    5. Project cost breakdown (BM-003 continued)

    Export (BM-005) is handled separately via ``export_bom()``.

    Args:
        result: The DRL agent's panelization output.
        store: The Knowledge Graph store for pricing lookups.
        contingency_pct: Contingency percentage for project cost
            (default 10%).

    Returns:
        A fully populated BillOfMaterials.
    """
    start = time.monotonic()

    # ── Step 1: CFS takeoff ──────────────────────────────────────────
    cfs_items = compute_cfs_takeoff(result, store)

    # ── Step 2: Pod takeoff ──────────────────────────────────────────
    pod_items = compute_pod_takeoff(result, store)

    # ── Merge and sort line items ────────────────────────────────────
    all_items = cfs_items + pod_items
    # Sort by category ordinal, then by SKU for deterministic output
    category_order = list(LineItemCategory)
    all_items.sort(key=lambda li: (category_order.index(li.category), li.sku))
    # Re-number item IDs sequentially after sort
    for idx, item in enumerate(all_items, start=1):
        item.item_id = f"LI-{idx:03d}"

    # ── Step 3: Material cost summary ────────────────────────────────
    material_summary = compute_material_summary(all_items)

    # ── Step 4: Labor estimation ─────────────────────────────────────
    # Extract total sheathing sqft from line items for labor calc
    sheathing_sqft = sum(
        item.quantity for item in all_items if item.category == LineItemCategory.SHEATHING
    )
    labor_estimates = compute_labor_estimates(result, store, sheathing_sqft=sheathing_sqft)

    total_labor_hours = round(sum(e.hours for e in labor_estimates), 2)
    total_labor_cost = round(sum(e.cost_usd for e in labor_estimates), 2)

    # ── Step 5: Project cost breakdown ───────────────────────────────
    project_cost = compute_project_cost(
        material_summary,
        labor_estimates,
        contingency_pct=contingency_pct,
    )

    # ── Export metadata ──────────────────────────────────────────────
    export_meta = ExportMetadata(
        generated_at=datetime.now(tz=UTC).isoformat(),
        generator_version=f"axon-bom v{_BOM_AGENT_VERSION}",
        kg_version=store.version,
    )

    elapsed = time.monotonic() - start

    bom = BillOfMaterials(
        source=result,
        line_items=all_items,
        material_summary=material_summary,
        labor_estimates=labor_estimates,
        total_labor_hours=total_labor_hours,
        total_labor_cost_usd=total_labor_cost,
        project_cost=project_cost,
        export=export_meta,
        metadata={
            "processing_time_seconds": round(elapsed, 4),
            "bom_agent_version": _BOM_AGENT_VERSION,
        },
    )

    logger.info(
        "BOM generated: %d line items, $%.2f material, $%.2f labor, $%.2f project total (%.3fs)",
        len(bom.line_items),
        bom.material_summary.material_total_usd,
        bom.total_labor_cost_usd,
        bom.project_cost.total_project_cost_usd,
        elapsed,
    )
    return bom
