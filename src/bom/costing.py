"""BM-003 / BM-004: Cost estimation and labor hour estimation.

BM-003: Computes material cost summaries from BOM line items, aggregating
        by LineItemCategory.
BM-004: Estimates labor hours by trade using Capsule Manufacturing's
        production rates.

Reference: TASKS.md Phase 9 BM-003, BM-004.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from docs.interfaces.bill_of_materials import (
    BOMLineItem,
    LaborEstimate,
    LaborTrade,
    LineItemCategory,
    MaterialCostSummary,
    ProjectCostBreakdown,
)

if TYPE_CHECKING:
    from docs.interfaces.drl_output import PanelizationResult
    from src.knowledge_graph.loader import KnowledgeGraphStore

logger = logging.getLogger(__name__)

# ── Capsule production rates ──────────────────────────────────────────────
_FRAMING_PANELS_PER_HOUR: float = 8.0
_SHEATHING_SQFT_PER_HOUR: float = 120.0
_ASSEMBLY_PANELS_PER_HOUR: float = 6.0
_POD_INSTALL_HOURS_PER_POD: float = 4.0

# ── Loaded hourly labor rates (wages + burden) ───────────────────────────
_HOURLY_RATES: dict[LaborTrade, float] = {
    LaborTrade.FRAMING: 65.0,
    LaborTrade.SHEATHING: 55.0,
    LaborTrade.ASSEMBLY: 60.0,
    LaborTrade.POD_INSTALL: 75.0,
    LaborTrade.GENERAL: 50.0,
}

# ── Default crew sizes ───────────────────────────────────────────────────
_CREW_SIZES: dict[LaborTrade, int] = {
    LaborTrade.FRAMING: 2,
    LaborTrade.SHEATHING: 2,
    LaborTrade.ASSEMBLY: 3,
    LaborTrade.POD_INSTALL: 4,
    LaborTrade.GENERAL: 1,
}


def compute_material_summary(
    line_items: list[BOMLineItem],
) -> MaterialCostSummary:
    """Aggregate material costs by line-item category.

    Args:
        line_items: All BOM line items (CFS + pods + connections).

    Returns:
        A MaterialCostSummary with subtotals per category and a
        grand total.
    """
    category_totals: dict[LineItemCategory, float] = {cat: 0.0 for cat in LineItemCategory}

    for item in line_items:
        category_totals[item.category] += item.extended_cost_usd

    material_total = sum(category_totals.values())

    summary = MaterialCostSummary(
        cfs_studs_usd=round(category_totals[LineItemCategory.CFS_STUD], 2),
        cfs_track_usd=round(category_totals[LineItemCategory.CFS_TRACK], 2),
        fasteners_usd=round(category_totals[LineItemCategory.FASTENER], 2),
        clips_usd=round(category_totals[LineItemCategory.CLIP], 2),
        bridging_usd=round(category_totals[LineItemCategory.BRIDGING], 2),
        blocking_usd=round(category_totals[LineItemCategory.BLOCKING], 2),
        sheathing_usd=round(category_totals[LineItemCategory.SHEATHING], 2),
        pods_usd=round(category_totals[LineItemCategory.POD_ASSEMBLY], 2),
        connection_hardware_usd=round(category_totals[LineItemCategory.CONNECTION_HARDWARE], 2),
        other_usd=round(category_totals[LineItemCategory.OTHER], 2),
        material_total_usd=round(material_total, 2),
    )

    logger.info("Material summary: $%.2f total", summary.material_total_usd)
    return summary


def compute_labor_estimates(
    result: PanelizationResult,
    store: KnowledgeGraphStore,
    *,
    sheathing_sqft: float = 0.0,
) -> list[LaborEstimate]:
    """Estimate labor hours and costs by trade.

    Uses Capsule Manufacturing's production rates:
    - Framing: 8 panels/hr
    - Sheathing: 120 sqft/hr
    - Assembly: 6 panels/hr
    - Pod install: 4 hours per pod

    Args:
        result: The DRL agent's panelization output.
        store: The Knowledge Graph store (reserved for future rate lookups).
        sheathing_sqft: Total sheathing area in square feet, extracted
            from line items by the caller.

    Returns:
        Labor estimates, one per trade.
    """
    total_panels = result.total_panel_count
    placed_pods = sum(1 for r in result.placement_map.rooms if r.placement is not None)

    estimates: list[LaborEstimate] = []

    # Framing
    framing_hours = total_panels / _FRAMING_PANELS_PER_HOUR if total_panels > 0 else 0.0
    framing_rate = _HOURLY_RATES[LaborTrade.FRAMING]
    estimates.append(
        LaborEstimate(
            trade=LaborTrade.FRAMING,
            hours=round(framing_hours, 2),
            hourly_rate_usd=framing_rate,
            cost_usd=round(framing_hours * framing_rate, 2),
            crew_size=_CREW_SIZES[LaborTrade.FRAMING],
            notes=f"Based on Capsule framing rate of {_FRAMING_PANELS_PER_HOUR:.0f} panels/hr",
        )
    )

    # Sheathing
    sheathing_hours = sheathing_sqft / _SHEATHING_SQFT_PER_HOUR if sheathing_sqft > 0 else 0.0
    sheathing_rate = _HOURLY_RATES[LaborTrade.SHEATHING]
    estimates.append(
        LaborEstimate(
            trade=LaborTrade.SHEATHING,
            hours=round(sheathing_hours, 2),
            hourly_rate_usd=sheathing_rate,
            cost_usd=round(sheathing_hours * sheathing_rate, 2),
            crew_size=_CREW_SIZES[LaborTrade.SHEATHING],
            notes=f"Based on {_SHEATHING_SQFT_PER_HOUR:.0f} sqft/hr rate",
        )
    )

    # Assembly
    assembly_hours = total_panels / _ASSEMBLY_PANELS_PER_HOUR if total_panels > 0 else 0.0
    assembly_rate = _HOURLY_RATES[LaborTrade.ASSEMBLY]
    estimates.append(
        LaborEstimate(
            trade=LaborTrade.ASSEMBLY,
            hours=round(assembly_hours, 2),
            hourly_rate_usd=assembly_rate,
            cost_usd=round(assembly_hours * assembly_rate, 2),
            crew_size=_CREW_SIZES[LaborTrade.ASSEMBLY],
            notes=f"Based on {_ASSEMBLY_PANELS_PER_HOUR:.0f} panels/hr assembly line rate",
        )
    )

    # Pod install
    pod_hours = placed_pods * _POD_INSTALL_HOURS_PER_POD
    pod_rate = _HOURLY_RATES[LaborTrade.POD_INSTALL]
    estimates.append(
        LaborEstimate(
            trade=LaborTrade.POD_INSTALL,
            hours=round(pod_hours, 2),
            hourly_rate_usd=pod_rate,
            cost_usd=round(pod_hours * pod_rate, 2),
            crew_size=_CREW_SIZES[LaborTrade.POD_INSTALL],
            notes=f"{_POD_INSTALL_HOURS_PER_POD:.0f} hours per pod (placement + MEP hookup)",
        )
    )

    # General labor: 5% of total framing + assembly hours as overhead
    general_hours = round((framing_hours + assembly_hours) * 0.05, 2)
    general_rate = _HOURLY_RATES[LaborTrade.GENERAL]
    estimates.append(
        LaborEstimate(
            trade=LaborTrade.GENERAL,
            hours=general_hours,
            hourly_rate_usd=general_rate,
            cost_usd=round(general_hours * general_rate, 2),
            crew_size=_CREW_SIZES[LaborTrade.GENERAL],
            notes="5% overhead on framing + assembly hours",
        )
    )

    total_hours = sum(e.hours for e in estimates)
    total_cost = sum(e.cost_usd for e in estimates)
    logger.info(
        "Labor estimates: %.1f total hours, $%.2f total cost",
        total_hours,
        total_cost,
    )
    return estimates


def compute_project_cost(
    material_summary: MaterialCostSummary,
    labor_estimates: list[LaborEstimate],
    *,
    contingency_pct: float = 10.0,
) -> ProjectCostBreakdown:
    """Compute the total project cost breakdown.

    Separates fabrication (shop), pod, and installation costs.

    Fabrication materials include CFS components, fasteners, clips,
    bridging, blocking, sheathing, and connection hardware.
    Fabrication labor includes framing, sheathing, and assembly trades.

    Pod cost is taken directly from pod line items (material + labor
    bundled in unit cost).

    Installation includes pod install labor. On-site installation
    material is estimated at 5% of fabrication material as field
    fasteners, sealants, etc.

    Args:
        material_summary: Aggregated material costs by category.
        labor_estimates: Labor estimates by trade.
        contingency_pct: Contingency percentage (default 10%).

    Returns:
        A ProjectCostBreakdown with all subtotals.
    """
    # Fabrication materials: everything except pods
    fab_material = (
        material_summary.cfs_studs_usd
        + material_summary.cfs_track_usd
        + material_summary.fasteners_usd
        + material_summary.clips_usd
        + material_summary.bridging_usd
        + material_summary.blocking_usd
        + material_summary.sheathing_usd
        + material_summary.connection_hardware_usd
        + material_summary.other_usd
    )

    # Fabrication labor: framing + sheathing + assembly
    labor_by_trade = {e.trade: e for e in labor_estimates}
    fab_labor = sum(
        labor_by_trade.get(trade, LaborEstimate(trade=trade)).cost_usd
        for trade in (LaborTrade.FRAMING, LaborTrade.SHEATHING, LaborTrade.ASSEMBLY)
    )

    fab_subtotal = fab_material + fab_labor
    pod_cost = material_summary.pods_usd

    # Installation labor: pod install + general
    install_labor = sum(
        labor_by_trade.get(trade, LaborEstimate(trade=trade)).cost_usd
        for trade in (LaborTrade.POD_INSTALL, LaborTrade.GENERAL)
    )

    # Installation material: estimate 5% of fabrication material for
    # field fasteners, sealants, anchors, etc.
    install_material = round(fab_material * 0.05, 2)
    install_subtotal = install_labor + install_material

    total = fab_subtotal + pod_cost + install_subtotal
    # Shipping is not available from KG; set to 0
    shipping = 0.0

    breakdown = ProjectCostBreakdown(
        fabrication_material_usd=round(fab_material, 2),
        fabrication_labor_usd=round(fab_labor, 2),
        fabrication_subtotal_usd=round(fab_subtotal, 2),
        pod_cost_usd=round(pod_cost, 2),
        shipping_usd=shipping,
        installation_labor_usd=round(install_labor, 2),
        installation_material_usd=install_material,
        installation_subtotal_usd=round(install_subtotal, 2),
        total_project_cost_usd=round(total, 2),
        contingency_pct=contingency_pct,
    )

    logger.info(
        "Project cost: $%.2f (fab: $%.2f, pods: $%.2f, install: $%.2f, contingency: %.0f%%)",
        breakdown.total_project_cost_usd,
        breakdown.fabrication_subtotal_usd,
        breakdown.pod_cost_usd,
        breakdown.installation_subtotal_usd,
        breakdown.contingency_pct,
    )
    return breakdown
