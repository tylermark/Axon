"""Interface contract: DRL Agent + KG → BOM Agent → Pipeline Output.

Defines the BillOfMaterials dataclass — the output of the BOM Agent
(Phase 9, tasks BM-001 through BM-005) and input to the final pipeline
output (Phase 12).

The BOM Agent receives a PanelizationResult (containing PanelMap,
PlacementMap, ClassifiedWallGraph, and summary statistics) together with
cost data from the Knowledge Graph (``Panel.unit_cost_per_foot``,
``Pod.unit_cost``, ``Connection.unit_cost``) and produces:

    - Itemized CFS component quantities (studs, track, fasteners, clips,
      bridging, blocking, sheathing)
    - Pod component quantities
    - Connection hardware quantities (splices, clips, fasteners)
    - Unit cost and extended cost for every line item
    - Material cost subtotal
    - Labor hour estimates by trade
    - Total project cost breakdown (fabrication, shipping, installation)
    - Export metadata for CSV, Excel, and PDF rendering

This contract intentionally excludes feasibility judgments — that is the
Feasibility Agent's domain (see ``feasibility_report.py``).

Reference: AGENTS.md §BOM Agent, TASKS.md §Phase 9.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from docs.interfaces.drl_output import PanelizationResult


# ══════════════════════════════════════════════════════════════════════════════
# 1. Enums
# ══════════════════════════════════════════════════════════════════════════════


class LineItemCategory(str, Enum):
    """Top-level categories for BOM line items."""

    CFS_STUD = "cfs_stud"
    """Cold-formed steel stud (vertical framing member)."""

    CFS_TRACK = "cfs_track"
    """Cold-formed steel track (top and bottom horizontal members)."""

    FASTENER = "fastener"
    """Screws, bolts, powder-actuated pins, etc."""

    CLIP = "clip"
    """Clip angles and connector clips."""

    BRIDGING = "bridging"
    """Horizontal bridging between studs."""

    BLOCKING = "blocking"
    """Solid blocking between studs for attachment points."""

    SHEATHING = "sheathing"
    """Sheathing panels (OSB, plywood, gypsum, etc.)."""

    POD_ASSEMBLY = "pod_assembly"
    """Complete prefabricated pod unit (bathroom, kitchen, MEP, etc.)."""

    CONNECTION_HARDWARE = "connection_hardware"
    """Splice plates, panel-to-panel connectors, panel-to-floor
    anchors, and other connection hardware."""

    OTHER = "other"
    """Miscellaneous items not covered by standard categories."""


class LaborTrade(str, Enum):
    """Trade classifications for labor hour estimation."""

    FRAMING = "framing"
    """CFS stud and track installation."""

    SHEATHING = "sheathing"
    """Sheathing attachment and finishing."""

    ASSEMBLY = "assembly"
    """Panel assembly on the fabrication line."""

    POD_INSTALL = "pod_install"
    """Pod placement, connection, and MEP hookup on site."""

    GENERAL = "general"
    """General labor not assigned to a specific trade."""


class ExportFormat(str, Enum):
    """Supported BOM export formats."""

    CSV = "csv"
    """Comma-separated values."""

    EXCEL = "excel"
    """Microsoft Excel (.xlsx)."""

    PDF = "pdf"
    """PDF summary report."""


# ══════════════════════════════════════════════════════════════════════════════
# 2. Line Items
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class BOMLineItem:
    """A single line item in the bill of materials.

    Represents one distinct material or component with its quantity,
    unit of measure, and pricing.  Line items are aggregated from the
    PanelMap and PlacementMap — individual panel assignments are rolled
    up by SKU and specification.
    """

    item_id: str
    """Unique line-item identifier within this BOM, e.g. 'LI-001'."""

    category: LineItemCategory
    """Top-level category for grouping and subtotaling."""

    sku: str
    """Product SKU from the Knowledge Graph catalog.

    References ``Panel.sku``, ``Pod.sku``, or ``Connection.sku``
    depending on the category.
    """

    description: str
    """Human-readable description of the item.

    Example: '362S162-54 CFS Stud, 3-5/8" x 1-5/8", 54 mil (16 ga)'
    """

    # ── Specification fields ─────────────────────────────────────────────

    gauge: int | None = None
    """Steel gauge (e.g., 20, 18, 16, 14) for CFS components.

    None for non-steel items (pods, sheathing, etc.).
    """

    depth_inches: float | None = None
    """Stud or track web depth in inches.

    None for items where depth is not applicable.
    """

    length_inches: float | None = None
    """Individual piece length in inches, if applicable.

    For studs: stud height (typically wall height).
    For track: cut length per piece.
    None for items sold by count or area.
    """

    # ── Quantity and pricing ─────────────────────────────────────────────

    quantity: float = 0.0
    """Number of units required.

    For piece goods: count of pieces.
    For linear goods: total linear footage.
    For area goods: total square footage.
    """

    unit: str = ""
    """Unit of measure for the quantity.

    Common values: 'ea' (each), 'lf' (linear feet), 'sf' (square feet),
    'box', 'bag'.
    """

    unit_cost_usd: float = 0.0
    """Cost per unit in US dollars, sourced from the Knowledge Graph.

    Matches ``Panel.unit_cost_per_foot``, ``Pod.unit_cost``, or
    ``Connection.unit_cost`` depending on item type.
    """

    extended_cost_usd: float = 0.0
    """Total cost for this line item: ``quantity * unit_cost_usd``.

    Pre-computed for convenience.  Consumers should verify consistency.
    """

    # ── Traceability ─────────────────────────────────────────────────────

    source_edge_ids: list[int] = field(default_factory=list)
    """Edge IDs of walls that generated demand for this line item.

    Empty for pod and room-level items.
    """

    source_room_ids: list[int] = field(default_factory=list)
    """Room IDs that generated demand for this line item.

    Empty for wall-level CFS items.
    """

    notes: str = ""
    """Additional notes (e.g., 'includes 5% waste allowance',
    'fire-rated assembly required')."""


# ══════════════════════════════════════════════════════════════════════════════
# 3. Labor Estimation
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class LaborEstimate:
    """Labor hour and cost estimate for a single trade.

    Hours are estimated from Capsule Manufacturing's production rates
    (panels per hour, pods per day, etc.) stored in the Knowledge Graph
    or configuration.
    """

    trade: LaborTrade
    """Trade classification."""

    hours: float = 0.0
    """Estimated labor hours for this trade."""

    hourly_rate_usd: float = 0.0
    """Loaded hourly labor rate in US dollars (wages + burden)."""

    cost_usd: float = 0.0
    """Total labor cost for this trade: ``hours * hourly_rate_usd``.

    Pre-computed for convenience.
    """

    crew_size: int = 1
    """Assumed crew size for this trade.

    Hours are total person-hours; elapsed time = hours / crew_size.
    """

    notes: str = ""
    """Additional notes (e.g., 'based on Capsule framing rate of
    8 panels/hr')."""


# ══════════════════════════════════════════════════════════════════════════════
# 4. Project Cost Breakdown
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class ProjectCostBreakdown:
    """Total project cost broken down by phase.

    Separates fabrication (shop), shipping/logistics, and on-site
    installation costs.
    """

    fabrication_material_usd: float = 0.0
    """Total material cost for factory fabrication (sum of all CFS and
    connection hardware line items)."""

    fabrication_labor_usd: float = 0.0
    """Total labor cost for factory fabrication (framing, sheathing,
    assembly trades)."""

    fabrication_subtotal_usd: float = 0.0
    """Fabrication subtotal: ``fabrication_material_usd + fabrication_labor_usd``."""

    pod_cost_usd: float = 0.0
    """Total cost of pod assemblies (material + labor are bundled in
    the pod unit cost from the KG)."""

    shipping_usd: float = 0.0
    """Estimated shipping and logistics cost.

    May be zero if shipping cost data is not available in the KG.
    """

    installation_labor_usd: float = 0.0
    """Estimated on-site installation labor cost (pod hookup, panel
    erection, connection tightening)."""

    installation_material_usd: float = 0.0
    """On-site installation materials (anchors, sealants, field fasteners)
    not included in the panel shop BOM."""

    installation_subtotal_usd: float = 0.0
    """Installation subtotal: ``installation_labor_usd + installation_material_usd``."""

    total_project_cost_usd: float = 0.0
    """Grand total: ``fabrication_subtotal_usd + pod_cost_usd +
    shipping_usd + installation_subtotal_usd``."""

    contingency_pct: float = 0.0
    """Contingency percentage applied to the total (e.g., 10.0 for 10%).

    The ``total_project_cost_usd`` does NOT include contingency.
    Consumers can compute the contingency-adjusted total as:
    ``total_project_cost_usd * (1 + contingency_pct / 100)``.
    """


# ══════════════════════════════════════════════════════════════════════════════
# 5. Material Cost Summary
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class MaterialCostSummary:
    """Subtotals of material cost by line-item category.

    Provides a quick breakdown without iterating all line items.
    """

    cfs_studs_usd: float = 0.0
    """Total cost of CFS stud line items."""

    cfs_track_usd: float = 0.0
    """Total cost of CFS track line items."""

    fasteners_usd: float = 0.0
    """Total cost of fastener line items."""

    clips_usd: float = 0.0
    """Total cost of clip line items."""

    bridging_usd: float = 0.0
    """Total cost of bridging line items."""

    blocking_usd: float = 0.0
    """Total cost of blocking line items."""

    sheathing_usd: float = 0.0
    """Total cost of sheathing line items."""

    pods_usd: float = 0.0
    """Total cost of pod assembly line items."""

    connection_hardware_usd: float = 0.0
    """Total cost of connection hardware line items."""

    other_usd: float = 0.0
    """Total cost of uncategorized line items."""

    material_total_usd: float = 0.0
    """Grand total of all material costs:
    sum of all category subtotals above."""


# ══════════════════════════════════════════════════════════════════════════════
# 6. Export Metadata
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class ExportMetadata:
    """Metadata about the BOM export.

    Tracks which formats were requested, when the BOM was generated,
    and versioning information for reproducibility.
    """

    requested_formats: list[ExportFormat] = field(default_factory=list)
    """Export formats requested by the user or pipeline configuration.

    Default pipeline behavior exports all three: CSV, Excel, PDF.
    """

    generated_at: str = ""
    """ISO 8601 timestamp of when the BOM was generated.

    Example: '2026-04-02T14:30:00Z'.
    """

    generator_version: str = ""
    """Version identifier of the BOM Agent that produced this output."""

    kg_version: str = ""
    """Version of the Knowledge Graph catalog used for pricing.

    Matches ``KnowledgeGraph.version`` from ``schema.py``.
    """

    notes: str = ""
    """Free-form notes about the export (e.g., 'pricing as of Q2 2026')."""


# ══════════════════════════════════════════════════════════════════════════════
# 7. Top-Level BillOfMaterials
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class BillOfMaterials:
    """Complete output of the BOM Agent (Phase 9).

    Wraps all quantity takeoffs, cost estimates, and labor projections
    into a single structure consumed by the final pipeline output
    (Phase 12) and exported to CSV, Excel, and PDF.

    This report answers: "What materials do we need, how much do they
    cost, how long will fabrication and installation take, and what is
    the total project cost?"

    **Consumers:**
    - Pipeline CLI (Phase 12) — renders cost summary and exports files
    - BIM Transplant Agent — reads line items for IFC property sets
      (material specifications attached to IFC entities)
    - Feasibility Agent — may cross-reference material costs for the
      cost-based coverage percentage

    **Must NOT contain:** Feasibility judgments or blocker analysis.
    That is the Feasibility Agent's exclusive domain (see
    ``feasibility_report.py``).

    Reference: AGENTS.md §BOM Agent, TASKS.md §Phase 9.
    """

    source: PanelizationResult
    """The PanelizationResult that was analyzed.

    Retained so downstream consumers can trace line items back to
    specific panel assignments and pod placements.
    """

    # ── Line items ───────────────────────────────────────────────────────

    line_items: list[BOMLineItem]
    """All BOM line items, ordered by category then by SKU.

    Includes CFS components (studs, track, fasteners, clips, bridging,
    blocking, sheathing), pod assemblies, and connection hardware.
    """

    # ── Cost summaries ───────────────────────────────────────────────────

    material_summary: MaterialCostSummary
    """Material cost subtotals broken down by category."""

    # ── Labor ────────────────────────────────────────────────────────────

    labor_estimates: list[LaborEstimate]
    """Labor hour and cost estimates, one per trade.

    Trades include framing, sheathing, assembly, pod installation,
    and general labor.
    """

    total_labor_hours: float = 0.0
    """Sum of labor hours across all trades."""

    total_labor_cost_usd: float = 0.0
    """Sum of labor cost across all trades."""

    # ── Project cost ─────────────────────────────────────────────────────

    project_cost: ProjectCostBreakdown = field(
        default_factory=ProjectCostBreakdown
    )
    """Total project cost breakdown (fabrication, shipping, installation)."""

    # ── Export ────────────────────────────────────────────────────────────

    export: ExportMetadata = field(default_factory=ExportMetadata)
    """Export metadata (formats, timestamp, versioning)."""

    # ── Metadata ─────────────────────────────────────────────────────────

    metadata: dict[str, object] = field(default_factory=dict)
    """Additional metadata (processing time, agent version, etc.)."""
