"""BM-001: CFS quantity takeoff from panelized walls.

Derives cold-formed steel component quantities (studs, track, fasteners,
clips, bridging, blocking, sheathing) from the PanelizationResult and
Knowledge Graph panel specifications.

Each panel assignment is expanded into its constituent CFS members using
the panel's stud spacing, height, gauge, and sheathing attributes from
the KG.  Line items are aggregated by unique specification to minimize
BOM line count.

Reference: TASKS.md Phase 9 BM-001.
"""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from typing import TYPE_CHECKING

from docs.interfaces.bill_of_materials import BOMLineItem, LineItemCategory

if TYPE_CHECKING:
    from docs.interfaces.drl_output import PanelizationResult
    from src.knowledge_graph.loader import KnowledgeGraphStore

logger = logging.getLogger(__name__)

# ── Default fastener schedule ─────────────────────────────────────────────
# 2 screws per stud-to-track connection, top and bottom = 4 per stud.
_SCREWS_PER_STUD: int = 4
# Default screw cost when no fastener connection is found in KG.
_DEFAULT_SCREW_COST_USD: float = 0.03

# ── Clip schedule ─────────────────────────────────────────────────────────
# One clip per panel-to-floor and panel-to-ceiling = 2 per panel.
_CLIPS_PER_PANEL: int = 2
# Default clip cost when not available in KG.
_DEFAULT_CLIP_COST_USD: float = 1.50

# ── Bridging ──────────────────────────────────────────────────────────────
# One row of bridging per panel at mid-height.
_BRIDGING_ROWS_PER_PANEL: int = 1

# ── Blocking ──────────────────────────────────────────────────────────────
# 2 pieces of blocking per opening (header + sill).
# Note: openings are not explicitly tracked in PanelizationResult, so
# we estimate zero unless future revisions add opening data.
_BLOCKING_PER_OPENING: int = 2


def compute_cfs_takeoff(
    result: PanelizationResult,
    store: KnowledgeGraphStore,
) -> list[BOMLineItem]:
    """Compute CFS component takeoff from panelized walls.

    For each panelized wall in the PanelMap, expands panel assignments
    into individual CFS members (studs, track, fasteners, clips,
    bridging, blocking, sheathing) using KG panel specs.

    Args:
        result: The DRL agent's panelization output.
        store: The Knowledge Graph store for panel lookups.

    Returns:
        Aggregated BOM line items for all CFS components plus splice
        connection hardware.
    """
    # Accumulators keyed by (category, aggregation_key) -> running totals
    stud_acc: dict[str, _StudAccum] = {}
    track_acc: dict[str, _TrackAccum] = {}
    fastener_count: int = 0
    clip_count: int = 0
    bridging_acc: dict[str, _BridgingAccum] = {}
    blocking_count: int = 0
    sheathing_acc: dict[str, _SheathingAccum] = {}
    splice_acc: dict[str, _SpliceAccum] = {}

    # Track which edge_ids contribute to each accumulator for traceability
    stud_edges: dict[str, list[int]] = defaultdict(list)
    track_edges: dict[str, list[int]] = defaultdict(list)
    bridging_edges: dict[str, list[int]] = defaultdict(list)
    sheathing_edges: dict[str, list[int]] = defaultdict(list)
    splice_edges: dict[str, list[int]] = defaultdict(list)
    all_fastener_edges: list[int] = []
    all_clip_edges: list[int] = []

    for wall in result.panel_map.walls:
        if not wall.is_panelizable:
            continue

        for pa in wall.panels:
            panel = store.panels.get(pa.panel_sku)
            if panel is None:
                logger.warning(
                    "Panel SKU '%s' not found in KG for edge %d, skipping",
                    pa.panel_sku,
                    wall.edge_id,
                )
                continue

            # ── Studs ────────────────────────────────────────────────
            num_studs = math.ceil(pa.cut_length_inches / panel.stud_spacing_inches) + 1
            stud_key = f"{panel.gauge}ga-{panel.stud_depth_inches}in"
            if stud_key not in stud_acc:
                stud_acc[stud_key] = _StudAccum(
                    gauge=panel.gauge,
                    depth=panel.stud_depth_inches,
                    height=panel.height_inches,
                    sku=pa.panel_sku,
                    unit_cost=panel.unit_cost_per_foot,
                    count=0,
                )
            stud_acc[stud_key].count += num_studs
            stud_edges[stud_key].append(wall.edge_id)

            # ── Track (top + bottom) ─────────────────────────────────
            track_key = f"track-{panel.gauge}ga-{panel.stud_depth_inches}in"
            if track_key not in track_acc:
                track_acc[track_key] = _TrackAccum(
                    gauge=panel.gauge,
                    depth=panel.stud_depth_inches,
                    sku=pa.panel_sku,
                    unit_cost=panel.unit_cost_per_foot,
                    total_lf=0.0,
                    piece_count=0,
                )
            # Two pieces per panel (top and bottom), each = cut_length
            track_acc[track_key].total_lf += (pa.cut_length_inches * 2) / 12.0
            track_acc[track_key].piece_count += 2
            track_edges[track_key].append(wall.edge_id)

            # ── Fasteners ────────────────────────────────────────────
            fastener_count += num_studs * _SCREWS_PER_STUD
            all_fastener_edges.append(wall.edge_id)

            # ── Clips ────────────────────────────────────────────────
            clip_count += _CLIPS_PER_PANEL
            all_clip_edges.append(wall.edge_id)

            # ── Bridging ─────────────────────────────────────────────
            bridging_key = f"bridging-{panel.gauge}ga-{panel.stud_depth_inches}in"
            if bridging_key not in bridging_acc:
                bridging_acc[bridging_key] = _BridgingAccum(
                    gauge=panel.gauge,
                    depth=panel.stud_depth_inches,
                    sku=pa.panel_sku,
                    unit_cost=panel.unit_cost_per_foot,
                    total_lf=0.0,
                )
            bridging_acc[bridging_key].total_lf += (
                pa.cut_length_inches * _BRIDGING_ROWS_PER_PANEL / 12.0
            )
            bridging_edges[bridging_key].append(wall.edge_id)

            # ── Sheathing ────────────────────────────────────────────
            if panel.sheathing_type:
                sheathing_key = (
                    f"sheathing-{panel.sheathing_type}-{panel.sheathing_thickness_inches}in"
                )
                area_sqft = (pa.cut_length_inches * panel.height_inches) / 144.0
                if sheathing_key not in sheathing_acc:
                    sheathing_acc[sheathing_key] = _SheathingAccum(
                        sheathing_type=panel.sheathing_type,
                        thickness=panel.sheathing_thickness_inches or 0.0,
                        total_sqft=0.0,
                    )
                sheathing_acc[sheathing_key].total_sqft += area_sqft
                sheathing_edges[sheathing_key].append(wall.edge_id)

        # ── Splice connections ───────────────────────────────────────
        for splice_sku in wall.splice_connection_skus:
            if splice_sku not in splice_acc:
                conn = store.connections.get(splice_sku)
                splice_acc[splice_sku] = _SpliceAccum(
                    sku=splice_sku,
                    name=conn.name if conn else splice_sku,
                    unit_cost=conn.unit_cost if conn else 0.0,
                    count=0,
                )
            splice_acc[splice_sku].count += 1
            splice_edges[splice_sku].append(wall.edge_id)

    # ── Build line items ─────────────────────────────────────────────────
    items: list[BOMLineItem] = []
    item_counter = 0

    # Studs
    for key, acc in sorted(stud_acc.items()):
        item_counter += 1
        # Cost per stud: (height in feet) * unit_cost_per_foot
        stud_length_ft = acc.height / 12.0
        unit_cost = round(stud_length_ft * acc.unit_cost, 2)
        items.append(
            BOMLineItem(
                item_id=f"LI-{item_counter:03d}",
                category=LineItemCategory.CFS_STUD,
                sku=acc.sku,
                description=(f'CFS Stud {acc.gauge}ga, {acc.depth}" depth, {acc.height}" height'),
                gauge=acc.gauge,
                depth_inches=acc.depth,
                length_inches=acc.height,
                quantity=float(acc.count),
                unit="ea",
                unit_cost_usd=unit_cost,
                extended_cost_usd=round(unit_cost * acc.count, 2),
                source_edge_ids=sorted(set(stud_edges[key])),
            )
        )

    # Track
    for key, acc in sorted(track_acc.items()):
        item_counter += 1
        items.append(
            BOMLineItem(
                item_id=f"LI-{item_counter:03d}",
                category=LineItemCategory.CFS_TRACK,
                sku=acc.sku,
                description=(f'CFS Track {acc.gauge}ga, {acc.depth}" depth'),
                gauge=acc.gauge,
                depth_inches=acc.depth,
                quantity=round(acc.total_lf, 2),
                unit="lf",
                unit_cost_usd=acc.unit_cost,
                extended_cost_usd=round(acc.total_lf * acc.unit_cost, 2),
                source_edge_ids=sorted(set(track_edges[key])),
                notes=f"{acc.piece_count} pieces (top + bottom track)",
            )
        )

    # Fasteners
    if fastener_count > 0:
        item_counter += 1
        items.append(
            BOMLineItem(
                item_id=f"LI-{item_counter:03d}",
                category=LineItemCategory.FASTENER,
                sku="FASTENER-STD",
                description=(
                    f"#8 self-drilling screws, stud-to-track ({_SCREWS_PER_STUD} per stud)"
                ),
                quantity=float(fastener_count),
                unit="ea",
                unit_cost_usd=_DEFAULT_SCREW_COST_USD,
                extended_cost_usd=round(fastener_count * _DEFAULT_SCREW_COST_USD, 2),
                source_edge_ids=sorted(set(all_fastener_edges)),
                notes="Standard fastener schedule: 4 screws per stud",
            )
        )

    # Clips
    if clip_count > 0:
        item_counter += 1
        items.append(
            BOMLineItem(
                item_id=f"LI-{item_counter:03d}",
                category=LineItemCategory.CLIP,
                sku="CLIP-STD",
                description="Clip angle, panel-to-floor/ceiling",
                quantity=float(clip_count),
                unit="ea",
                unit_cost_usd=_DEFAULT_CLIP_COST_USD,
                extended_cost_usd=round(clip_count * _DEFAULT_CLIP_COST_USD, 2),
                source_edge_ids=sorted(set(all_clip_edges)),
                notes=f"{_CLIPS_PER_PANEL} clips per panel (floor + ceiling)",
            )
        )

    # Bridging
    for key, acc in sorted(bridging_acc.items()):
        item_counter += 1
        items.append(
            BOMLineItem(
                item_id=f"LI-{item_counter:03d}",
                category=LineItemCategory.BRIDGING,
                sku=acc.sku,
                description=(f'Bridging {acc.gauge}ga, {acc.depth}" depth, mid-height'),
                gauge=acc.gauge,
                depth_inches=acc.depth,
                quantity=round(acc.total_lf, 2),
                unit="lf",
                unit_cost_usd=acc.unit_cost,
                extended_cost_usd=round(acc.total_lf * acc.unit_cost, 2),
                source_edge_ids=sorted(set(bridging_edges[key])),
                notes="1 row at mid-height per panel",
            )
        )

    # Blocking (currently 0 unless opening data is available)
    if blocking_count > 0:
        item_counter += 1
        items.append(
            BOMLineItem(
                item_id=f"LI-{item_counter:03d}",
                category=LineItemCategory.BLOCKING,
                sku="BLOCKING-STD",
                description="Solid blocking at openings (header/sill)",
                quantity=float(blocking_count),
                unit="ea",
                unit_cost_usd=0.0,
                extended_cost_usd=0.0,
                notes="2 pieces per opening",
            )
        )

    # Sheathing
    for key, acc in sorted(sheathing_acc.items()):
        item_counter += 1
        # Estimate sheathing cost at $1.50/sqft as default
        sheathing_unit_cost = 1.50
        items.append(
            BOMLineItem(
                item_id=f"LI-{item_counter:03d}",
                category=LineItemCategory.SHEATHING,
                sku=f"SHEATHING-{acc.sheathing_type.upper()}",
                description=(f'{acc.sheathing_type} sheathing, {acc.thickness}" thick'),
                quantity=round(acc.total_sqft, 2),
                unit="sf",
                unit_cost_usd=sheathing_unit_cost,
                extended_cost_usd=round(acc.total_sqft * sheathing_unit_cost, 2),
                source_edge_ids=sorted(set(sheathing_edges[key])),
            )
        )

    # Splice connection hardware
    for sku, acc in sorted(splice_acc.items()):
        item_counter += 1
        items.append(
            BOMLineItem(
                item_id=f"LI-{item_counter:03d}",
                category=LineItemCategory.CONNECTION_HARDWARE,
                sku=acc.sku,
                description=f"Splice connection: {acc.name}",
                quantity=float(acc.count),
                unit="ea",
                unit_cost_usd=acc.unit_cost,
                extended_cost_usd=round(acc.count * acc.unit_cost, 2),
                source_edge_ids=sorted(set(splice_edges[sku])),
            )
        )

    logger.info(
        "CFS takeoff: %d line items from %d panelized walls",
        len(items),
        sum(1 for w in result.panel_map.walls if w.is_panelizable),
    )
    return items


# ── Internal accumulator types ─────────────────────────────────────────────


class _StudAccum:
    """Accumulates stud counts by gauge+depth combination."""

    __slots__ = ("count", "depth", "gauge", "height", "sku", "unit_cost")

    def __init__(
        self,
        gauge: int,
        depth: float,
        height: float,
        sku: str,
        unit_cost: float,
        count: int,
    ) -> None:
        self.gauge = gauge
        self.depth = depth
        self.height = height
        self.sku = sku
        self.unit_cost = unit_cost
        self.count = count


class _TrackAccum:
    """Accumulates track linear footage by gauge+depth."""

    __slots__ = ("depth", "gauge", "piece_count", "sku", "total_lf", "unit_cost")

    def __init__(
        self,
        gauge: int,
        depth: float,
        sku: str,
        unit_cost: float,
        total_lf: float,
        piece_count: int,
    ) -> None:
        self.gauge = gauge
        self.depth = depth
        self.sku = sku
        self.unit_cost = unit_cost
        self.total_lf = total_lf
        self.piece_count = piece_count


class _BridgingAccum:
    """Accumulates bridging linear footage by gauge+depth."""

    __slots__ = ("depth", "gauge", "sku", "total_lf", "unit_cost")

    def __init__(
        self,
        gauge: int,
        depth: float,
        sku: str,
        unit_cost: float,
        total_lf: float,
    ) -> None:
        self.gauge = gauge
        self.depth = depth
        self.sku = sku
        self.unit_cost = unit_cost
        self.total_lf = total_lf


class _SheathingAccum:
    """Accumulates sheathing area by type+thickness."""

    __slots__ = ("sheathing_type", "thickness", "total_sqft")

    def __init__(
        self,
        sheathing_type: str,
        thickness: float,
        total_sqft: float,
    ) -> None:
        self.sheathing_type = sheathing_type
        self.thickness = thickness
        self.total_sqft = total_sqft


class _SpliceAccum:
    """Accumulates splice connection counts by SKU."""

    __slots__ = ("count", "name", "sku", "unit_cost")

    def __init__(
        self,
        sku: str,
        name: str,
        unit_cost: float,
        count: int,
    ) -> None:
        self.sku = sku
        self.name = name
        self.unit_cost = unit_cost
        self.count = count
