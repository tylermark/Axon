"""Unit tests for the BOM module (Q-018).

Tests cover all five BOM submodules:
    - cfs_takeoff.py  (BM-001)
    - pod_takeoff.py  (BM-002)
    - costing.py      (BM-003/BM-004)
    - export.py       (BM-005)
    - generator.py    (orchestrator)

Fixture data constructs PanelizationResult, KnowledgeGraphStore, and
supporting types directly — no PDF parsing or DRL inference required.
"""

from __future__ import annotations

import csv
import math
from typing import TYPE_CHECKING

import numpy as np
import pytest

from docs.interfaces.bill_of_materials import (
    BillOfMaterials,
    BOMLineItem,
    ExportFormat,
    ExportMetadata,
    LaborEstimate,
    LaborTrade,
    LineItemCategory,
    MaterialCostSummary,
    ProjectCostBreakdown,
)
from docs.interfaces.classified_wall_graph import (
    ClassifiedWallGraph,
    WallClassification,
)
from docs.interfaces.drl_output import (
    PanelAssignment,
    PanelizationResult,
    PanelMap,
    PlacementMap,
    ProductPlacement,
    RoomPlacement,
    WallPanelization,
)
from docs.interfaces.graph_to_serializer import (
    FinalizedGraph,
    Room,
    WallSegment,
    WallType,
)
from src.bom.cfs_takeoff import compute_cfs_takeoff
from src.bom.costing import (
    compute_labor_estimates,
    compute_material_summary,
    compute_project_cost,
)
from src.bom.export import export_bom
from src.bom.generator import generate_bom
from src.bom.pod_takeoff import compute_pod_takeoff
from src.knowledge_graph.loader import KnowledgeGraphStore
from src.knowledge_graph.schema import (
    Connection,
    KnowledgeGraph,
    Panel,
    PanelType,
    Pod,
)

if TYPE_CHECKING:
    from pathlib import Path


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers — construct minimal valid types
# ═══════════════════════════════════════════════════════════════════════════════


def _make_panel(
    sku: str = "PNL-LB-16-6",
    panel_type: PanelType = PanelType.LOAD_BEARING,
    gauge: int = 16,
    stud_depth_inches: float = 6.0,
    stud_spacing_inches: float = 16.0,
    height_inches: float = 96.0,
    unit_cost_per_foot: float = 14.50,
    sheathing_type: str | None = "OSB",
    sheathing_thickness_inches: float | None = 0.4375,
) -> Panel:
    """Create a minimal Panel for testing."""
    return Panel(
        sku=sku,
        name=f"Test Panel {sku}",
        panel_type=panel_type,
        gauge=gauge,
        stud_depth_inches=stud_depth_inches,
        stud_spacing_inches=stud_spacing_inches,
        min_length_inches=24.0,
        max_length_inches=300.0,
        height_inches=height_inches,
        fire_rating_hours=0.0,
        load_capacity_plf=2100.0,
        sheathing_type=sheathing_type,
        sheathing_thickness_inches=sheathing_thickness_inches,
        insulation_type=None,
        insulation_r_value=None,
        weight_per_foot_lbs=7.2,
        unit_cost_per_foot=unit_cost_per_foot,
        compatible_connections=["CONN-SPLICE-01"],
        fabricated_by=["MACH-RF-01"],
    )


def _make_pod(
    sku: str = "POD-BATH-01",
    name: str = "Standard Bathroom Pod",
    pod_type: str = "bathroom",
    unit_cost: float = 12500.00,
) -> Pod:
    """Create a minimal Pod for testing."""
    return Pod(
        sku=sku,
        name=name,
        pod_type=pod_type,
        width_inches=60.0,
        depth_inches=96.0,
        height_inches=96.0,
        min_room_width_inches=72.0,
        min_room_depth_inches=108.0,
        clearance_inches=3.0,
        included_trades=["plumbing", "electrical"],
        connection_type="bolt-on",
        weight_lbs=3200.0,
        unit_cost=unit_cost,
        lead_time_days=14,
        compatible_panel_types=[PanelType.LOAD_BEARING],
    )


def _make_connection(
    sku: str = "CONN-SPLICE-01",
    name: str = "Panel Splice Plate",
    unit_cost: float = 18.50,
) -> Connection:
    """Create a minimal Connection for testing."""
    return Connection(
        sku=sku,
        name=name,
        connection_type="splice",
        compatible_gauges=[16, 18, 20],
        compatible_stud_depths=[3.5, 6.0],
        load_rating_lbs=5000.0,
        fire_rated=False,
        unit_cost=unit_cost,
        units_per="joint",
    )


def _make_kg_store(
    panels: list[Panel] | None = None,
    pods: list[Pod] | None = None,
    connections: list[Connection] | None = None,
) -> KnowledgeGraphStore:
    """Build a KnowledgeGraphStore from explicit entities."""
    if panels is None:
        panels = [_make_panel()]
    if pods is None:
        pods = [_make_pod()]
    if connections is None:
        connections = [_make_connection()]

    kg = KnowledgeGraph(
        version="test-1.0.0",
        last_updated="2026-04-02",
        panels=panels,
        pods=pods,
        machines=[],
        connections=connections,
        compliance_rules=[],
    )
    store = KnowledgeGraphStore()
    store.load_from_knowledge_graph(kg)
    return store


def _make_wall_segment(
    edge_id: int = 0,
    start_coord: tuple[float, float] = (0.0, 0.0),
    end_coord: tuple[float, float] = (120.0, 0.0),
    thickness: float = 6.0,
    wall_type: WallType = WallType.LOAD_BEARING,
) -> WallSegment:
    """Create a minimal WallSegment."""
    sc = np.array(start_coord, dtype=np.float64)
    ec = np.array(end_coord, dtype=np.float64)
    length = float(np.linalg.norm(ec - sc))
    return WallSegment(
        edge_id=edge_id,
        start_node=edge_id,
        end_node=edge_id + 1,
        start_coord=sc,
        end_coord=ec,
        thickness=thickness,
        height=96.0,
        wall_type=wall_type,
        angle=0.0,
        length=length,
        confidence=0.95,
    )


def _make_finalized_graph(
    wall_segments: list[WallSegment] | None = None,
    rooms: list[Room] | None = None,
) -> FinalizedGraph:
    """Create a minimal FinalizedGraph."""
    if wall_segments is None:
        wall_segments = [_make_wall_segment()]
    if rooms is None:
        rooms = [
            Room(
                room_id=0,
                boundary_edges=[0],
                boundary_nodes=[0, 1],
                area=1000.0,
                label="Bathroom",
            )
        ]

    num_nodes = max(
        max(ws.start_node, ws.end_node) for ws in wall_segments
    ) + 1
    nodes = np.zeros((num_nodes, 2), dtype=np.float64)
    for ws in wall_segments:
        nodes[ws.start_node] = ws.start_coord
        nodes[ws.end_node] = ws.end_coord

    edges = np.array(
        [[ws.start_node, ws.end_node] for ws in wall_segments],
        dtype=np.int64,
    )

    return FinalizedGraph(
        nodes=nodes,
        edges=edges,
        wall_segments=wall_segments,
        openings=[],
        rooms=rooms,
        page_width=612.0,
        page_height=792.0,
    )


def _make_classified_wall_graph(
    graph: FinalizedGraph | None = None,
) -> ClassifiedWallGraph:
    """Create a minimal ClassifiedWallGraph."""
    if graph is None:
        graph = _make_finalized_graph()
    classifications = [
        WallClassification(
            edge_id=ws.edge_id,
            wall_type=ws.wall_type,
            fire_rating="none",
            confidence=0.95,
        )
        for ws in graph.wall_segments
    ]
    return ClassifiedWallGraph(
        graph=graph,
        classifications=classifications,
    )


def _make_panelization_result(
    walls: list[WallPanelization] | None = None,
    rooms_placement: list[RoomPlacement] | None = None,
    source_graph: ClassifiedWallGraph | None = None,
    total_panel_count: int | None = None,
) -> PanelizationResult:
    """Build a PanelizationResult with sensible defaults."""
    if source_graph is None:
        source_graph = _make_classified_wall_graph()

    if walls is None:
        walls = [
            WallPanelization(
                edge_id=0,
                wall_length_inches=120.0,
                panels=[
                    PanelAssignment(
                        panel_sku="PNL-LB-16-6",
                        cut_length_inches=120.0,
                        position_along_wall=0.0,
                        panel_index=0,
                    )
                ],
                requires_splice=False,
                is_panelizable=True,
            )
        ]

    if rooms_placement is None:
        rooms_placement = [
            RoomPlacement(
                room_id=0,
                room_label="Bathroom",
                room_area_sqft=50.0,
                placement=ProductPlacement(
                    pod_sku="POD-BATH-01",
                    position=np.array([60.0, 48.0], dtype=np.float64),
                    orientation_deg=0.0,
                    clearance_met=True,
                ),
            )
        ]

    if total_panel_count is None:
        total_panel_count = sum(
            len(w.panels) for w in walls if w.is_panelizable
        )

    panel_map = PanelMap(
        walls=walls,
        panelized_wall_count=sum(1 for w in walls if w.is_panelizable),
        total_wall_count=len(walls),
    )

    placement_map = PlacementMap(
        rooms=rooms_placement,
        placed_room_count=sum(
            1 for r in rooms_placement if r.placement is not None
        ),
        total_room_count=len(rooms_placement),
    )

    return PanelizationResult(
        source_graph=source_graph,
        panel_map=panel_map,
        placement_map=placement_map,
        total_panel_count=total_panel_count,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.fixture()
def default_panel() -> Panel:
    """A standard 16ga 6in load-bearing panel with OSB sheathing."""
    return _make_panel()


@pytest.fixture()
def default_pod() -> Pod:
    """A standard bathroom pod."""
    return _make_pod()


@pytest.fixture()
def default_connection() -> Connection:
    """A standard splice plate connection."""
    return _make_connection()


@pytest.fixture()
def kg_store(
    default_panel: Panel,
    default_pod: Pod,
    default_connection: Connection,
) -> KnowledgeGraphStore:
    """KG store loaded with one panel, one pod, one connection."""
    return _make_kg_store(
        panels=[default_panel],
        pods=[default_pod],
        connections=[default_connection],
    )


@pytest.fixture()
def single_wall_result() -> PanelizationResult:
    """PanelizationResult with one wall, one panel, one room with pod."""
    return _make_panelization_result()


@pytest.fixture()
def multi_wall_result() -> PanelizationResult:
    """PanelizationResult with multiple walls and a spliced wall."""
    ws0 = _make_wall_segment(edge_id=0, end_coord=(120.0, 0.0))
    ws1 = _make_wall_segment(edge_id=1, start_coord=(120.0, 0.0), end_coord=(240.0, 0.0))
    ws2 = _make_wall_segment(edge_id=2, start_coord=(0.0, 0.0), end_coord=(0.0, 96.0))
    graph = _make_finalized_graph(
        wall_segments=[ws0, ws1, ws2],
        rooms=[
            Room(room_id=0, boundary_edges=[0, 1, 2], boundary_nodes=[0, 1, 2], area=2000.0),
        ],
    )
    cwg = _make_classified_wall_graph(graph)

    walls = [
        WallPanelization(
            edge_id=0,
            wall_length_inches=120.0,
            panels=[
                PanelAssignment(
                    panel_sku="PNL-LB-16-6",
                    cut_length_inches=120.0,
                    position_along_wall=0.0,
                    panel_index=0,
                ),
            ],
            requires_splice=False,
            is_panelizable=True,
        ),
        # Wall 1: spliced into two panels
        WallPanelization(
            edge_id=1,
            wall_length_inches=240.0,
            panels=[
                PanelAssignment(
                    panel_sku="PNL-LB-16-6",
                    cut_length_inches=120.0,
                    position_along_wall=0.0,
                    panel_index=0,
                ),
                PanelAssignment(
                    panel_sku="PNL-LB-16-6",
                    cut_length_inches=120.0,
                    position_along_wall=120.0,
                    panel_index=1,
                ),
            ],
            requires_splice=True,
            splice_connection_skus=["CONN-SPLICE-01"],
            is_panelizable=True,
        ),
        # Wall 2: not panelizable
        WallPanelization(
            edge_id=2,
            wall_length_inches=96.0,
            panels=[],
            requires_splice=False,
            is_panelizable=False,
            rejection_reason="Wall too short for any panel",
        ),
    ]

    rooms_placement = [
        RoomPlacement(
            room_id=0,
            room_label="Bathroom",
            room_area_sqft=50.0,
            placement=ProductPlacement(
                pod_sku="POD-BATH-01",
                position=np.array([60.0, 48.0]),
                orientation_deg=0.0,
                clearance_met=True,
            ),
        ),
    ]

    return _make_panelization_result(
        walls=walls,
        rooms_placement=rooms_placement,
        source_graph=cwg,
        total_panel_count=3,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# CFS Takeoff Tests (BM-001)
# ═══════════════════════════════════════════════════════════════════════════════


class TestCFSTakeoff:
    """Tests for compute_cfs_takeoff."""

    def test_single_panel_stud_count(
        self,
        single_wall_result: PanelizationResult,
        kg_store: KnowledgeGraphStore,
    ) -> None:
        """Stud count = ceil(length / spacing) + 1."""
        items = compute_cfs_takeoff(single_wall_result, kg_store)
        stud_items = [i for i in items if i.category == LineItemCategory.CFS_STUD]
        assert len(stud_items) == 1

        panel = kg_store.panels["PNL-LB-16-6"]
        expected_studs = math.ceil(120.0 / panel.stud_spacing_inches) + 1
        assert stud_items[0].quantity == float(expected_studs)

    def test_track_pieces_two_per_panel(
        self,
        single_wall_result: PanelizationResult,
        kg_store: KnowledgeGraphStore,
    ) -> None:
        """Track: 2 pieces per panel (top + bottom), measured in linear feet."""
        items = compute_cfs_takeoff(single_wall_result, kg_store)
        track_items = [i for i in items if i.category == LineItemCategory.CFS_TRACK]
        assert len(track_items) == 1

        # 2 pieces x 120 inches / 12 = 20.0 lf
        expected_lf = round((120.0 * 2) / 12.0, 2)
        assert track_items[0].quantity == expected_lf
        assert "2 pieces" in track_items[0].notes

    def test_fastener_count_four_per_stud(
        self,
        single_wall_result: PanelizationResult,
        kg_store: KnowledgeGraphStore,
    ) -> None:
        """Fastener count = 4 * stud_count per panel."""
        items = compute_cfs_takeoff(single_wall_result, kg_store)
        fastener_items = [i for i in items if i.category == LineItemCategory.FASTENER]
        assert len(fastener_items) == 1

        panel = kg_store.panels["PNL-LB-16-6"]
        num_studs = math.ceil(120.0 / panel.stud_spacing_inches) + 1
        expected_fasteners = num_studs * 4
        assert fastener_items[0].quantity == float(expected_fasteners)
        assert fastener_items[0].unit_cost_usd == 0.03

    def test_clips_two_per_panel(
        self,
        single_wall_result: PanelizationResult,
        kg_store: KnowledgeGraphStore,
    ) -> None:
        """Clips = 2 per panel (floor + ceiling)."""
        items = compute_cfs_takeoff(single_wall_result, kg_store)
        clip_items = [i for i in items if i.category == LineItemCategory.CLIP]
        assert len(clip_items) == 1
        assert clip_items[0].quantity == 2.0
        assert clip_items[0].unit_cost_usd == 1.50

    def test_bridging_one_row_per_panel(
        self,
        single_wall_result: PanelizationResult,
        kg_store: KnowledgeGraphStore,
    ) -> None:
        """Bridging = 1 row per panel, length = cut_length in lf."""
        items = compute_cfs_takeoff(single_wall_result, kg_store)
        bridging_items = [i for i in items if i.category == LineItemCategory.BRIDGING]
        assert len(bridging_items) == 1

        expected_lf = round(120.0 / 12.0, 2)
        assert bridging_items[0].quantity == expected_lf

    def test_sheathing_area_calculation(
        self,
        single_wall_result: PanelizationResult,
        kg_store: KnowledgeGraphStore,
    ) -> None:
        """Sheathing area = cut_length * height / 144 sqft."""
        items = compute_cfs_takeoff(single_wall_result, kg_store)
        sheathing_items = [i for i in items if i.category == LineItemCategory.SHEATHING]
        assert len(sheathing_items) == 1

        panel = kg_store.panels["PNL-LB-16-6"]
        expected_sqft = round((120.0 * panel.height_inches) / 144.0, 2)
        assert sheathing_items[0].quantity == expected_sqft

    def test_no_sheathing_when_type_is_none(
        self,
        single_wall_result: PanelizationResult,
    ) -> None:
        """No sheathing line items when panel has no sheathing_type."""
        panel_no_sheathing = _make_panel(sheathing_type=None, sheathing_thickness_inches=None)
        store = _make_kg_store(panels=[panel_no_sheathing])

        items = compute_cfs_takeoff(single_wall_result, store)
        sheathing_items = [i for i in items if i.category == LineItemCategory.SHEATHING]
        assert len(sheathing_items) == 0

    def test_splice_connection_hardware(
        self,
        multi_wall_result: PanelizationResult,
        kg_store: KnowledgeGraphStore,
    ) -> None:
        """Splice connections from PanelMap splice_connection_skus."""
        items = compute_cfs_takeoff(multi_wall_result, kg_store)
        conn_items = [i for i in items if i.category == LineItemCategory.CONNECTION_HARDWARE]
        assert len(conn_items) == 1
        assert conn_items[0].sku == "CONN-SPLICE-01"
        assert conn_items[0].quantity == 1.0
        assert conn_items[0].unit_cost_usd == 18.50
        assert conn_items[0].extended_cost_usd == 18.50

    def test_aggregation_same_gauge_depth(
        self,
        multi_wall_result: PanelizationResult,
        kg_store: KnowledgeGraphStore,
    ) -> None:
        """Multiple walls with same gauge+depth produce one aggregated stud line item."""
        items = compute_cfs_takeoff(multi_wall_result, kg_store)
        stud_items = [i for i in items if i.category == LineItemCategory.CFS_STUD]
        # All 3 panels use the same PNL-LB-16-6 (16ga, 6in) so should aggregate
        assert len(stud_items) == 1

        panel = kg_store.panels["PNL-LB-16-6"]
        # Wall 0: 1 panel x 120in -> ceil(120/16)+1 = 9 studs
        # Wall 1: 2 panels x 120in each -> 2 * 9 = 18 studs
        # Total = 27
        studs_per_120 = math.ceil(120.0 / panel.stud_spacing_inches) + 1
        expected_total = studs_per_120 * 3  # 3 panels
        assert stud_items[0].quantity == float(expected_total)

    def test_source_edge_ids_traceability(
        self,
        multi_wall_result: PanelizationResult,
        kg_store: KnowledgeGraphStore,
    ) -> None:
        """Line items carry source_edge_ids for traceability."""
        items = compute_cfs_takeoff(multi_wall_result, kg_store)
        stud_items = [i for i in items if i.category == LineItemCategory.CFS_STUD]
        assert len(stud_items) == 1
        # Edge 0 and 1 are panelizable; edge 2 is not
        assert 0 in stud_items[0].source_edge_ids
        assert 1 in stud_items[0].source_edge_ids
        assert 2 not in stud_items[0].source_edge_ids

    def test_non_panelizable_wall_produces_no_items(
        self,
        kg_store: KnowledgeGraphStore,
    ) -> None:
        """Wall with is_panelizable=False generates no line items."""
        walls = [
            WallPanelization(
                edge_id=0,
                wall_length_inches=48.0,
                panels=[],
                requires_splice=False,
                is_panelizable=False,
                rejection_reason="Too short",
            )
        ]
        result = _make_panelization_result(
            walls=walls,
            rooms_placement=[
                RoomPlacement(room_id=0, is_eligible=False),
            ],
            total_panel_count=0,
        )
        items = compute_cfs_takeoff(result, kg_store)
        assert items == []

    def test_stud_unit_cost_calculation(
        self,
        single_wall_result: PanelizationResult,
        kg_store: KnowledgeGraphStore,
    ) -> None:
        """Stud unit cost = (height_inches / 12) * unit_cost_per_foot."""
        items = compute_cfs_takeoff(single_wall_result, kg_store)
        stud_items = [i for i in items if i.category == LineItemCategory.CFS_STUD]
        panel = kg_store.panels["PNL-LB-16-6"]

        stud_length_ft = panel.height_inches / 12.0
        expected_unit_cost = round(stud_length_ft * panel.unit_cost_per_foot, 2)
        assert stud_items[0].unit_cost_usd == expected_unit_cost

    def test_extended_cost_equals_quantity_times_unit_cost(
        self,
        single_wall_result: PanelizationResult,
        kg_store: KnowledgeGraphStore,
    ) -> None:
        """Extended cost = quantity * unit_cost for each line item."""
        items = compute_cfs_takeoff(single_wall_result, kg_store)
        for item in items:
            expected = round(item.quantity * item.unit_cost_usd, 2)
            assert item.extended_cost_usd == expected, (
                f"{item.category.value}: {item.extended_cost_usd} != {expected}"
            )

    def test_different_gauges_produce_separate_line_items(self) -> None:
        """Panels of different gauge+depth create separate stud line items."""
        panel_16 = _make_panel(sku="PNL-16", gauge=16, stud_depth_inches=6.0)
        panel_20 = _make_panel(sku="PNL-20", gauge=20, stud_depth_inches=3.5)
        store = _make_kg_store(panels=[panel_16, panel_20])

        walls = [
            WallPanelization(
                edge_id=0,
                wall_length_inches=120.0,
                panels=[
                    PanelAssignment(
                        panel_sku="PNL-16",
                        cut_length_inches=120.0,
                        position_along_wall=0.0,
                        panel_index=0,
                    )
                ],
                requires_splice=False,
                is_panelizable=True,
            ),
            WallPanelization(
                edge_id=1,
                wall_length_inches=96.0,
                panels=[
                    PanelAssignment(
                        panel_sku="PNL-20",
                        cut_length_inches=96.0,
                        position_along_wall=0.0,
                        panel_index=0,
                    )
                ],
                requires_splice=False,
                is_panelizable=True,
            ),
        ]

        ws0 = _make_wall_segment(edge_id=0, end_coord=(120.0, 0.0))
        ws1 = _make_wall_segment(edge_id=1, start_coord=(0.0, 0.0), end_coord=(96.0, 0.0))
        graph = _make_finalized_graph(wall_segments=[ws0, ws1], rooms=[])
        cwg = _make_classified_wall_graph(graph)

        result = _make_panelization_result(
            walls=walls,
            rooms_placement=[],
            source_graph=cwg,
        )

        items = compute_cfs_takeoff(result, store)
        stud_items = [i for i in items if i.category == LineItemCategory.CFS_STUD]
        assert len(stud_items) == 2

    def test_unknown_panel_sku_skipped(
        self,
        kg_store: KnowledgeGraphStore,
    ) -> None:
        """Panel with an SKU not in the KG is skipped with a warning."""
        walls = [
            WallPanelization(
                edge_id=0,
                wall_length_inches=120.0,
                panels=[
                    PanelAssignment(
                        panel_sku="NONEXISTENT-SKU",
                        cut_length_inches=120.0,
                        position_along_wall=0.0,
                        panel_index=0,
                    )
                ],
                requires_splice=False,
                is_panelizable=True,
            ),
        ]
        result = _make_panelization_result(
            walls=walls,
            rooms_placement=[],
        )
        items = compute_cfs_takeoff(result, kg_store)
        # No studs, track, etc. because the SKU lookup failed
        assert len(items) == 0


# ═══════════════════════════════════════════════════════════════════════════════
# Pod Takeoff Tests (BM-002)
# ═══════════════════════════════════════════════════════════════════════════════


class TestPodTakeoff:
    """Tests for compute_pod_takeoff."""

    def test_room_with_placed_pod(
        self,
        single_wall_result: PanelizationResult,
        kg_store: KnowledgeGraphStore,
    ) -> None:
        """Room with placed pod generates a POD_ASSEMBLY line item."""
        items = compute_pod_takeoff(single_wall_result, kg_store)
        assert len(items) == 1
        assert items[0].category == LineItemCategory.POD_ASSEMBLY
        assert items[0].sku == "POD-BATH-01"
        assert items[0].quantity == 1.0
        assert items[0].unit_cost_usd == 12500.00
        assert items[0].extended_cost_usd == 12500.00

    def test_room_without_pod_no_item(
        self,
        kg_store: KnowledgeGraphStore,
    ) -> None:
        """Room without a placed pod generates no line item."""
        result = _make_panelization_result(
            rooms_placement=[
                RoomPlacement(
                    room_id=0,
                    room_label="Hallway",
                    is_eligible=False,
                    placement=None,
                ),
            ],
        )
        items = compute_pod_takeoff(result, kg_store)
        assert items == []

    def test_multiple_rooms_same_pod_aggregated(
        self,
        kg_store: KnowledgeGraphStore,
    ) -> None:
        """Multiple rooms with the same pod SKU are aggregated into one line item."""
        rooms = [
            RoomPlacement(
                room_id=0,
                room_label="Bathroom A",
                room_area_sqft=50.0,
                placement=ProductPlacement(
                    pod_sku="POD-BATH-01",
                    position=np.array([60.0, 48.0]),
                    orientation_deg=0.0,
                    clearance_met=True,
                ),
            ),
            RoomPlacement(
                room_id=1,
                room_label="Bathroom B",
                room_area_sqft=55.0,
                placement=ProductPlacement(
                    pod_sku="POD-BATH-01",
                    position=np.array([160.0, 48.0]),
                    orientation_deg=90.0,
                    clearance_met=True,
                ),
            ),
        ]
        result = _make_panelization_result(rooms_placement=rooms)
        items = compute_pod_takeoff(result, kg_store)

        # Same SKU -> aggregated into 1 line item with quantity 2
        assert len(items) == 1
        assert items[0].quantity == 2.0
        assert items[0].extended_cost_usd == round(2 * 12500.00, 2)

    def test_multiple_rooms_different_pods(self) -> None:
        """Rooms with different pod SKUs create separate line items."""
        pod_bath = _make_pod(sku="POD-BATH-01", unit_cost=12500.0)
        pod_kitchen = _make_pod(
            sku="POD-KITCH-01",
            name="Standard Kitchen Pod",
            pod_type="kitchen",
            unit_cost=18000.0,
        )
        store = _make_kg_store(pods=[pod_bath, pod_kitchen])

        rooms = [
            RoomPlacement(
                room_id=0,
                placement=ProductPlacement(
                    pod_sku="POD-BATH-01",
                    position=np.array([60.0, 48.0]),
                    orientation_deg=0.0,
                    clearance_met=True,
                ),
            ),
            RoomPlacement(
                room_id=1,
                placement=ProductPlacement(
                    pod_sku="POD-KITCH-01",
                    position=np.array([160.0, 48.0]),
                    orientation_deg=0.0,
                    clearance_met=True,
                ),
            ),
        ]
        result = _make_panelization_result(rooms_placement=rooms)
        items = compute_pod_takeoff(result, store)

        assert len(items) == 2
        skus = {i.sku for i in items}
        assert skus == {"POD-BATH-01", "POD-KITCH-01"}

    def test_source_room_ids_traceability(
        self,
        kg_store: KnowledgeGraphStore,
    ) -> None:
        """Pod line items carry source_room_ids for traceability."""
        rooms = [
            RoomPlacement(
                room_id=5,
                placement=ProductPlacement(
                    pod_sku="POD-BATH-01",
                    position=np.array([60.0, 48.0]),
                    orientation_deg=0.0,
                    clearance_met=True,
                ),
            ),
            RoomPlacement(
                room_id=9,
                placement=ProductPlacement(
                    pod_sku="POD-BATH-01",
                    position=np.array([160.0, 48.0]),
                    orientation_deg=0.0,
                    clearance_met=True,
                ),
            ),
        ]
        result = _make_panelization_result(rooms_placement=rooms)
        items = compute_pod_takeoff(result, kg_store)

        assert len(items) == 1
        assert items[0].source_room_ids == [5, 9]

    def test_unknown_pod_sku_skipped(self) -> None:
        """Pod with SKU not in KG is skipped."""
        store = _make_kg_store(pods=[])  # No pods in KG
        rooms = [
            RoomPlacement(
                room_id=0,
                placement=ProductPlacement(
                    pod_sku="POD-NONEXISTENT",
                    position=np.array([60.0, 48.0]),
                    orientation_deg=0.0,
                    clearance_met=True,
                ),
            ),
        ]
        result = _make_panelization_result(rooms_placement=rooms)
        items = compute_pod_takeoff(result, store)
        assert items == []


# ═══════════════════════════════════════════════════════════════════════════════
# Costing Tests (BM-003 / BM-004)
# ═══════════════════════════════════════════════════════════════════════════════


class TestMaterialSummary:
    """Tests for compute_material_summary (BM-003)."""

    def test_aggregation_by_category(self) -> None:
        """Material summary aggregates extended_cost by category."""
        items = [
            BOMLineItem(
                item_id="LI-001",
                category=LineItemCategory.CFS_STUD,
                sku="X",
                description="studs",
                quantity=10.0,
                unit="ea",
                unit_cost_usd=5.0,
                extended_cost_usd=50.0,
            ),
            BOMLineItem(
                item_id="LI-002",
                category=LineItemCategory.CFS_STUD,
                sku="Y",
                description="more studs",
                quantity=5.0,
                unit="ea",
                unit_cost_usd=6.0,
                extended_cost_usd=30.0,
            ),
            BOMLineItem(
                item_id="LI-003",
                category=LineItemCategory.FASTENER,
                sku="F",
                description="screws",
                quantity=100.0,
                unit="ea",
                unit_cost_usd=0.03,
                extended_cost_usd=3.0,
            ),
            BOMLineItem(
                item_id="LI-004",
                category=LineItemCategory.POD_ASSEMBLY,
                sku="P",
                description="pod",
                quantity=1.0,
                unit="ea",
                unit_cost_usd=12500.0,
                extended_cost_usd=12500.0,
            ),
        ]
        summary = compute_material_summary(items)

        assert summary.cfs_studs_usd == 80.0  # 50 + 30
        assert summary.fasteners_usd == 3.0
        assert summary.pods_usd == 12500.0
        assert summary.material_total_usd == 80.0 + 3.0 + 12500.0

    def test_empty_items_zero_totals(self) -> None:
        """Empty item list produces all-zero summary."""
        summary = compute_material_summary([])
        assert summary.material_total_usd == 0.0
        assert summary.cfs_studs_usd == 0.0
        assert summary.pods_usd == 0.0

    def test_all_categories_summed(self) -> None:
        """Grand total equals the sum of all category subtotals."""
        items = [
            BOMLineItem(
                item_id="LI-001",
                category=cat,
                sku="X",
                description="test",
                quantity=1.0,
                unit="ea",
                unit_cost_usd=100.0,
                extended_cost_usd=100.0,
            )
            for cat in LineItemCategory
        ]
        summary = compute_material_summary(items)
        expected_total = 100.0 * len(LineItemCategory)
        assert summary.material_total_usd == expected_total


class TestLaborEstimates:
    """Tests for compute_labor_estimates (BM-004)."""

    def test_framing_rate(
        self,
        single_wall_result: PanelizationResult,
        kg_store: KnowledgeGraphStore,
    ) -> None:
        """Framing hours = total_panels / 8 panels per hr."""
        estimates = compute_labor_estimates(single_wall_result, kg_store)
        framing = next(e for e in estimates if e.trade == LaborTrade.FRAMING)

        # hours is rounded for display; cost is computed from unrounded hours
        raw_hours = 1 / 8.0  # 0.125
        expected_hours = round(raw_hours, 2)  # 0.12
        assert framing.hours == expected_hours
        assert framing.hourly_rate_usd == 65.0
        assert framing.cost_usd == round(raw_hours * 65.0, 2)  # 8.12
        assert framing.crew_size == 2

    def test_sheathing_rate(
        self,
        single_wall_result: PanelizationResult,
        kg_store: KnowledgeGraphStore,
    ) -> None:
        """Sheathing hours = sqft / 120 sqft per hr."""
        sheathing_sqft = 50.0
        estimates = compute_labor_estimates(
            single_wall_result, kg_store, sheathing_sqft=sheathing_sqft
        )
        sheathing = next(e for e in estimates if e.trade == LaborTrade.SHEATHING)

        expected_hours = round(50.0 / 120.0, 2)
        assert sheathing.hours == expected_hours

    def test_assembly_rate(
        self,
        multi_wall_result: PanelizationResult,
        kg_store: KnowledgeGraphStore,
    ) -> None:
        """Assembly hours = total_panels / 6 panels per hr."""
        estimates = compute_labor_estimates(multi_wall_result, kg_store)
        assembly = next(e for e in estimates if e.trade == LaborTrade.ASSEMBLY)

        expected_hours = round(3 / 6.0, 2)
        assert assembly.hours == expected_hours
        assert assembly.hourly_rate_usd == 60.0

    def test_pod_install_hours(
        self,
        single_wall_result: PanelizationResult,
        kg_store: KnowledgeGraphStore,
    ) -> None:
        """Pod install hours = placed_pods * 4 hrs/pod."""
        estimates = compute_labor_estimates(single_wall_result, kg_store)
        pod_install = next(e for e in estimates if e.trade == LaborTrade.POD_INSTALL)

        assert pod_install.hours == 4.0
        assert pod_install.hourly_rate_usd == 75.0
        assert pod_install.cost_usd == 300.0
        assert pod_install.crew_size == 4

    def test_general_labor_five_pct_overhead(
        self,
        single_wall_result: PanelizationResult,
        kg_store: KnowledgeGraphStore,
    ) -> None:
        """General labor = 5% of (framing + assembly) hours."""
        estimates = compute_labor_estimates(single_wall_result, kg_store)
        general = next(e for e in estimates if e.trade == LaborTrade.GENERAL)
        framing = next(e for e in estimates if e.trade == LaborTrade.FRAMING)
        assembly = next(e for e in estimates if e.trade == LaborTrade.ASSEMBLY)

        expected = round((framing.hours + assembly.hours) * 0.05, 2)
        assert general.hours == expected

    def test_zero_panels_zero_labor(
        self,
        kg_store: KnowledgeGraphStore,
    ) -> None:
        """Zero panels produces zero framing, assembly, and general labor hours."""
        result = _make_panelization_result(
            walls=[],
            rooms_placement=[],
            total_panel_count=0,
        )
        estimates = compute_labor_estimates(result, kg_store)

        framing = next(e for e in estimates if e.trade == LaborTrade.FRAMING)
        assembly = next(e for e in estimates if e.trade == LaborTrade.ASSEMBLY)
        general = next(e for e in estimates if e.trade == LaborTrade.GENERAL)
        pod_install = next(e for e in estimates if e.trade == LaborTrade.POD_INSTALL)

        assert framing.hours == 0.0
        assert assembly.hours == 0.0
        assert general.hours == 0.0
        assert pod_install.hours == 0.0

    def test_all_five_trades_present(
        self,
        single_wall_result: PanelizationResult,
        kg_store: KnowledgeGraphStore,
    ) -> None:
        """compute_labor_estimates returns estimates for all 5 trades."""
        estimates = compute_labor_estimates(single_wall_result, kg_store)
        trades = {e.trade for e in estimates}
        assert trades == {
            LaborTrade.FRAMING,
            LaborTrade.SHEATHING,
            LaborTrade.ASSEMBLY,
            LaborTrade.POD_INSTALL,
            LaborTrade.GENERAL,
        }


class TestProjectCost:
    """Tests for compute_project_cost."""

    def test_fabrication_subtotal(self) -> None:
        """Fabrication subtotal = fab material + fab labor."""
        summary = MaterialCostSummary(
            cfs_studs_usd=500.0,
            cfs_track_usd=200.0,
            fasteners_usd=10.0,
            clips_usd=20.0,
            bridging_usd=50.0,
            blocking_usd=0.0,
            sheathing_usd=300.0,
            pods_usd=12500.0,
            connection_hardware_usd=18.50,
            material_total_usd=13598.50,
        )
        labor = [
            LaborEstimate(trade=LaborTrade.FRAMING, hours=2.0, hourly_rate_usd=65.0, cost_usd=130.0),
            LaborEstimate(trade=LaborTrade.SHEATHING, hours=1.0, hourly_rate_usd=55.0, cost_usd=55.0),
            LaborEstimate(trade=LaborTrade.ASSEMBLY, hours=1.5, hourly_rate_usd=60.0, cost_usd=90.0),
            LaborEstimate(trade=LaborTrade.POD_INSTALL, hours=4.0, hourly_rate_usd=75.0, cost_usd=300.0),
            LaborEstimate(trade=LaborTrade.GENERAL, hours=0.18, hourly_rate_usd=50.0, cost_usd=9.0),
        ]

        breakdown = compute_project_cost(summary, labor)

        # Fab material = everything except pods
        expected_fab_material = 500.0 + 200.0 + 10.0 + 20.0 + 50.0 + 0.0 + 300.0 + 18.50 + 0.0
        assert breakdown.fabrication_material_usd == round(expected_fab_material, 2)

        # Fab labor = framing + sheathing + assembly
        expected_fab_labor = 130.0 + 55.0 + 90.0
        assert breakdown.fabrication_labor_usd == round(expected_fab_labor, 2)

        expected_fab_subtotal = expected_fab_material + expected_fab_labor
        assert breakdown.fabrication_subtotal_usd == round(expected_fab_subtotal, 2)

    def test_pod_cost_from_material_summary(self) -> None:
        """Pod cost comes directly from material_summary.pods_usd."""
        summary = MaterialCostSummary(pods_usd=25000.0, material_total_usd=25000.0)
        labor = [
            LaborEstimate(trade=LaborTrade.FRAMING),
            LaborEstimate(trade=LaborTrade.SHEATHING),
            LaborEstimate(trade=LaborTrade.ASSEMBLY),
            LaborEstimate(trade=LaborTrade.POD_INSTALL),
            LaborEstimate(trade=LaborTrade.GENERAL),
        ]
        breakdown = compute_project_cost(summary, labor)
        assert breakdown.pod_cost_usd == 25000.0

    def test_installation_material_five_pct_of_fab(self) -> None:
        """Installation material = 5% of fabrication material."""
        summary = MaterialCostSummary(
            cfs_studs_usd=1000.0,
            material_total_usd=1000.0,
        )
        labor = [
            LaborEstimate(trade=LaborTrade.FRAMING),
            LaborEstimate(trade=LaborTrade.SHEATHING),
            LaborEstimate(trade=LaborTrade.ASSEMBLY),
            LaborEstimate(trade=LaborTrade.POD_INSTALL),
            LaborEstimate(trade=LaborTrade.GENERAL),
        ]
        breakdown = compute_project_cost(summary, labor)
        assert breakdown.installation_material_usd == round(1000.0 * 0.05, 2)

    def test_contingency_pct_stored(self) -> None:
        """Contingency percentage is stored on the breakdown."""
        summary = MaterialCostSummary(material_total_usd=0.0)
        labor = [
            LaborEstimate(trade=LaborTrade.FRAMING),
            LaborEstimate(trade=LaborTrade.SHEATHING),
            LaborEstimate(trade=LaborTrade.ASSEMBLY),
            LaborEstimate(trade=LaborTrade.POD_INSTALL),
            LaborEstimate(trade=LaborTrade.GENERAL),
        ]
        breakdown = compute_project_cost(summary, labor, contingency_pct=15.0)
        assert breakdown.contingency_pct == 15.0

    def test_total_sums_correctly(self) -> None:
        """Total = fab_subtotal + pod_cost + install_subtotal."""
        summary = MaterialCostSummary(
            cfs_studs_usd=500.0,
            sheathing_usd=200.0,
            pods_usd=10000.0,
            material_total_usd=10700.0,
        )
        labor = [
            LaborEstimate(trade=LaborTrade.FRAMING, cost_usd=100.0),
            LaborEstimate(trade=LaborTrade.SHEATHING, cost_usd=50.0),
            LaborEstimate(trade=LaborTrade.ASSEMBLY, cost_usd=75.0),
            LaborEstimate(trade=LaborTrade.POD_INSTALL, cost_usd=300.0),
            LaborEstimate(trade=LaborTrade.GENERAL, cost_usd=10.0),
        ]
        breakdown = compute_project_cost(summary, labor)

        fab_material = 500.0 + 200.0  # studs + sheathing (no other CFS)
        fab_labor = 100.0 + 50.0 + 75.0
        fab_subtotal = fab_material + fab_labor
        pod_cost = 10000.0
        install_labor = 300.0 + 10.0  # pod_install + general
        install_material = round(fab_material * 0.05, 2)
        install_subtotal = install_labor + install_material

        expected_total = fab_subtotal + pod_cost + install_subtotal
        assert breakdown.total_project_cost_usd == round(expected_total, 2)

    def test_shipping_is_zero(self) -> None:
        """Shipping is always 0 (not available from KG)."""
        summary = MaterialCostSummary(material_total_usd=0.0)
        labor = [
            LaborEstimate(trade=LaborTrade.FRAMING),
            LaborEstimate(trade=LaborTrade.SHEATHING),
            LaborEstimate(trade=LaborTrade.ASSEMBLY),
            LaborEstimate(trade=LaborTrade.POD_INSTALL),
            LaborEstimate(trade=LaborTrade.GENERAL),
        ]
        breakdown = compute_project_cost(summary, labor)
        assert breakdown.shipping_usd == 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# Export Tests (BM-005)
# ═══════════════════════════════════════════════════════════════════════════════


class TestExport:
    """Tests for export_bom."""

    @pytest.fixture()
    def sample_bom(self) -> BillOfMaterials:
        """A minimal BOM for export testing."""
        items = [
            BOMLineItem(
                item_id="LI-001",
                category=LineItemCategory.CFS_STUD,
                sku="PNL-001",
                description="Test stud",
                gauge=16,
                depth_inches=6.0,
                length_inches=96.0,
                quantity=9.0,
                unit="ea",
                unit_cost_usd=11.60,
                extended_cost_usd=104.40,
                source_edge_ids=[0],
            ),
            BOMLineItem(
                item_id="LI-002",
                category=LineItemCategory.POD_ASSEMBLY,
                sku="POD-BATH-01",
                description="Bathroom pod",
                quantity=1.0,
                unit="ea",
                unit_cost_usd=12500.0,
                extended_cost_usd=12500.0,
                source_room_ids=[0],
            ),
        ]
        summary = MaterialCostSummary(
            cfs_studs_usd=104.40,
            pods_usd=12500.0,
            material_total_usd=12604.40,
        )
        labor = [
            LaborEstimate(
                trade=LaborTrade.FRAMING,
                hours=0.13,
                hourly_rate_usd=65.0,
                cost_usd=8.45,
                crew_size=2,
            ),
        ]
        project_cost = ProjectCostBreakdown(
            fabrication_material_usd=104.40,
            fabrication_labor_usd=8.45,
            fabrication_subtotal_usd=112.85,
            pod_cost_usd=12500.0,
            installation_subtotal_usd=0.0,
            total_project_cost_usd=12612.85,
            contingency_pct=10.0,
        )
        export_meta = ExportMetadata(
            generated_at="2026-04-02T12:00:00Z",
            generator_version="axon-bom v1.0.0",
            kg_version="test-1.0.0",
        )

        # Create a minimal PanelizationResult for the source field
        source = _make_panelization_result()

        return BillOfMaterials(
            source=source,
            line_items=items,
            material_summary=summary,
            labor_estimates=labor,
            total_labor_hours=0.13,
            total_labor_cost_usd=8.45,
            project_cost=project_cost,
            export=export_meta,
        )

    def test_csv_export_creates_file(
        self,
        sample_bom: BillOfMaterials,
        tmp_path: Path,
    ) -> None:
        """CSV export creates a valid file in the output directory."""
        paths = export_bom(sample_bom, tmp_path, formats=[ExportFormat.CSV])

        assert len(paths) == 1
        assert paths[0].suffix == ".csv"
        assert paths[0].exists()

    def test_csv_has_correct_headers(
        self,
        sample_bom: BillOfMaterials,
        tmp_path: Path,
    ) -> None:
        """CSV file has the expected header row."""
        paths = export_bom(sample_bom, tmp_path, formats=[ExportFormat.CSV])
        with open(paths[0], newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            headers = next(reader)

        expected_headers = [
            "item_id", "category", "sku", "description",
            "gauge", "depth_inches", "length_inches",
            "quantity", "unit", "unit_cost_usd", "extended_cost_usd",
            "source_edge_ids", "source_room_ids", "notes",
        ]
        assert headers == expected_headers

    def test_csv_row_count_matches_line_items(
        self,
        sample_bom: BillOfMaterials,
        tmp_path: Path,
    ) -> None:
        """CSV data rows match the number of line items."""
        paths = export_bom(sample_bom, tmp_path, formats=[ExportFormat.CSV])
        with open(paths[0], newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            rows = list(reader)

        # 1 header + 2 data rows
        assert len(rows) == 3

    def test_export_creates_directory(
        self,
        sample_bom: BillOfMaterials,
        tmp_path: Path,
    ) -> None:
        """Export to non-existent directory creates it."""
        deep_dir = tmp_path / "nested" / "output"
        assert not deep_dir.exists()

        paths = export_bom(sample_bom, deep_dir, formats=[ExportFormat.CSV])
        assert deep_dir.exists()
        assert len(paths) == 1

    def test_pdf_export_creates_txt_file(
        self,
        sample_bom: BillOfMaterials,
        tmp_path: Path,
    ) -> None:
        """PDF export creates a .txt text summary file."""
        paths = export_bom(sample_bom, tmp_path, formats=[ExportFormat.PDF])
        assert len(paths) == 1
        assert paths[0].suffix == ".txt"
        assert paths[0].exists()

        content = paths[0].read_text(encoding="utf-8")
        assert "BILL OF MATERIALS REPORT" in content
        assert "LINE ITEMS" in content
        assert "MATERIAL COST SUMMARY" in content

    def test_multiple_formats(
        self,
        sample_bom: BillOfMaterials,
        tmp_path: Path,
    ) -> None:
        """Exporting CSV + PDF creates both files."""
        paths = export_bom(
            sample_bom,
            tmp_path,
            formats=[ExportFormat.CSV, ExportFormat.PDF],
        )
        assert len(paths) == 2
        suffixes = {p.suffix for p in paths}
        assert ".csv" in suffixes
        assert ".txt" in suffixes

    def test_empty_bom_csv_export(
        self,
        tmp_path: Path,
    ) -> None:
        """Exporting a BOM with zero line items produces a CSV with only headers."""
        source = _make_panelization_result(walls=[], rooms_placement=[], total_panel_count=0)
        bom = BillOfMaterials(
            source=source,
            line_items=[],
            material_summary=MaterialCostSummary(),
            labor_estimates=[],
            total_labor_hours=0.0,
            total_labor_cost_usd=0.0,
            project_cost=ProjectCostBreakdown(),
            export=ExportMetadata(
                generated_at="2026-04-02T12:00:00Z",
                generator_version="test",
                kg_version="test",
            ),
        )
        paths = export_bom(bom, tmp_path, formats=[ExportFormat.CSV])
        with open(paths[0], newline="", encoding="utf-8") as f:
            rows = list(csv.reader(f))
        assert len(rows) == 1  # headers only


# ═══════════════════════════════════════════════════════════════════════════════
# Generator Tests (orchestrator)
# ═══════════════════════════════════════════════════════════════════════════════


class TestGenerator:
    """Tests for generate_bom."""

    def test_full_bom_all_fields_populated(
        self,
        single_wall_result: PanelizationResult,
        kg_store: KnowledgeGraphStore,
    ) -> None:
        """Full BOM generation populates all required fields."""
        bom = generate_bom(single_wall_result, kg_store)

        assert isinstance(bom, BillOfMaterials)
        assert len(bom.line_items) > 0
        assert bom.material_summary.material_total_usd > 0
        assert len(bom.labor_estimates) == 5
        assert bom.total_labor_hours > 0
        assert bom.total_labor_cost_usd > 0
        assert bom.project_cost.total_project_cost_usd > 0
        assert bom.export.generated_at != ""
        assert bom.export.generator_version != ""
        assert bom.export.kg_version == "test-1.0.0"

    def test_bom_source_is_input_result(
        self,
        single_wall_result: PanelizationResult,
        kg_store: KnowledgeGraphStore,
    ) -> None:
        """BOM.source references the input PanelizationResult."""
        bom = generate_bom(single_wall_result, kg_store)
        assert bom.source is single_wall_result

    def test_line_items_sorted_by_category_then_sku(
        self,
        multi_wall_result: PanelizationResult,
        kg_store: KnowledgeGraphStore,
    ) -> None:
        """Line items are sorted by category ordinal, then by SKU."""
        bom = generate_bom(multi_wall_result, kg_store)

        category_order = list(LineItemCategory)
        for i in range(len(bom.line_items) - 1):
            a = bom.line_items[i]
            b = bom.line_items[i + 1]
            a_idx = category_order.index(a.category)
            b_idx = category_order.index(b.category)
            assert (a_idx, a.sku) <= (b_idx, b.sku), (
                f"Line items not sorted: {a.item_id} ({a.category}, {a.sku}) "
                f"should come before {b.item_id} ({b.category}, {b.sku})"
            )

    def test_line_item_ids_renumbered_sequentially(
        self,
        multi_wall_result: PanelizationResult,
        kg_store: KnowledgeGraphStore,
    ) -> None:
        """Line item IDs are renumbered LI-001, LI-002, ... after sorting."""
        bom = generate_bom(multi_wall_result, kg_store)
        for idx, item in enumerate(bom.line_items, start=1):
            assert item.item_id == f"LI-{idx:03d}"

    def test_bom_with_no_panels_empty_cfs(
        self,
        kg_store: KnowledgeGraphStore,
    ) -> None:
        """BOM with no panels has empty CFS line items and zero costs."""
        result = _make_panelization_result(
            walls=[
                WallPanelization(
                    edge_id=0,
                    wall_length_inches=48.0,
                    panels=[],
                    requires_splice=False,
                    is_panelizable=False,
                ),
            ],
            rooms_placement=[
                RoomPlacement(room_id=0, is_eligible=False, placement=None),
            ],
            total_panel_count=0,
        )
        bom = generate_bom(result, kg_store)

        assert len(bom.line_items) == 0
        assert bom.material_summary.material_total_usd == 0.0
        assert bom.project_cost.total_project_cost_usd == 0.0

    def test_export_metadata_populated(
        self,
        single_wall_result: PanelizationResult,
        kg_store: KnowledgeGraphStore,
    ) -> None:
        """ExportMetadata has timestamp and KG version."""
        bom = generate_bom(single_wall_result, kg_store)

        assert bom.export.generated_at != ""
        assert "axon-bom" in bom.export.generator_version
        assert bom.export.kg_version == kg_store.version

    def test_metadata_contains_processing_time(
        self,
        single_wall_result: PanelizationResult,
        kg_store: KnowledgeGraphStore,
    ) -> None:
        """BOM metadata contains processing_time_seconds."""
        bom = generate_bom(single_wall_result, kg_store)
        assert "processing_time_seconds" in bom.metadata
        assert bom.metadata["processing_time_seconds"] >= 0

    def test_total_labor_hours_is_sum(
        self,
        single_wall_result: PanelizationResult,
        kg_store: KnowledgeGraphStore,
    ) -> None:
        """total_labor_hours equals sum of individual trade hours."""
        bom = generate_bom(single_wall_result, kg_store)
        expected = round(sum(e.hours for e in bom.labor_estimates), 2)
        assert bom.total_labor_hours == expected

    def test_total_labor_cost_is_sum(
        self,
        single_wall_result: PanelizationResult,
        kg_store: KnowledgeGraphStore,
    ) -> None:
        """total_labor_cost_usd equals sum of individual trade costs."""
        bom = generate_bom(single_wall_result, kg_store)
        expected = round(sum(e.cost_usd for e in bom.labor_estimates), 2)
        assert bom.total_labor_cost_usd == expected

    def test_contingency_pct_forwarded(
        self,
        single_wall_result: PanelizationResult,
        kg_store: KnowledgeGraphStore,
    ) -> None:
        """Custom contingency_pct is forwarded to ProjectCostBreakdown."""
        bom = generate_bom(single_wall_result, kg_store, contingency_pct=15.0)
        assert bom.project_cost.contingency_pct == 15.0

    def test_material_summary_consistent_with_line_items(
        self,
        multi_wall_result: PanelizationResult,
        kg_store: KnowledgeGraphStore,
    ) -> None:
        """Material summary totals are consistent with line item sums."""
        bom = generate_bom(multi_wall_result, kg_store)

        stud_cost = sum(
            i.extended_cost_usd for i in bom.line_items
            if i.category == LineItemCategory.CFS_STUD
        )
        assert bom.material_summary.cfs_studs_usd == round(stud_cost, 2)

        pod_cost = sum(
            i.extended_cost_usd for i in bom.line_items
            if i.category == LineItemCategory.POD_ASSEMBLY
        )
        assert bom.material_summary.pods_usd == round(pod_cost, 2)

        total_from_items = sum(i.extended_cost_usd for i in bom.line_items)
        assert bom.material_summary.material_total_usd == round(total_from_items, 2)
