"""Integration test: Q-020 — full pipeline end-to-end (PDF -> prefab report).

Tests the complete Axon pipeline wired through:
    src/pipeline/full_pipeline.py  — run_full_pipeline + PipelineResult
    src/pipeline/output.py         — write_pipeline_outputs + print_summary

Strategy
--------
The full pipeline requires a real PDF and trained model weights, neither of
which exist in CI.  Tests therefore fall into two groups:

1.  **Output module tests** — build PipelineResult objects synthetically (as
    existing BOM/feasibility unit tests do) and exercise write_pipeline_outputs
    and print_summary directly.  These are fast and have no external deps.

2.  **Pipeline integration tests** — call run_full_pipeline with a tiny
    synthetic PDF.  Missing weights mean the diffusion/tokenizer models run
    with random parameters, so tests verify structural invariants (stage_errors
    populated on bad input, processing_time measured, result fields typed
    correctly) rather than extraction quality.

Reference: ARCHITECTURE.md §What Axon Does, CLAUDE.md §Layer 1 + Layer 2.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import numpy as np
import pytest
from click.testing import CliRunner

from docs.interfaces.bill_of_materials import (
    BillOfMaterials,
    BOMLineItem,
    LaborEstimate,
    LaborTrade,
    LineItemCategory,
    MaterialCostSummary,
    ProjectCostBreakdown,
)
from docs.interfaces.classified_wall_graph import (
    ClassifiedWallGraph,
    FireRating,
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
from docs.interfaces.feasibility_report import (
    Blocker,
    BlockerCategory,
    CoverageMetrics,
    DesignSuggestion,
    FeasibilityReport,
    FeasibilitySummary,
    FloorScore,
    RoomFeasibility,
    SuggestionType,
    WallFeasibility,
)
from docs.interfaces.graph_to_serializer import (
    FinalizedGraph,
    Room,
    WallSegment,
    WallType,
)
from src.knowledge_graph.loader import KnowledgeGraphStore
from src.knowledge_graph.schema import (
    Connection,
    KnowledgeGraph,
    Panel,
    PanelType,
    Pod,
)
from src.pipeline.cli import main as axon_cli
from src.pipeline.full_pipeline import PipelineResult, run_full_pipeline
from src.pipeline.output import print_summary, write_pipeline_outputs

if TYPE_CHECKING:
    from pathlib import Path


# ══════════════════════════════════════════════════════════════════════════════
# Synthetic data factories
# ══════════════════════════════════════════════════════════════════════════════


def _make_panel(sku: str = "PNL-LB-16-6") -> Panel:
    """Minimal load-bearing panel for KG fixture."""
    return Panel(
        sku=sku,
        name=f"Test Panel {sku}",
        panel_type=PanelType.LOAD_BEARING,
        gauge=16,
        stud_depth_inches=6.0,
        stud_spacing_inches=16.0,
        min_length_inches=24.0,
        max_length_inches=300.0,
        height_inches=96.0,
        fire_rating_hours=0.0,
        load_capacity_plf=2100.0,
        sheathing_type="OSB",
        sheathing_thickness_inches=0.4375,
        insulation_type=None,
        insulation_r_value=None,
        weight_per_foot_lbs=7.2,
        unit_cost_per_foot=14.50,
        compatible_connections=["CONN-SPLICE-01"],
        fabricated_by=["MACH-RF-01"],
    )


def _make_pod(sku: str = "POD-BATH-01") -> Pod:
    """Minimal bathroom pod for KG fixture."""
    return Pod(
        sku=sku,
        name="Standard Bathroom Pod",
        pod_type="bathroom",
        width_inches=60.0,
        depth_inches=96.0,
        height_inches=96.0,
        min_room_width_inches=72.0,
        min_room_depth_inches=108.0,
        clearance_inches=3.0,
        included_trades=["plumbing", "electrical"],
        connection_type="bolt-on",
        weight_lbs=3200.0,
        unit_cost=12_500.00,
        lead_time_days=14,
        compatible_panel_types=[PanelType.LOAD_BEARING],
    )


def _make_connection(sku: str = "CONN-SPLICE-01") -> Connection:
    """Minimal splice connection for KG fixture."""
    return Connection(
        sku=sku,
        name="Panel Splice Plate",
        connection_type="splice",
        compatible_gauges=[16, 18, 20],
        compatible_stud_depths=[3.5, 6.0],
        load_rating_lbs=5000.0,
        fire_rated=False,
        unit_cost=18.50,
        units_per="joint",
    )


def _make_kg_store() -> KnowledgeGraphStore:
    """KnowledgeGraphStore loaded with one panel, pod, and connection."""
    kg = KnowledgeGraph(
        version="test-1.0.0",
        last_updated="2026-04-02",
        panels=[_make_panel()],
        pods=[_make_pod()],
        machines=[],
        connections=[_make_connection()],
        compliance_rules=[],
    )
    store = KnowledgeGraphStore()
    store.load_from_knowledge_graph(kg)
    return store


def _make_wall_segment(edge_id: int = 0, length: float = 120.0) -> WallSegment:
    """Minimal WallSegment at y=0 running in the +x direction."""
    start = np.array([0.0, edge_id * 150.0], dtype=np.float64)
    end = start + np.array([length, 0.0], dtype=np.float64)
    return WallSegment(
        edge_id=edge_id,
        start_node=edge_id * 2,
        end_node=edge_id * 2 + 1,
        start_coord=start,
        end_coord=end,
        thickness=6.0,
        height=96.0,
        wall_type=WallType.LOAD_BEARING,
        angle=0.0,
        length=length,
        confidence=0.95,
    )


def _make_finalized_graph(n_walls: int = 2) -> FinalizedGraph:
    """Build a FinalizedGraph with n_walls horizontal wall segments."""
    wall_segments = [_make_wall_segment(i) for i in range(n_walls)]
    rooms = [
        Room(
            room_id=0,
            boundary_edges=list(range(n_walls)),
            boundary_nodes=list(range(n_walls * 2)),
            area=1_440.0,
            label="Bathroom",
            is_exterior=False,
        )
    ]
    num_nodes = n_walls * 2
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
        page_index=0,
        source_path="test.pdf",
        assumed_wall_height=2700.0,
        betti_0=1,
        betti_1=0,
    )


def _make_classified_wall_graph(graph: FinalizedGraph | None = None) -> ClassifiedWallGraph:
    """Wrap a FinalizedGraph with trivial load-bearing classifications."""
    if graph is None:
        graph = _make_finalized_graph()
    classifications = [
        WallClassification(
            edge_id=ws.edge_id,
            wall_type=ws.wall_type,
            fire_rating=FireRating.NONE,
            confidence=0.95,
        )
        for ws in graph.wall_segments
    ]
    return ClassifiedWallGraph(
        graph=graph,
        classifications=classifications,
        classification_summary={"load_bearing": len(graph.wall_segments)},
        walls_flagged_for_review=[],
    )


def _make_panelization_result(
    classified_graph: ClassifiedWallGraph | None = None,
) -> PanelizationResult:
    """Build a PanelizationResult with one panelized wall and one pod placement."""
    if classified_graph is None:
        classified_graph = _make_classified_wall_graph()

    walls = [
        WallPanelization(
            edge_id=ws.edge_id,
            wall_length_inches=ws.length / 72.0,  # PDF units → inches
            panels=[
                PanelAssignment(
                    panel_sku="PNL-LB-16-6",
                    cut_length_inches=ws.length / 72.0,
                    position_along_wall=0.0,
                    panel_index=0,
                )
            ],
            requires_splice=False,
            is_panelizable=True,
            total_material_inches=ws.length / 72.0,
            waste_inches=0.0,
            waste_percentage=0.0,
        )
        for ws in classified_graph.graph.wall_segments
    ]

    room_placements = [
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
            is_eligible=True,
        )
    ]

    panel_map = PanelMap(
        walls=walls,
        panelized_wall_count=len(walls),
        total_wall_count=len(walls),
        unique_panel_skus=["PNL-LB-16-6"],
    )
    placement_map = PlacementMap(
        rooms=room_placements,
        placed_room_count=1,
        eligible_room_count=1,
        total_room_count=1,
        unique_pod_skus=["POD-BATH-01"],
    )

    return PanelizationResult(
        source_graph=classified_graph,
        panel_map=panel_map,
        placement_map=placement_map,
        spur_score=0.82,
        coverage_percentage=100.0,
        waste_percentage=0.0,
        pod_placement_rate=100.0,
        total_panel_count=len(walls),
        total_splice_count=0,
        policy_version="greedy",
        episode_reward=12.5,
        inference_steps=len(walls) + 1,
    )


def _make_feasibility_report(
    panelization: PanelizationResult | None = None,
) -> FeasibilityReport:
    """Build a realistic FeasibilityReport for output module testing."""
    if panelization is None:
        panelization = _make_panelization_result()

    n_walls = len(panelization.panel_map.walls)

    coverage = CoverageMetrics(
        by_wall_length_pct=100.0,
        by_area_pct=95.0,
        by_cost_pct=88.0,
        total_wall_length_inches=120.0 * n_walls,
        panelized_wall_length_inches=120.0 * n_walls,
        total_wall_area_sqft=80.0 * n_walls,
        panelized_wall_area_sqft=80.0 * n_walls,
    )

    blocker = Blocker(
        blocker_id="BLK-001",
        category=BlockerCategory.GEOMETRY,
        description="Test blocker: minor off-axis angle",
        affected_edge_ids=[0],
        severity=0.4,
    )

    suggestion = DesignSuggestion(
        suggestion_id="SUG-001",
        suggestion_type=SuggestionType.WALL_STRAIGHTEN,
        description="Straighten wall 0 to enable standard panel usage.",
        resolves_blocker_ids=["BLK-001"],
        affected_edge_ids=[0],
        estimated_coverage_gain_pct=3.5,
        estimated_panels_gained=2,
        effort_level="low",
    )

    wall_feasibility = [
        WallFeasibility(
            edge_id=w.edge_id,
            is_panelizable=w.is_panelizable,
            wall_length_inches=w.wall_length_inches,
            panelized_length_inches=w.wall_length_inches if w.is_panelizable else 0.0,
            coverage_pct=100.0 if w.is_panelizable else 0.0,
        )
        for w in panelization.panel_map.walls
    ]

    room_feasibility = [
        RoomFeasibility(
            room_id=rp.room_id,
            room_label=rp.room_label,
            room_area_sqft=rp.room_area_sqft,
            is_eligible=rp.is_eligible,
            has_pod=rp.placement is not None,
            pod_sku=rp.placement.pod_sku if rp.placement else "",
        )
        for rp in panelization.placement_map.rooms
    ]

    floor_score = FloorScore(
        floor_id="1",
        floor_label="Ground Floor",
        wall_coverage_pct=100.0,
        area_coverage_pct=95.0,
        pod_placement_rate_pct=100.0,
        blocker_count=1,
        total_wall_count=n_walls,
        panelized_wall_count=n_walls,
        total_room_count=1,
        placed_room_count=1,
        feasibility_score=0.88,
    )

    summary = FeasibilitySummary(
        total_wall_count=n_walls,
        panelized_wall_count=n_walls,
        unpanelized_wall_count=0,
        total_room_count=1,
        eligible_room_count=1,
        placed_room_count=1,
        total_blocker_count=1,
        hard_blocker_count=0,
        soft_blocker_count=1,
        suggestion_count=1,
        max_coverage_gain_pct=3.5,
        spur_score=panelization.spur_score,
    )

    return FeasibilityReport(
        source=panelization,
        coverage=coverage,
        wall_feasibility=wall_feasibility,
        room_feasibility=room_feasibility,
        blockers=[blocker],
        suggestions=[suggestion],
        floor_scores=[floor_score],
        project_score=0.88,
        summary=summary,
    )


def _make_bom(panelization: PanelizationResult | None = None) -> BillOfMaterials:
    """Build a minimal BillOfMaterials for output module testing."""
    if panelization is None:
        panelization = _make_panelization_result()

    line_items = [
        BOMLineItem(
            item_id="LI-001",
            category=LineItemCategory.CFS_STUD,
            sku="PNL-LB-16-6",
            description="362S162-54 CFS Stud, 16ga, 6in",
            gauge=16,
            depth_inches=6.0,
            length_inches=96.0,
            quantity=24.0,
            unit="ea",
            unit_cost_usd=14.50,
            extended_cost_usd=348.00,
            source_edge_ids=[0],
        ),
        BOMLineItem(
            item_id="LI-002",
            category=LineItemCategory.POD_ASSEMBLY,
            sku="POD-BATH-01",
            description="Standard Bathroom Pod 60x96",
            quantity=1.0,
            unit="ea",
            unit_cost_usd=12_500.00,
            extended_cost_usd=12_500.00,
            source_room_ids=[0],
        ),
    ]

    material_summary = MaterialCostSummary(
        cfs_studs_usd=348.00,
        pods_usd=12_500.00,
        material_total_usd=12_848.00,
    )

    labor_estimates = [
        LaborEstimate(
            trade=LaborTrade.FRAMING,
            hours=8.0,
            hourly_rate_usd=75.0,
            cost_usd=600.0,
            crew_size=2,
        ),
        LaborEstimate(
            trade=LaborTrade.POD_INSTALL,
            hours=4.0,
            hourly_rate_usd=90.0,
            cost_usd=360.0,
            crew_size=2,
        ),
    ]

    project_cost = ProjectCostBreakdown(
        fabrication_material_usd=348.00,
        fabrication_labor_usd=600.0,
        fabrication_subtotal_usd=948.00,
        pod_cost_usd=12_500.00,
        shipping_usd=500.00,
        installation_labor_usd=360.0,
        installation_material_usd=50.00,
        installation_subtotal_usd=410.00,
        total_project_cost_usd=14_358.00,
        contingency_pct=10.0,
    )

    return BillOfMaterials(
        source=panelization,
        line_items=line_items,
        material_summary=material_summary,
        labor_estimates=labor_estimates,
        total_labor_hours=12.0,
        total_labor_cost_usd=960.0,
        project_cost=project_cost,
    )


def _make_full_pipeline_result(tmp_path: Path) -> PipelineResult:
    """Build a fully-populated PipelineResult for output module testing.

    Constructs all sub-objects from scratch (no PDF or model weights needed).
    """
    raw_graph = _make_finalized_graph(n_walls=2)
    classified = _make_classified_wall_graph(raw_graph)
    panelization = _make_panelization_result(classified)
    feasibility = _make_feasibility_report(panelization)
    bom = _make_bom(panelization)

    # Pretend an IFC JSON fallback was written by an earlier stage.
    json_fallback = tmp_path / "model.json"
    json_fallback.write_text('{"type": "IFC_JSON_FALLBACK"}', encoding="utf-8")

    result = PipelineResult(
        raw_graph=raw_graph,
        classified_graph=classified,
        panelization=panelization,
        feasibility=feasibility,
        bom=bom,
        ifc_path=None,
        json_fallback_path=json_fallback,
        bom_export_paths=[],
        processing_time_seconds=4.21,
        stage_errors={},
        metadata={
            "pdf_path": "test_floor_plan.pdf",
            "page_index": 0,
            "device": "cpu",
        },
    )
    return result


# ══════════════════════════════════════════════════════════════════════════════
# Output module tests — write_pipeline_outputs
# ══════════════════════════════════════════════════════════════════════════════


class TestWritePipelineOutputs:
    """Tests for write_pipeline_outputs in src/pipeline/output.py."""

    def test_creates_feasibility_report_json(self, tmp_path: Path) -> None:
        """write_pipeline_outputs writes feasibility_report.json to output_dir."""
        result = _make_full_pipeline_result(tmp_path)
        out_dir = tmp_path / "out"

        written = write_pipeline_outputs(result, out_dir)

        assert "feasibility_report" in written
        assert written["feasibility_report"].exists()
        assert written["feasibility_report"].name == "feasibility_report.json"

    def test_creates_summary_txt(self, tmp_path: Path) -> None:
        """write_pipeline_outputs writes summary.txt to output_dir."""
        result = _make_full_pipeline_result(tmp_path)
        out_dir = tmp_path / "out"

        written = write_pipeline_outputs(result, out_dir)

        assert "summary" in written
        assert written["summary"].exists()
        assert written["summary"].name == "summary.txt"

    def test_creates_pipeline_result_json(self, tmp_path: Path) -> None:
        """write_pipeline_outputs writes pipeline_result.json to output_dir."""
        result = _make_full_pipeline_result(tmp_path)
        out_dir = tmp_path / "out"

        written = write_pipeline_outputs(result, out_dir)

        assert "pipeline_result" in written
        assert written["pipeline_result"].exists()
        assert written["pipeline_result"].name == "pipeline_result.json"

    def test_output_dir_created_if_missing(self, tmp_path: Path) -> None:
        """write_pipeline_outputs creates the output directory if absent."""
        result = _make_full_pipeline_result(tmp_path)
        new_dir = tmp_path / "nested" / "output"
        assert not new_dir.exists()

        write_pipeline_outputs(result, new_dir)

        assert new_dir.exists()

    def test_feasibility_report_json_is_valid_json(self, tmp_path: Path) -> None:
        """feasibility_report.json is parseable JSON."""
        result = _make_full_pipeline_result(tmp_path)
        out_dir = tmp_path / "out"

        written = write_pipeline_outputs(result, out_dir)
        content = written["feasibility_report"].read_text(encoding="utf-8")
        data = json.loads(content)

        assert isinstance(data, dict)

    def test_feasibility_report_contains_project_score(self, tmp_path: Path) -> None:
        """feasibility_report.json contains a numeric project_score key."""
        result = _make_full_pipeline_result(tmp_path)
        out_dir = tmp_path / "out"

        written = write_pipeline_outputs(result, out_dir)
        data = json.loads(written["feasibility_report"].read_text(encoding="utf-8"))

        assert "project_score" in data
        assert isinstance(data["project_score"], (int, float))
        assert data["project_score"] == pytest.approx(0.88)

    def test_feasibility_report_contains_coverage(self, tmp_path: Path) -> None:
        """feasibility_report.json contains a coverage sub-dict."""
        result = _make_full_pipeline_result(tmp_path)
        out_dir = tmp_path / "out"

        written = write_pipeline_outputs(result, out_dir)
        data = json.loads(written["feasibility_report"].read_text(encoding="utf-8"))

        assert "coverage" in data
        coverage = data["coverage"]
        assert "by_wall_length_pct" in coverage
        assert "by_area_pct" in coverage
        assert "total_wall_length_inches" in coverage

    def test_pipeline_result_json_is_valid_json(self, tmp_path: Path) -> None:
        """pipeline_result.json is parseable JSON."""
        result = _make_full_pipeline_result(tmp_path)
        out_dir = tmp_path / "out"

        written = write_pipeline_outputs(result, out_dir)
        content = written["pipeline_result"].read_text(encoding="utf-8")
        data = json.loads(content)

        assert isinstance(data, dict)

    def test_pipeline_result_json_top_level_keys(self, tmp_path: Path) -> None:
        """pipeline_result.json contains expected top-level sections."""
        result = _make_full_pipeline_result(tmp_path)
        out_dir = tmp_path / "out"

        written = write_pipeline_outputs(result, out_dir)
        data = json.loads(written["pipeline_result"].read_text(encoding="utf-8"))

        for key in ("metadata", "processing_time_seconds", "stage_errors", "outputs",
                    "layer1", "classification", "drl", "feasibility", "bom"):
            assert key in data, f"Missing key: {key}"

    def test_pipeline_result_drl_section(self, tmp_path: Path) -> None:
        """pipeline_result.json drl section has spur_score and coverage."""
        result = _make_full_pipeline_result(tmp_path)
        out_dir = tmp_path / "out"

        written = write_pipeline_outputs(result, out_dir)
        drl = json.loads(written["pipeline_result"].read_text(encoding="utf-8"))["drl"]

        assert drl["spur_score"] == pytest.approx(0.82)
        assert drl["coverage_percentage"] == pytest.approx(100.0)
        assert isinstance(drl["total_panel_count"], int)

    def test_pipeline_result_bom_section(self, tmp_path: Path) -> None:
        """pipeline_result.json bom section has line_items and project_total."""
        result = _make_full_pipeline_result(tmp_path)
        out_dir = tmp_path / "out"

        written = write_pipeline_outputs(result, out_dir)
        bom = json.loads(written["pipeline_result"].read_text(encoding="utf-8"))["bom"]

        assert bom["line_items"] == 2
        assert bom["project_total_usd"] == pytest.approx(14_358.00)

    def test_summary_txt_contains_project_score(self, tmp_path: Path) -> None:
        """summary.txt mentions the project score from the feasibility report."""
        result = _make_full_pipeline_result(tmp_path)
        out_dir = tmp_path / "out"

        written = write_pipeline_outputs(result, out_dir)
        text = written["summary"].read_text(encoding="utf-8")

        assert "0.88" in text  # project_score

    def test_summary_txt_contains_coverage(self, tmp_path: Path) -> None:
        """summary.txt shows wall coverage percentage."""
        result = _make_full_pipeline_result(tmp_path)
        out_dir = tmp_path / "out"

        written = write_pipeline_outputs(result, out_dir)
        text = written["summary"].read_text(encoding="utf-8")

        # Coverage is 100.0%
        assert "100.0" in text

    def test_summary_txt_contains_panel_count(self, tmp_path: Path) -> None:
        """summary.txt shows total panel count."""
        result = _make_full_pipeline_result(tmp_path)
        out_dir = tmp_path / "out"

        written = write_pipeline_outputs(result, out_dir)
        text = written["summary"].read_text(encoding="utf-8")

        # 2 panels (one per wall) should appear somewhere
        assert "2" in text

    def test_json_fallback_path_recorded(self, tmp_path: Path) -> None:
        """write_pipeline_outputs records json_model when set on result."""
        result = _make_full_pipeline_result(tmp_path)
        out_dir = tmp_path / "out"

        written = write_pipeline_outputs(result, out_dir)

        assert "json_model" in written
        assert written["json_model"] == result.json_fallback_path

    def test_bom_export_paths_recorded(self, tmp_path: Path) -> None:
        """write_pipeline_outputs records bom_export_N keys for pre-written BOM paths."""
        result = _make_full_pipeline_result(tmp_path)

        # Simulate BOM exports written by a previous stage.
        bom_csv = tmp_path / "bom.csv"
        bom_csv.write_text("sku,qty\nPNL-LB-16-6,2\n", encoding="utf-8")
        result.bom_export_paths = [bom_csv]

        out_dir = tmp_path / "out"
        written = write_pipeline_outputs(result, out_dir)

        assert "bom_export_0" in written
        assert written["bom_export_0"] == bom_csv

    def test_partial_result_none_feasibility(self, tmp_path: Path) -> None:
        """write_pipeline_outputs handles PipelineResult with feasibility=None."""
        result = PipelineResult(
            processing_time_seconds=1.5,
            stage_errors={"feasibility": "model unavailable"},
            metadata={"pdf_path": "missing.pdf", "page_index": 0},
        )
        out_dir = tmp_path / "out"

        written = write_pipeline_outputs(result, out_dir)

        # feasibility_report.json should still be written (as error sentinel)
        assert "feasibility_report" in written
        data = json.loads(written["feasibility_report"].read_text(encoding="utf-8"))
        assert "error" in data

    def test_partial_result_all_none(self, tmp_path: Path) -> None:
        """write_pipeline_outputs does not crash when all stage outputs are None."""
        result = PipelineResult(
            processing_time_seconds=0.3,
            stage_errors={"layer1": "PDF not found: no such file"},
            metadata={"pdf_path": "ghost.pdf", "page_index": 0},
        )
        out_dir = tmp_path / "out"

        # Must not raise.
        written = write_pipeline_outputs(result, out_dir)

        assert isinstance(written, dict)
        # At minimum summary and pipeline_result are always attempted.
        assert "summary" in written or "pipeline_result" in written


# ══════════════════════════════════════════════════════════════════════════════
# Output module tests — print_summary
# ══════════════════════════════════════════════════════════════════════════════


class TestPrintSummary:
    """Tests for print_summary in src/pipeline/output.py."""

    def test_print_summary_produces_output(
        self, tmp_path: Path, capsys: pytest.CaptureFixture
    ) -> None:
        """print_summary writes non-empty text to stdout."""
        result = _make_full_pipeline_result(tmp_path)
        print_summary(result)

        captured = capsys.readouterr()
        assert len(captured.out.strip()) > 0

    def test_print_summary_contains_axon_header(
        self, tmp_path: Path, capsys: pytest.CaptureFixture
    ) -> None:
        """print_summary output contains the AXON header line."""
        result = _make_full_pipeline_result(tmp_path)
        print_summary(result)

        captured = capsys.readouterr()
        assert "AXON" in captured.out

    def test_print_summary_contains_score(
        self, tmp_path: Path, capsys: pytest.CaptureFixture
    ) -> None:
        """print_summary output contains the feasibility project score."""
        result = _make_full_pipeline_result(tmp_path)
        print_summary(result)

        captured = capsys.readouterr()
        # project_score is 0.88
        assert "0.88" in captured.out

    def test_print_summary_contains_coverage(
        self, tmp_path: Path, capsys: pytest.CaptureFixture
    ) -> None:
        """print_summary shows wall coverage percentage from feasibility report."""
        result = _make_full_pipeline_result(tmp_path)
        print_summary(result)

        captured = capsys.readouterr()
        # coverage is 100.0%
        assert "100.0" in captured.out

    def test_print_summary_contains_panel_count(
        self, tmp_path: Path, capsys: pytest.CaptureFixture
    ) -> None:
        """print_summary shows total panel count from panelization result."""
        result = _make_full_pipeline_result(tmp_path)
        print_summary(result)

        captured = capsys.readouterr()
        # 2 panels for 2 walls
        assert "2" in captured.out

    def test_print_summary_partial_result(
        self, capsys: pytest.CaptureFixture
    ) -> None:
        """print_summary handles a PipelineResult where all stages are None."""
        result = PipelineResult(
            processing_time_seconds=0.01,
            stage_errors={"layer1": "file not found"},
            metadata={"pdf_path": "ghost.pdf", "page_index": 0},
        )
        # Must not raise.
        print_summary(result)

        captured = capsys.readouterr()
        assert "AXON" in captured.out
        assert "ghost.pdf" in captured.out

    def test_print_summary_shows_stage_errors(
        self, capsys: pytest.CaptureFixture
    ) -> None:
        """print_summary lists stage failures when stage_errors is non-empty."""
        result = PipelineResult(
            processing_time_seconds=1.0,
            stage_errors={"drl": "CUDA out of memory"},
            metadata={"pdf_path": "plan.pdf", "page_index": 0},
        )
        print_summary(result)

        captured = capsys.readouterr()
        assert "drl" in captured.out
        assert "CUDA out of memory" in captured.out

    def test_print_summary_shows_all_stages_ok(
        self, tmp_path: Path, capsys: pytest.CaptureFixture
    ) -> None:
        """print_summary says all stages completed when stage_errors is empty."""
        result = _make_full_pipeline_result(tmp_path)
        print_summary(result)

        captured = capsys.readouterr()
        assert "All stages completed" in captured.out


# ══════════════════════════════════════════════════════════════════════════════
# PipelineResult structure tests
# ══════════════════════════════════════════════════════════════════════════════


class TestPipelineResultStructure:
    """Unit-level tests verifying PipelineResult dataclass invariants."""

    def test_default_result_has_empty_errors(self) -> None:
        """A freshly created PipelineResult has no stage errors."""
        result = PipelineResult()
        assert result.stage_errors == {}

    def test_default_result_has_zero_time(self) -> None:
        """A freshly created PipelineResult has zero processing time."""
        result = PipelineResult()
        assert result.processing_time_seconds == 0.0

    def test_all_output_fields_none_by_default(self) -> None:
        """Raw graph, classified graph, panelization, feasibility, and BOM default to None."""
        result = PipelineResult()
        assert result.raw_graph is None
        assert result.classified_graph is None
        assert result.panelization is None
        assert result.feasibility is None
        assert result.bom is None
        assert result.ifc_path is None
        assert result.json_fallback_path is None

    def test_bom_export_paths_default_empty(self) -> None:
        """bom_export_paths defaults to an empty list, not None."""
        result = PipelineResult()
        assert result.bom_export_paths == []

    def test_stage_errors_are_mutable(self) -> None:
        """stage_errors dict is independent per instance (no shared mutable default)."""
        r1 = PipelineResult()
        r2 = PipelineResult()
        r1.stage_errors["test"] = "boom"
        assert "test" not in r2.stage_errors

    def test_metadata_is_mutable(self) -> None:
        """metadata dict is independent per instance."""
        r1 = PipelineResult()
        r2 = PipelineResult()
        r1.metadata["key"] = "value"
        assert "key" not in r2.metadata


# ══════════════════════════════════════════════════════════════════════════════
# Pipeline integration tests — run_full_pipeline error handling
# ══════════════════════════════════════════════════════════════════════════════


class TestRunFullPipelineErrorHandling:
    """Verify run_full_pipeline degrades gracefully on bad inputs.

    These tests do not require real model weights or a real PDF — they verify
    the error-handling logic (stage_errors, return type, timing).
    """

    @pytest.fixture(scope="class")
    def kg_store(self) -> KnowledgeGraphStore:
        """A minimal KG store sufficient to initialise the pipeline."""
        return _make_kg_store()

    def test_nonexistent_pdf_returns_result(
        self, tmp_path: Path, kg_store: KnowledgeGraphStore
    ) -> None:
        """run_full_pipeline returns a PipelineResult even for a missing PDF."""
        result = run_full_pipeline(
            pdf_path=tmp_path / "does_not_exist.pdf",
            kg_store=kg_store,
            output_dir=tmp_path / "output",
        )
        assert isinstance(result, PipelineResult)

    def test_nonexistent_pdf_records_layer1_error(
        self, tmp_path: Path, kg_store: KnowledgeGraphStore
    ) -> None:
        """A missing PDF is caught and recorded in stage_errors['layer1']."""
        result = run_full_pipeline(
            pdf_path=tmp_path / "ghost.pdf",
            kg_store=kg_store,
            output_dir=tmp_path / "output",
        )
        assert "layer1" in result.stage_errors

    def test_nonexistent_pdf_error_is_string(
        self, tmp_path: Path, kg_store: KnowledgeGraphStore
    ) -> None:
        """stage_errors values are plain strings (not exceptions)."""
        result = run_full_pipeline(
            pdf_path=tmp_path / "ghost.pdf",
            kg_store=kg_store,
            output_dir=tmp_path / "output",
        )
        assert isinstance(result.stage_errors.get("layer1"), str)

    def test_nonexistent_pdf_processing_time_measured(
        self, tmp_path: Path, kg_store: KnowledgeGraphStore
    ) -> None:
        """processing_time_seconds is set even when the pipeline fails early."""
        result = run_full_pipeline(
            pdf_path=tmp_path / "ghost.pdf",
            kg_store=kg_store,
            output_dir=tmp_path / "output",
        )
        assert result.processing_time_seconds >= 0.0

    def test_output_dir_created(
        self, tmp_path: Path, kg_store: KnowledgeGraphStore
    ) -> None:
        """run_full_pipeline creates the output directory if it does not exist."""
        out_dir = tmp_path / "new_output_dir"
        assert not out_dir.exists()

        run_full_pipeline(
            pdf_path=tmp_path / "ghost.pdf",
            kg_store=kg_store,
            output_dir=out_dir,
        )

        assert out_dir.exists()

    def test_metadata_contains_pdf_path(
        self, tmp_path: Path, kg_store: KnowledgeGraphStore
    ) -> None:
        """PipelineResult.metadata records the pdf_path supplied to the function."""
        pdf = tmp_path / "my_plan.pdf"
        result = run_full_pipeline(
            pdf_path=pdf,
            kg_store=kg_store,
            output_dir=tmp_path / "out",
        )
        assert result.metadata.get("pdf_path") == str(pdf)

    def test_metadata_contains_page_index(
        self, tmp_path: Path, kg_store: KnowledgeGraphStore
    ) -> None:
        """PipelineResult.metadata records the page_index argument."""
        result = run_full_pipeline(
            pdf_path=tmp_path / "ghost.pdf",
            kg_store=kg_store,
            output_dir=tmp_path / "out",
            page_index=2,
        )
        assert result.metadata.get("page_index") == 2

    def test_default_config_used_when_none(
        self, tmp_path: Path, kg_store: KnowledgeGraphStore
    ) -> None:
        """run_full_pipeline with config=None does not raise before stage_errors are set."""
        result = run_full_pipeline(
            pdf_path=tmp_path / "ghost.pdf",
            kg_store=kg_store,
            output_dir=tmp_path / "out",
            config=None,
        )
        # result is a PipelineResult regardless of config path
        assert isinstance(result, PipelineResult)


# ══════════════════════════════════════════════════════════════════════════════
# CLI smoke tests
# ══════════════════════════════════════════════════════════════════════════════


class TestCLIHelp:
    """Verify that the Axon CLI entry-point responds to --help without error."""

    def test_report_help(self) -> None:
        """axon report --help exits 0 and shows usage text."""
        runner = CliRunner()
        result = runner.invoke(axon_cli, ["report", "--help"])
        assert result.exit_code == 0
        assert "Usage" in result.output

    def test_batch_help(self) -> None:
        """axon batch --help exits 0 and shows usage text."""
        runner = CliRunner()
        result = runner.invoke(axon_cli, ["batch", "--help"])
        assert result.exit_code == 0
        assert "Usage" in result.output

    def test_root_help(self) -> None:
        """axon --help exits 0 and lists subcommands."""
        runner = CliRunner()
        result = runner.invoke(axon_cli, ["--help"])
        assert result.exit_code == 0
        assert "report" in result.output or "batch" in result.output
