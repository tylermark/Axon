"""Unit tests for src/knowledge_graph/schema.py — Pydantic entity models."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.knowledge_graph.schema import (
    ComplianceRule,
    Connection,
    KnowledgeGraph,
    Machine,
    Panel,
    PanelType,
    Pod,
    RelationshipType,
)

# ---------------------------------------------------------------------------
# PanelType enum
# ---------------------------------------------------------------------------


class TestPanelType:
    def test_has_five_types(self) -> None:
        assert len(PanelType) == 5

    def test_values(self) -> None:
        expected = {"load_bearing", "partition", "shear", "fire_rated", "envelope"}
        assert {pt.value for pt in PanelType} == expected

    def test_is_strenum(self) -> None:
        assert str(PanelType.LOAD_BEARING) == "load_bearing"


# ---------------------------------------------------------------------------
# RelationshipType enum
# ---------------------------------------------------------------------------


class TestRelationshipType:
    def test_has_six_types(self) -> None:
        assert len(RelationshipType) == 6

    def test_values(self) -> None:
        expected = {
            "fabricated_by",
            "compatible_with",
            "requires",
            "rated_for",
            "attaches_to",
            "produced_at",
        }
        assert {rt.value for rt in RelationshipType} == expected


# ---------------------------------------------------------------------------
# Panel model
# ---------------------------------------------------------------------------


def _valid_panel_data() -> dict:
    """Minimal valid Panel data dict."""
    return {
        "sku": "TEST-PNL-001",
        "name": "Test Panel",
        "panel_type": "load_bearing",
        "gauge": 16,
        "stud_depth_inches": 6.0,
        "stud_spacing_inches": 16.0,
        "min_length_inches": 24.0,
        "max_length_inches": 300.0,
        "height_inches": 96.0,
        "fire_rating_hours": 0.0,
        "load_capacity_plf": 2100.0,
        "sheathing_type": None,
        "sheathing_thickness_inches": None,
        "insulation_type": None,
        "insulation_r_value": None,
        "weight_per_foot_lbs": 7.2,
        "unit_cost_per_foot": 14.50,
        "compatible_connections": ["CAP-CON-SPL-16GA"],
        "fabricated_by": ["CAP-MCH-HW2500"],
    }


class TestPanelModel:
    def test_valid_panel(self) -> None:
        panel = Panel(**_valid_panel_data())
        assert panel.sku == "TEST-PNL-001"
        assert panel.panel_type == PanelType.LOAD_BEARING
        assert panel.gauge == 16

    def test_missing_sku_raises(self) -> None:
        data = _valid_panel_data()
        del data["sku"]
        with pytest.raises(ValidationError):
            Panel(**data)

    def test_missing_name_raises(self) -> None:
        data = _valid_panel_data()
        del data["name"]
        with pytest.raises(ValidationError):
            Panel(**data)

    def test_invalid_panel_type_raises(self) -> None:
        data = _valid_panel_data()
        data["panel_type"] = "nonexistent_type"
        with pytest.raises(ValidationError):
            Panel(**data)

    def test_optional_sheathing_fields(self) -> None:
        panel = Panel(**_valid_panel_data())
        assert panel.sheathing_type is None
        assert panel.sheathing_thickness_inches is None
        assert panel.insulation_type is None
        assert panel.insulation_r_value is None

    def test_compatible_connections_list(self) -> None:
        panel = Panel(**_valid_panel_data())
        assert isinstance(panel.compatible_connections, list)
        assert "CAP-CON-SPL-16GA" in panel.compatible_connections

    def test_fabricated_by_list(self) -> None:
        panel = Panel(**_valid_panel_data())
        assert isinstance(panel.fabricated_by, list)
        assert "CAP-MCH-HW2500" in panel.fabricated_by


# ---------------------------------------------------------------------------
# Pod model
# ---------------------------------------------------------------------------


def _valid_pod_data() -> dict:
    """Minimal valid Pod data dict."""
    return {
        "sku": "TEST-POD-001",
        "name": "Test Pod",
        "pod_type": "bathroom",
        "width_inches": 60.0,
        "depth_inches": 96.0,
        "height_inches": 96.0,
        "min_room_width_inches": 66.0,
        "min_room_depth_inches": 102.0,
        "clearance_inches": 3.0,
        "included_trades": ["plumbing", "electrical"],
        "connection_type": "clip_angle",
        "weight_lbs": 1800.0,
        "unit_cost": 12500.00,
        "lead_time_days": 21,
        "compatible_panel_types": ["load_bearing", "partition"],
    }


class TestPodModel:
    def test_valid_pod(self) -> None:
        pod = Pod(**_valid_pod_data())
        assert pod.sku == "TEST-POD-001"
        assert pod.pod_type == "bathroom"

    def test_missing_sku_raises(self) -> None:
        data = _valid_pod_data()
        del data["sku"]
        with pytest.raises(ValidationError):
            Pod(**data)

    def test_compatible_panel_types_parsed(self) -> None:
        pod = Pod(**_valid_pod_data())
        assert PanelType.LOAD_BEARING in pod.compatible_panel_types
        assert PanelType.PARTITION in pod.compatible_panel_types

    def test_included_trades(self) -> None:
        pod = Pod(**_valid_pod_data())
        assert "plumbing" in pod.included_trades


# ---------------------------------------------------------------------------
# Machine model
# ---------------------------------------------------------------------------


def _valid_machine_data() -> dict:
    """Minimal valid Machine data dict."""
    return {
        "sku": "TEST-MCH-001",
        "name": "Test Machine",
        "machine_type": "roll_former",
        "max_gauge": 16,
        "min_gauge": 25,
        "max_length_inches": 300.0,
        "max_web_depth_inches": 6.0,
        "max_flange_width_inches": 2.5,
        "coil_width_range_inches": (6.0, 16.0),
        "speed_feet_per_minute": 45.0,
        "tolerance_inches": 0.02,
    }


class TestMachineModel:
    def test_valid_machine(self) -> None:
        machine = Machine(**_valid_machine_data())
        assert machine.sku == "TEST-MCH-001"
        assert machine.machine_type == "roll_former"

    def test_missing_sku_raises(self) -> None:
        data = _valid_machine_data()
        del data["sku"]
        with pytest.raises(ValidationError):
            Machine(**data)

    def test_coil_width_range_is_tuple(self) -> None:
        machine = Machine(**_valid_machine_data())
        assert machine.coil_width_range_inches == (6.0, 16.0)

    def test_gauge_range_inverted_convention(self) -> None:
        """max_gauge (thickest, lower number) < min_gauge (thinnest, higher number)."""
        machine = Machine(**_valid_machine_data())
        assert machine.max_gauge < machine.min_gauge


# ---------------------------------------------------------------------------
# Connection model
# ---------------------------------------------------------------------------


def _valid_connection_data() -> dict:
    """Minimal valid Connection data dict."""
    return {
        "sku": "TEST-CON-001",
        "name": "Test Splice",
        "connection_type": "splice",
        "compatible_gauges": [16, 18],
        "compatible_stud_depths": [3.5, 6.0],
        "load_rating_lbs": 4500.0,
        "fire_rated": False,
        "unit_cost": 3.25,
        "units_per": "each",
    }


class TestConnectionModel:
    def test_valid_connection(self) -> None:
        conn = Connection(**_valid_connection_data())
        assert conn.sku == "TEST-CON-001"
        assert conn.connection_type == "splice"

    def test_missing_sku_raises(self) -> None:
        data = _valid_connection_data()
        del data["sku"]
        with pytest.raises(ValidationError):
            Connection(**data)

    def test_nullable_load_rating(self) -> None:
        data = _valid_connection_data()
        data["load_rating_lbs"] = None
        conn = Connection(**data)
        assert conn.load_rating_lbs is None

    def test_compatible_gauges_list(self) -> None:
        conn = Connection(**_valid_connection_data())
        assert 16 in conn.compatible_gauges
        assert 18 in conn.compatible_gauges


# ---------------------------------------------------------------------------
# ComplianceRule model
# ---------------------------------------------------------------------------


class TestComplianceRuleModel:
    def test_valid_compliance_rule(self) -> None:
        rule = ComplianceRule(
            code="IBC",
            section="2211.4",
            description="Test rule",
            applies_to=[PanelType.LOAD_BEARING, PanelType.SHEAR],
            constraint_type="min_gauge",
            constraint_value=16,
        )
        assert rule.code == "IBC"
        assert PanelType.LOAD_BEARING in rule.applies_to

    def test_constraint_value_can_be_string(self) -> None:
        rule = ComplianceRule(
            code="IBC",
            section="2211.1",
            description="Test rule",
            applies_to=[PanelType.FIRE_RATED],
            constraint_type="sheathing_required",
            constraint_value="gypsum",
        )
        assert rule.constraint_value == "gypsum"


# ---------------------------------------------------------------------------
# KnowledgeGraph model
# ---------------------------------------------------------------------------


class TestKnowledgeGraphModel:
    def test_valid_knowledge_graph(self) -> None:
        kg = KnowledgeGraph(
            version="1.0.0",
            last_updated="2026-04-01",
            panels=[Panel(**_valid_panel_data())],
            pods=[Pod(**_valid_pod_data())],
            machines=[Machine(**_valid_machine_data())],
            connections=[Connection(**_valid_connection_data())],
            compliance_rules=[],
        )
        assert kg.version == "1.0.0"
        assert len(kg.panels) == 1
        assert len(kg.pods) == 1
        assert len(kg.machines) == 1
        assert len(kg.connections) == 1

    def test_empty_entity_lists(self) -> None:
        kg = KnowledgeGraph(
            version="0.0.1",
            last_updated="",
            panels=[],
            pods=[],
            machines=[],
            connections=[],
            compliance_rules=[],
        )
        assert len(kg.panels) == 0

    def test_missing_version_raises(self) -> None:
        with pytest.raises(ValidationError):
            KnowledgeGraph(
                last_updated="",
                panels=[],
                pods=[],
                machines=[],
                connections=[],
                compliance_rules=[],
            )
