"""Unit tests for src/knowledge_graph/loader.py — KG loading and graph construction."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest
from pydantic import ValidationError

from src.knowledge_graph.loader import KnowledgeGraphStore, load_knowledge_graph
from src.knowledge_graph.schema import (
    Connection,
    Machine,
    Panel,
    Pod,
    RelationshipType,
)

if TYPE_CHECKING:
    from pathlib import Path


# ---------------------------------------------------------------------------
# Module-scoped fixture: load KG once for all tests in this file
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def kg_store() -> KnowledgeGraphStore:
    return load_knowledge_graph()


# ---------------------------------------------------------------------------
# Factory / load_knowledge_graph
# ---------------------------------------------------------------------------


class TestLoadKnowledgeGraph:
    def test_returns_store(self, kg_store: KnowledgeGraphStore) -> None:
        assert isinstance(kg_store, KnowledgeGraphStore)

    def test_version_populated(self, kg_store: KnowledgeGraphStore) -> None:
        assert kg_store.version
        assert isinstance(kg_store.version, str)


# ---------------------------------------------------------------------------
# Entity counts
# ---------------------------------------------------------------------------


class TestEntityCounts:
    def test_panel_count(self, kg_store: KnowledgeGraphStore) -> None:
        assert len(kg_store.panels) == 19

    def test_pod_count(self, kg_store: KnowledgeGraphStore) -> None:
        assert len(kg_store.pods) == 8

    def test_machine_count(self, kg_store: KnowledgeGraphStore) -> None:
        assert len(kg_store.machines) == 3

    def test_connection_count(self, kg_store: KnowledgeGraphStore) -> None:
        assert len(kg_store.connections) == 12


# ---------------------------------------------------------------------------
# Graph structure
# ---------------------------------------------------------------------------


class TestGraphStructure:
    def test_node_count(self, kg_store: KnowledgeGraphStore) -> None:
        # 19 panels + 8 pods + 3 machines + 12 connections = 42 (no compliance rules)
        assert kg_store.graph.number_of_nodes() == 42

    def test_edge_count_reasonable(self, kg_store: KnowledgeGraphStore) -> None:
        # FABRICATED_BY + COMPATIBLE_WITH + ATTACHES_TO edges — should be 250+
        assert kg_store.graph.number_of_edges() >= 250

    def test_all_panels_are_nodes(self, kg_store: KnowledgeGraphStore) -> None:
        for sku in kg_store.panels:
            assert sku in kg_store.graph

    def test_all_machines_are_nodes(self, kg_store: KnowledgeGraphStore) -> None:
        for sku in kg_store.machines:
            assert sku in kg_store.graph

    def test_all_connections_are_nodes(self, kg_store: KnowledgeGraphStore) -> None:
        for sku in kg_store.connections:
            assert sku in kg_store.graph

    def test_all_pods_are_nodes(self, kg_store: KnowledgeGraphStore) -> None:
        for sku in kg_store.pods:
            assert sku in kg_store.graph


# ---------------------------------------------------------------------------
# get_entity lookups
# ---------------------------------------------------------------------------


class TestGetEntity:
    def test_get_panel(self, kg_store: KnowledgeGraphStore) -> None:
        entity = kg_store.get_entity("CAP-PNL-LB-16GA-600D-16OC-96H")
        assert isinstance(entity, Panel)
        assert entity.sku == "CAP-PNL-LB-16GA-600D-16OC-96H"

    def test_get_pod(self, kg_store: KnowledgeGraphStore) -> None:
        entity = kg_store.get_entity("CAP-POD-BATH-STD")
        assert isinstance(entity, Pod)
        assert entity.pod_type == "bathroom"

    def test_get_machine(self, kg_store: KnowledgeGraphStore) -> None:
        entity = kg_store.get_entity("CAP-MCH-HW3500")
        assert isinstance(entity, Machine)
        assert entity.name == "Howick FRAMA 3500"

    def test_get_connection(self, kg_store: KnowledgeGraphStore) -> None:
        entity = kg_store.get_entity("CAP-CON-SPL-16GA")
        assert isinstance(entity, Connection)
        assert entity.connection_type == "splice"

    def test_unknown_sku_returns_none(self, kg_store: KnowledgeGraphStore) -> None:
        assert kg_store.get_entity("DOES-NOT-EXIST") is None


# ---------------------------------------------------------------------------
# get_neighbors
# ---------------------------------------------------------------------------


class TestGetNeighbors:
    def test_fabricated_by_returns_machines(self, kg_store: KnowledgeGraphStore) -> None:
        neighbors = kg_store.get_neighbors(
            "CAP-PNL-LB-16GA-600D-16OC-96H",
            relationship=RelationshipType.FABRICATED_BY,
        )
        assert "CAP-MCH-HW2500" in neighbors
        assert "CAP-MCH-HW3500" in neighbors

    def test_compatible_with_returns_connections(self, kg_store: KnowledgeGraphStore) -> None:
        neighbors = kg_store.get_neighbors(
            "CAP-PNL-LB-16GA-600D-16OC-96H",
            relationship=RelationshipType.COMPATIBLE_WITH,
        )
        assert "CAP-CON-SPL-16GA" in neighbors
        assert "CAP-CON-CLP-600" in neighbors

    def test_no_filter_returns_all(self, kg_store: KnowledgeGraphStore) -> None:
        all_neighbors = kg_store.get_neighbors("CAP-PNL-LB-16GA-600D-16OC-96H")
        filtered_fab = kg_store.get_neighbors(
            "CAP-PNL-LB-16GA-600D-16OC-96H",
            relationship=RelationshipType.FABRICATED_BY,
        )
        filtered_compat = kg_store.get_neighbors(
            "CAP-PNL-LB-16GA-600D-16OC-96H",
            relationship=RelationshipType.COMPATIBLE_WITH,
        )
        # Unfiltered should include at least everything from the two filtered sets
        assert len(all_neighbors) >= len(filtered_fab) + len(filtered_compat)

    def test_unknown_sku_returns_empty(self, kg_store: KnowledgeGraphStore) -> None:
        assert kg_store.get_neighbors("DOES-NOT-EXIST") == []

    def test_14ga_panel_only_hw3500(self, kg_store: KnowledgeGraphStore) -> None:
        """14ga shear panel is only fabricated by HW3500."""
        neighbors = kg_store.get_neighbors(
            "CAP-PNL-SH-14GA-600D-16OC-96H",
            relationship=RelationshipType.FABRICATED_BY,
        )
        assert neighbors == ["CAP-MCH-HW3500"]


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------


class TestProperties:
    def test_panels_dict(self, kg_store: KnowledgeGraphStore) -> None:
        panels = kg_store.panels
        assert isinstance(panels, dict)
        assert all(isinstance(v, Panel) for v in panels.values())

    def test_pods_dict(self, kg_store: KnowledgeGraphStore) -> None:
        pods = kg_store.pods
        assert isinstance(pods, dict)
        assert all(isinstance(v, Pod) for v in pods.values())

    def test_machines_dict(self, kg_store: KnowledgeGraphStore) -> None:
        machines = kg_store.machines
        assert isinstance(machines, dict)
        assert all(isinstance(v, Machine) for v in machines.values())

    def test_connections_dict(self, kg_store: KnowledgeGraphStore) -> None:
        connections = kg_store.connections
        assert isinstance(connections, dict)
        assert all(isinstance(v, Connection) for v in connections.values())


# ---------------------------------------------------------------------------
# validate()
# ---------------------------------------------------------------------------


class TestValidation:
    def test_validate_no_errors(self, kg_store: KnowledgeGraphStore) -> None:
        issues = kg_store.validate()
        errors = [i for i in issues if not i.startswith("WARN:")]
        assert errors == [], f"Unexpected validation errors: {errors}"


# ---------------------------------------------------------------------------
# Error cases — loading from bad data
# ---------------------------------------------------------------------------


class TestLoadErrors:
    def test_missing_directory_raises(self, tmp_path: Path) -> None:
        nonexistent = tmp_path / "does_not_exist"
        with pytest.raises(FileNotFoundError):
            load_knowledge_graph(nonexistent)

    def test_empty_directory_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_knowledge_graph(tmp_path)

    def test_invalid_json_raises(self, tmp_path: Path) -> None:
        for name in ("panels.json", "pods.json", "machines.json", "connections.json"):
            (tmp_path / name).write_text("not valid json")
        with pytest.raises(json.JSONDecodeError):
            load_knowledge_graph(tmp_path)

    def test_malformed_panel_data_raises(self, tmp_path: Path) -> None:
        (tmp_path / "panels.json").write_text('[{"sku": "BAD"}]')
        (tmp_path / "pods.json").write_text("[]")
        (tmp_path / "machines.json").write_text("[]")
        (tmp_path / "connections.json").write_text("[]")
        with pytest.raises(ValidationError):
            load_knowledge_graph(tmp_path)
