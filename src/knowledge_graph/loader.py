"""Knowledge Graph loader — JSON product data to queryable NetworkX graph.

Ingests Capsule Manufacturing's JSON catalog files, validates against
Pydantic schema, and builds an in-memory directed graph with typed
nodes and relationship edges.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import networkx as nx

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

if TYPE_CHECKING:
    from os import PathLike

logger = logging.getLogger(__name__)


class KnowledgeGraphStore:
    """In-memory queryable Knowledge Graph backed by NetworkX.

    Loads Capsule's product catalog from JSON data files, validates against
    the Pydantic schema, and constructs a NetworkX directed graph with
    typed nodes and edges.
    """

    def __init__(self) -> None:
        self._graph: nx.DiGraph = nx.DiGraph()
        self._panels: dict[str, Panel] = {}
        self._pods: dict[str, Pod] = {}
        self._machines: dict[str, Machine] = {}
        self._connections: dict[str, Connection] = {}
        self._compliance_rules: list[ComplianceRule] = []
        self._version: str = ""
        self._last_updated: str = ""

    # ── Loading ──────────────────────────────────────────────────────────

    def load_from_directory(self, data_dir: Path | str | None = None) -> None:
        """Load all JSON data files from a directory.

        Args:
            data_dir: Path to directory containing panels.json, pods.json,
                machines.json, connections.json. Defaults to
                ``src/knowledge_graph/data/``.
        """
        if data_dir is None:
            data_dir = Path(__file__).parent / "data"
        data_dir = Path(data_dir)

        panels_raw = _read_json(data_dir / "panels.json")
        pods_raw = _read_json(data_dir / "pods.json")
        machines_raw = _read_json(data_dir / "machines.json")
        connections_raw = _read_json(data_dir / "connections.json")

        panels = [Panel.model_validate(p) for p in panels_raw]
        pods = [Pod.model_validate(p) for p in pods_raw]
        machines = [Machine.model_validate(m) for m in machines_raw]
        connections = [Connection.model_validate(c) for c in connections_raw]

        # Compliance rules are embedded in a full KnowledgeGraph JSON if present,
        # otherwise loaded from individual files above.
        compliance_rules: list[ComplianceRule] = []
        compliance_path = data_dir / "compliance_rules.json"
        if compliance_path.exists():
            rules_raw = _read_json(compliance_path)
            compliance_rules = [ComplianceRule.model_validate(r) for r in rules_raw]

        kg = KnowledgeGraph(
            version="1.0.0",
            last_updated="",
            panels=panels,
            pods=pods,
            machines=machines,
            connections=connections,
            compliance_rules=compliance_rules,
        )
        self.load_from_knowledge_graph(kg)

    def load_from_knowledge_graph(self, kg: KnowledgeGraph) -> None:
        """Load from a pre-validated KnowledgeGraph Pydantic model.

        Args:
            kg: A fully populated ``KnowledgeGraph`` instance.
        """
        self._version = kg.version
        self._last_updated = kg.last_updated
        self._graph.clear()

        self._load_panels(kg.panels)
        self._load_pods(kg.pods)
        self._load_machines(kg.machines)
        self._load_connections(kg.connections)
        self._load_compliance_rules(kg.compliance_rules)
        self._build_relationships()

        logger.info(
            "KG loaded: %d nodes, %d edges (v%s)",
            self._graph.number_of_nodes(),
            self._graph.number_of_edges(),
            self._version,
        )

    # ── Private loaders ──────────────────────────────────────────────────

    def _load_panels(self, panels: list[Panel]) -> None:
        """Add panel nodes to graph."""
        for panel in panels:
            self._panels[panel.sku] = panel
            self._graph.add_node(
                panel.sku,
                node_type="panel",
                entity=panel,
            )

    def _load_pods(self, pods: list[Pod]) -> None:
        """Add pod nodes to graph."""
        for pod in pods:
            self._pods[pod.sku] = pod
            self._graph.add_node(
                pod.sku,
                node_type="pod",
                entity=pod,
            )

    def _load_machines(self, machines: list[Machine]) -> None:
        """Add machine nodes to graph."""
        for machine in machines:
            self._machines[machine.sku] = machine
            self._graph.add_node(
                machine.sku,
                node_type="machine",
                entity=machine,
            )

    def _load_connections(self, connections: list[Connection]) -> None:
        """Add connection nodes to graph."""
        for conn in connections:
            self._connections[conn.sku] = conn
            self._graph.add_node(
                conn.sku,
                node_type="connection",
                entity=conn,
            )

    def _load_compliance_rules(self, rules: list[ComplianceRule]) -> None:
        """Add compliance rule nodes to graph."""
        self._compliance_rules = list(rules)
        for rule in rules:
            node_id = _compliance_rule_id(rule)
            self._graph.add_node(
                node_id,
                node_type="compliance_rule",
                entity=rule,
            )

    def _build_relationships(self) -> None:
        """Build edges between nodes based on entity cross-references.

        Creates edges for:
        - FABRICATED_BY: Panel → Machine (from panel.fabricated_by SKUs)
        - COMPATIBLE_WITH: Panel → Connection (from panel.compatible_connections)
        - ATTACHES_TO: Pod → Panel (where panel.panel_type in pod.compatible_panel_types)
        - RATED_FOR: Panel → ComplianceRule (where panel.panel_type in rule.applies_to)
        """
        for panel in self._panels.values():
            # FABRICATED_BY
            for machine_sku in panel.fabricated_by:
                if machine_sku in self._machines:
                    self._graph.add_edge(
                        panel.sku,
                        machine_sku,
                        relationship=RelationshipType.FABRICATED_BY,
                    )

            # COMPATIBLE_WITH
            for conn_sku in panel.compatible_connections:
                if conn_sku in self._connections:
                    self._graph.add_edge(
                        panel.sku,
                        conn_sku,
                        relationship=RelationshipType.COMPATIBLE_WITH,
                    )

        # ATTACHES_TO: Pod → Panel (matching panel_type)
        for pod in self._pods.values():
            for panel in self._panels.values():
                if panel.panel_type in pod.compatible_panel_types:
                    self._graph.add_edge(
                        pod.sku,
                        panel.sku,
                        relationship=RelationshipType.ATTACHES_TO,
                    )

        # RATED_FOR: Panel → ComplianceRule
        for rule in self._compliance_rules:
            rule_id = _compliance_rule_id(rule)
            for panel in self._panels.values():
                if panel.panel_type in rule.applies_to:
                    self._graph.add_edge(
                        panel.sku,
                        rule_id,
                        relationship=RelationshipType.RATED_FOR,
                    )

    # ── Properties ───────────────────────────────────────────────────────

    @property
    def panels(self) -> dict[str, Panel]:
        """All panels indexed by SKU."""
        return self._panels

    @property
    def pods(self) -> dict[str, Pod]:
        """All pods indexed by SKU."""
        return self._pods

    @property
    def machines(self) -> dict[str, Machine]:
        """All machines indexed by SKU."""
        return self._machines

    @property
    def connections(self) -> dict[str, Connection]:
        """All connections indexed by SKU."""
        return self._connections

    @property
    def compliance_rules(self) -> list[ComplianceRule]:
        """All compliance rules."""
        return self._compliance_rules

    @property
    def graph(self) -> nx.DiGraph:
        """The underlying NetworkX graph."""
        return self._graph

    @property
    def version(self) -> str:
        """Catalog version string."""
        return self._version

    # ── Lookups ──────────────────────────────────────────────────────────

    def get_entity(self, sku: str) -> Panel | Pod | Machine | Connection | None:
        """Look up any entity by SKU.

        Args:
            sku: The entity's unique SKU identifier.

        Returns:
            The Pydantic model instance, or ``None`` if not found.
        """
        if sku in self._panels:
            return self._panels[sku]
        if sku in self._pods:
            return self._pods[sku]
        if sku in self._machines:
            return self._machines[sku]
        if sku in self._connections:
            return self._connections[sku]
        return None

    def get_neighbors(self, sku: str, relationship: RelationshipType | None = None) -> list[str]:
        """Get SKUs of entities connected to the given SKU.

        Considers both outgoing and incoming edges so that relationships
        are traversable in either direction.

        Args:
            sku: Source node SKU.
            relationship: If provided, only return neighbors connected
                by this relationship type.

        Returns:
            List of neighbor node IDs (SKUs or compliance rule IDs).
        """
        if sku not in self._graph:
            return []

        neighbors: list[str] = []

        # Outgoing edges
        for _, target, data in self._graph.out_edges(sku, data=True):
            if relationship is None or data.get("relationship") == relationship:
                neighbors.append(target)

        # Incoming edges
        for source, _, data in self._graph.in_edges(sku, data=True):
            if relationship is None or data.get("relationship") == relationship:
                neighbors.append(source)

        return neighbors

    # ── Validation ───────────────────────────────────────────────────────

    def validate(self) -> list[str]:
        """Run integrity checks on the loaded KG.

        Checks:
        - All cross-referenced SKUs exist (no dangling references)
        - All panels have at least one fabricating machine
        - All machines have at least one panel they can produce
        - No duplicate SKUs across entity types
        - Compliance rules reference valid panel types

        Returns:
            List of warning/error strings. Empty list means valid.
        """
        errors: list[str] = []

        # Check panel cross-references
        for sku, panel in self._panels.items():
            for machine_sku in panel.fabricated_by:
                if machine_sku not in self._machines:
                    errors.append(
                        f"Panel {sku}: fabricated_by references unknown machine {machine_sku}"
                    )
            if not panel.fabricated_by:
                errors.append(f"Panel {sku}: has no fabricating machines")

            for conn_sku in panel.compatible_connections:
                if conn_sku not in self._connections:
                    errors.append(
                        f"Panel {sku}: compatible_connections references "
                        f"unknown connection {conn_sku}"
                    )

        # Check all machines produce at least one panel (warning, not error)
        machines_used: set[str] = set()
        for panel in self._panels.values():
            machines_used.update(panel.fabricated_by)
        for machine_sku in self._machines:
            if machine_sku not in machines_used:
                errors.append(f"WARN: Machine {machine_sku}: not referenced by any panel")

        # Check for duplicate SKUs across entity types
        all_skus: list[str] = (
            list(self._panels.keys())
            + list(self._pods.keys())
            + list(self._machines.keys())
            + list(self._connections.keys())
        )
        seen: set[str] = set()
        for sku in all_skus:
            if sku in seen:
                errors.append(f"Duplicate SKU across entity types: {sku}")
            seen.add(sku)

        # Validate compliance rules reference valid panel types
        valid_panel_types = {pt.value for pt in PanelType}
        for rule in self._compliance_rules:
            for pt in rule.applies_to:
                if pt.value not in valid_panel_types:
                    errors.append(
                        f"ComplianceRule {rule.code}:{rule.section}: "
                        f"applies_to references unknown panel type {pt}"
                    )

        return errors


# ── Helpers ──────────────────────────────────────────────────────────────


def _read_json(path: Path | PathLike[str]) -> list[dict]:
    """Read and parse a JSON file.

    Args:
        path: Path to the JSON file.

    Returns:
        Parsed JSON content (expected to be a list of dicts).

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    with open(path) as f:
        return json.load(f)


def _compliance_rule_id(rule: ComplianceRule) -> str:
    """Generate a stable node ID for a compliance rule."""
    return f"{rule.code}:{rule.section}"


# ── Factory ──────────────────────────────────────────────────────────────


def load_knowledge_graph(data_dir: Path | str | None = None) -> KnowledgeGraphStore:
    """Convenience: load KG from default or specified data directory.

    Loads, validates, and returns a ready-to-query KnowledgeGraphStore.
    Raises ValueError if validation finds critical errors.

    Args:
        data_dir: Optional path to data directory. Defaults to
            ``src/knowledge_graph/data/``.

    Returns:
        A fully loaded and validated ``KnowledgeGraphStore``.

    Raises:
        ValueError: If validation detects integrity errors.
    """
    store = KnowledgeGraphStore()
    store.load_from_directory(data_dir)

    issues = store.validate()
    warnings = [i for i in issues if i.startswith("WARN:")]
    errors = [i for i in issues if not i.startswith("WARN:")]

    for warn in warnings:
        logger.warning("KG validation: %s", warn)
    for err in errors:
        logger.error("KG validation: %s", err)

    if errors:
        raise ValueError(
            f"Knowledge Graph validation failed with {len(errors)} error(s):\n"
            + "\n".join(f"  - {e}" for e in errors)
        )

    return store
