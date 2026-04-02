"""Axon Knowledge Graph — Capsule Manufacturing product catalog.

The Knowledge Graph is the single source of truth for all product data.
All Layer 2 agents query it. No probabilistic guessing — if the KG says
a panel doesn't exist, it doesn't get placed.
"""

from __future__ import annotations

from pathlib import Path

from src.knowledge_graph.loader import KnowledgeGraphStore, load_knowledge_graph
from src.knowledge_graph.query import (
    FabricationValidation,
    PanelRecommendation,
    get_bim_family,
    get_connections_for_panel,
    get_fabrication_limits,
    get_machine_for_panel,
    get_machine_for_spec,
    get_panels_for_wall_segment,
    get_valid_panels,
    get_valid_pods,
    validate_panel_fabrication,
    validate_wall_panelization,
)
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

DATA_DIR = Path(__file__).parent / "data"

__all__ = [
    "DATA_DIR",
    "ComplianceRule",
    "Connection",
    "FabricationValidation",
    "KnowledgeGraph",
    "KnowledgeGraphStore",
    "Machine",
    "Panel",
    "PanelRecommendation",
    "PanelType",
    "Pod",
    "RelationshipType",
    "get_bim_family",
    "get_connections_for_panel",
    "get_fabrication_limits",
    "get_machine_for_panel",
    "get_machine_for_spec",
    "get_panels_for_wall_segment",
    "get_valid_panels",
    "get_valid_pods",
    "load_knowledge_graph",
    "validate_panel_fabrication",
    "validate_wall_panelization",
]
