"""Knowledge Graph schema for Capsule Manufacturing's CFS product catalog.

Defines all entity types (Panel, Pod, Machine, Connection, ComplianceRule)
and relationship types used by the Axon Knowledge Graph — the single source
of truth for all Layer 2 agents.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel


class PanelType(StrEnum):
    """Wall panel classification types."""

    LOAD_BEARING = "load_bearing"
    PARTITION = "partition"
    SHEAR = "shear"
    FIRE_RATED = "fire_rated"
    ENVELOPE = "envelope"


class RelationshipType(StrEnum):
    """Edge types in the Knowledge Graph."""

    FABRICATED_BY = "fabricated_by"
    COMPATIBLE_WITH = "compatible_with"
    REQUIRES = "requires"
    RATED_FOR = "rated_for"
    ATTACHES_TO = "attaches_to"
    PRODUCED_AT = "produced_at"


class Panel(BaseModel):
    """A CFS wall panel product in Capsule's catalog."""

    sku: str
    name: str
    panel_type: PanelType
    gauge: int
    stud_depth_inches: float
    stud_spacing_inches: float
    min_length_inches: float
    max_length_inches: float
    height_inches: float
    fire_rating_hours: float
    load_capacity_plf: float
    sheathing_type: str | None
    sheathing_thickness_inches: float | None
    insulation_type: str | None
    insulation_r_value: float | None
    weight_per_foot_lbs: float
    unit_cost_per_foot: float
    compatible_connections: list[str]
    fabricated_by: list[str]


class Pod(BaseModel):
    """A prefabricated pod assembly (bathroom, kitchen, MEP, etc.)."""

    sku: str
    name: str
    pod_type: str
    width_inches: float
    depth_inches: float
    height_inches: float
    min_room_width_inches: float
    min_room_depth_inches: float
    clearance_inches: float
    included_trades: list[str]
    connection_type: str
    weight_lbs: float
    unit_cost: float
    lead_time_days: int
    compatible_panel_types: list[PanelType]


class Machine(BaseModel):
    """A fabrication machine at Capsule's facility."""

    sku: str
    name: str
    machine_type: str
    max_gauge: int
    min_gauge: int
    max_length_inches: float
    max_web_depth_inches: float
    max_flange_width_inches: float
    coil_width_range_inches: tuple[float, float]
    speed_feet_per_minute: float
    tolerance_inches: float


class Connection(BaseModel):
    """Connection hardware (clips, splices, fasteners, bridging, blocking)."""

    sku: str
    name: str
    connection_type: str
    compatible_gauges: list[int]
    compatible_stud_depths: list[float]
    load_rating_lbs: float | None
    fire_rated: bool
    unit_cost: float
    units_per: str


class ComplianceRule(BaseModel):
    """Building code and standards compliance rules."""

    code: str
    section: str
    description: str
    applies_to: list[PanelType]
    constraint_type: str
    constraint_value: float | str


class KnowledgeGraph(BaseModel):
    """The complete Capsule Manufacturing product Knowledge Graph."""

    version: str
    last_updated: str
    panels: list[Panel]
    pods: list[Pod]
    machines: list[Machine]
    connections: list[Connection]
    compliance_rules: list[ComplianceRule]
