"""Interface contract: BIM Transplant Agent → Pipeline Output.

Defines the IFCExportResult dataclass -- the output of the BIM Transplant
Agent (Phase 10, tasks BT-001 through BT-006) and input to the final pipeline
output (Phase 12).

The BIM Transplant Agent receives a PanelizationResult (containing PanelMap,
PlacementMap, ClassifiedWallGraph, and summary statistics) together with BIM
family data from the Knowledge Graph and produces:

    - BIM family matching: each 2D panel slot is mapped to a high-LOD 3D
      BIM family by panel type, gauge, length, and fire rating.
    - 3D wall assemblies: per-wall compositions of matched BIM families,
      including panel seams, splice hardware, and product SKUs.
    - Opening attachments: IfcRelVoidsElement associations linking openings
      (doors, windows) to their host wall assemblies.
    - Room assignments: IfcSpace entities with IfcRelSpaceBoundary
      relationships and pod placement data.
    - IFC serialization: IFC-SPF output per ISO 16739-1:2024, validated
      for import into Revit 2024+ and ArchiCAD 27+.

Consumers:
    - Pipeline CLI (Phase 12) -- presents the IFC file path to the user
    - External BIM software (Revit 2024+, ArchiCAD 27+) -- imports the IFC

Reference: AGENTS.md SS BIM Transplant Agent, TASKS.md SS Phase 10.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from docs.interfaces.drl_output import PanelizationResult


# ══════════════════════════════════════════════════════════════════════════════
# 1. Enums
# ══════════════════════════════════════════════════════════════════════════════


class IFCSchemaVersion(str, Enum):
    """Supported IFC schema versions for export."""

    IFC4 = "IFC4"
    """IFC4 (ISO 16739:2013), widely supported by current BIM tools."""

    IFC4X3 = "IFC4X3"
    """IFC 4.3 (ISO 16739-1:2024), latest standard with infrastructure
    extensions.  Target version for Axon."""


class LODLevel(str, Enum):
    """Level of Development for BIM family geometry.

    Follows the AIA / BIMForum LOD Specification.
    """

    LOD_200 = "LOD_200"
    """Generic placeholder geometry (approximate size and shape)."""

    LOD_300 = "LOD_300"
    """Accurate geometry with real dimensions, suitable for coordination."""

    LOD_350 = "LOD_350"
    """LOD 300 plus connection points, supports, and interfaces with
    adjacent elements.  Standard target for Axon panel exports."""

    LOD_400 = "LOD_400"
    """Fabrication-level detail including exact member profiles, fastener
    locations, and shop-drawing-ready geometry."""


class ValidationStatus(str, Enum):
    """Result of validating the IFC file against a target BIM application."""

    PASSED = "passed"
    """IFC file imported successfully with all entities intact."""

    WARNINGS = "warnings"
    """IFC file imported but with non-critical warnings (e.g., missing
    property sets, unsupported geometric representations)."""

    FAILED = "failed"
    """IFC file failed to import or produced critical errors."""

    NOT_TESTED = "not_tested"
    """Validation was not performed for this target application."""


class OpeningHostRelation(str, Enum):
    """IFC relationship type for attaching openings to host walls."""

    REL_VOIDS_ELEMENT = "IfcRelVoidsElement"
    """Standard IFC relationship: the opening element voids the host wall."""


# ══════════════════════════════════════════════════════════════════════════════
# 2. BIM Family Matching
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class MaterialLayer:
    """A single material layer in a wall panel's cross-section.

    Ordered from exterior face to interior face.  Used to populate
    IfcMaterialLayerSet on the exported IfcWallStandardCase.
    """

    material_name: str
    """Material identifier (e.g., 'CFS 54mil Stud', 'Gypsum Board',
    'Batt Insulation R-19')."""

    thickness_inches: float
    """Layer thickness in inches."""

    is_structural: bool = False
    """True if this layer is the structural CFS framing layer."""

    material_category: str = ""
    """IFC material category string (e.g., 'Steel', 'Gypsum',
    'Insulation').  Maps to IfcMaterialLayer.Category."""


@dataclass
class BIMFamilyMatch:
    """Mapping from a 2D panel slot to its 3D BIM family.

    The BIM Transplant Agent queries the Knowledge Graph for each panel
    SKU in the PanelMap and resolves it to a high-LOD 3D BIM family
    with full material layer composition.

    One BIMFamilyMatch per PanelAssignment in the upstream PanelMap.
    """

    panel_sku: str
    """Panel SKU from the PanelMap (matches ``PanelAssignment.panel_sku``
    and ``Panel.sku`` in the KG schema)."""

    revit_family_name: str
    """Revit family name for this panel type.

    Example: 'Capsule_CFS_Wall_LB_54mil_362'.  Used for Revit-side
    family mapping when the IFC is imported.
    """

    revit_type_name: str = ""
    """Revit type name within the family.

    Example: 'Type A - Fire Rated 1HR'.  Together with
    ``revit_family_name``, uniquely identifies the Revit family type.
    """

    lod: LODLevel = LODLevel.LOD_350
    """Level of Development of the exported 3D geometry.

    Default LOD 350 includes connection points and interfaces.
    """

    material_layers: list[MaterialLayer] = field(default_factory=list)
    """Ordered material layers from exterior to interior face.

    Populated from the KG panel spec (sheathing, insulation, stud depth)
    and compliance rules (fire-rated assemblies, vapor barriers).
    """

    total_thickness_inches: float = 0.0
    """Sum of all material layer thicknesses.

    Must match the wall thickness from the upstream graph within
    fabrication tolerance.
    """

    ifc_entity_type: str = "IfcWallStandardCase"
    """IFC entity type this panel maps to.

    Almost always IfcWallStandardCase.  May be IfcWall for non-standard
    geometries that cannot use SweptSolid representation.
    """

    ifc_predefined_type: str = ""
    """IFC PredefinedType attribute (e.g., 'STANDARD', 'PARTITIONING',
    'SHEAR').  Maps the panel's structural classification to IFC semantics.
    """

    fire_rating_hours: float = 0.0
    """Fire resistance rating in hours, from the KG panel spec.

    Serialized as IfcPropertySingleValue 'FireRating' in the IFC
    property set.
    """

    gauge: int = 0
    """Steel gauge of the CFS framing, from the KG panel spec.

    Serialized as an IFC property for fabrication reference.
    """

    kg_query_confidence: float = 1.0
    """Confidence that the KG lookup returned the correct BIM family,
    in [0, 1].

    Less than 1.0 when the panel spec required fuzzy matching
    (e.g., non-standard length outside catalog increments).
    """


# ══════════════════════════════════════════════════════════════════════════════
# 3. Wall Assembly — 3D Representation
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class PanelRepresentation:
    """3D representation of a single panel within a wall assembly.

    Corresponds to one PanelAssignment from the DRL output, resolved
    to its BIM family and positioned along the wall axis.
    """

    panel_index: int
    """Zero-based index within the wall's panel list.

    Matches ``PanelAssignment.panel_index`` from the DRL output.
    """

    bim_family: BIMFamilyMatch
    """Resolved BIM family for this panel."""

    position_along_wall_inches: float
    """Start offset of this panel from the wall's start node, in inches.

    Matches ``PanelAssignment.position_along_wall``.
    """

    cut_length_inches: float
    """Actual cut length of this panel piece, in inches.

    Matches ``PanelAssignment.cut_length_inches``.
    """

    ifc_global_id: str = ""
    """IFC GlobalId (GUID) assigned to this panel's IfcWallStandardCase
    entity in the exported file.

    Populated during serialization; empty before export.
    """


@dataclass
class SeamLocation:
    """Location of a seam (splice joint) between adjacent panels.

    Seams carry the splice connection hardware SKU and the exact
    position along the wall where the joint occurs.
    """

    seam_index: int
    """Zero-based index of this seam within the wall.

    Seam i is between panel_index i and panel_index i+1.
    """

    position_along_wall_inches: float
    """Position of the seam from the wall's start node, in inches.

    Equals the end of panel i (or equivalently, the start of panel i+1).
    """

    splice_connection_sku: str = ""
    """SKU of the splice connection hardware at this joint.

    Matches entries in ``WallPanelization.splice_connection_skus``
    and ``Connection.sku`` in the KG schema.
    """

    ifc_global_id: str = ""
    """IFC GlobalId for the splice connection entity, if serialized
    as a discrete IFC element.  Empty if splices are represented as
    properties rather than standalone entities."""


@dataclass
class WallAssembly:
    """3D wall representation built from matched BIM families.

    Aggregates all panel representations, seam locations, and splice
    hardware for a single wall segment.  One WallAssembly per
    WallPanelization in the upstream DRL output.
    """

    edge_id: int
    """Edge index matching ``WallSegment.edge_id`` and
    ``WallPanelization.edge_id`` in upstream contracts."""

    panels: list[PanelRepresentation]
    """Ordered list of 3D panel representations along this wall.

    Ordered from start node to end node, matching the upstream
    PanelAssignment order.
    """

    seams: list[SeamLocation]
    """Seam (splice joint) locations between adjacent panels.

    Length is ``len(panels) - 1`` when splicing is required, empty
    for single-panel walls.
    """

    wall_length_inches: float = 0.0
    """Total wall length in inches, carried from the upstream
    WallPanelization for convenience."""

    ifc_global_id: str = ""
    """IFC GlobalId for the wall-level IfcWallStandardCase entity
    that aggregates the panel representations.

    In most cases, each wall is serialized as a single
    IfcWallStandardCase with an IfcMaterialLayerSetUsage.
    Individual panel subdivisions are represented via
    IfcRelAggregates if LOD 400 is requested.
    """

    is_serialized: bool = True
    """True if this wall was successfully serialized to IFC.

    False when the upstream WallPanelization has ``is_panelizable=False``
    or BIM family matching failed.  Non-serialized walls are noted in
    the export statistics but omitted from the IFC file.
    """

    serialization_note: str = ""
    """Explanation when ``is_serialized`` is False (e.g., 'No matching
    BIM family for panel SKU XYZ', 'Wall not panelizable')."""


# ══════════════════════════════════════════════════════════════════════════════
# 4. Opening Attachments
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class OpeningAttachment:
    """IfcRelVoidsElement data attaching an opening to its host wall.

    Each opening (door, window, portal) detected in the upstream
    FinalizedGraph is associated with its host wall assembly via an
    IfcRelVoidsElement relationship.  The opening element creates a
    void in the host wall's solid geometry.
    """

    opening_index: int
    """Index into the upstream FinalizedGraph.openings list."""

    host_wall_edge_id: int
    """Edge ID of the wall that this opening belongs to.

    Matches ``Opening.wall_edge_id`` in the upstream graph.
    """

    opening_type: str
    """Opening classification (e.g., 'door', 'window', 'portal').

    Matches ``Opening.opening_type.value`` from the upstream graph.
    """

    relation_type: OpeningHostRelation = OpeningHostRelation.REL_VOIDS_ELEMENT
    """IFC relationship type used to attach this opening.

    Always IfcRelVoidsElement per IFC4 / IFC4X3 schema.
    """

    # ── Void geometry ────────────────────────────────────────────────────

    void_width_inches: float = 0.0
    """Width of the void in the host wall, in inches.

    Sourced from ``Opening.width`` converted via the graph's
    scale_factor.
    """

    void_height_inches: float = 0.0
    """Height of the void, in inches.

    Sourced from ``Opening.height``.
    """

    sill_height_inches: float = 0.0
    """Sill height above finished floor level, in inches.

    Zero for doors and portals.  Sourced from ``Opening.sill_height``.
    """

    position_along_wall: float = 0.0
    """Normalized position [0, 1] of the opening center along the
    host wall.

    Sourced from ``Opening.position_along_wall``.
    """

    # ── IFC identifiers ──────────────────────────────────────────────────

    opening_ifc_global_id: str = ""
    """IFC GlobalId assigned to the IfcOpeningElement."""

    relation_ifc_global_id: str = ""
    """IFC GlobalId assigned to the IfcRelVoidsElement relationship."""

    has_filling: bool = False
    """True if the opening has an IfcRelFillsElement (e.g., an
    IfcDoor or IfcWindow entity filling the void).

    False when only the void is modeled (no door/window family placed).
    """

    filling_ifc_global_id: str = ""
    """IFC GlobalId of the filling element (IfcDoor / IfcWindow), if any.

    Empty when ``has_filling`` is False.
    """


# ══════════════════════════════════════════════════════════════════════════════
# 5. Room Assignments
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class RoomAssignment:
    """IfcSpace mapping for a single room.

    Each interior room from the upstream FinalizedGraph is serialized
    as an IfcSpace entity with IfcRelSpaceBoundary relationships to
    its bounding walls.  If the DRL Agent placed a pod in this room,
    the pod is serialized as an IfcProduct contained within the space.
    """

    room_id: int
    """Room identifier matching ``Room.room_id`` in the upstream graph."""

    label: str = ""
    """Semantic label (e.g., 'Bathroom', 'Kitchen', 'Bedroom').

    Serialized as the IfcSpace.LongName attribute.
    """

    boundary_edge_ids: list[int] = field(default_factory=list)
    """Ordered list of wall edge IDs forming the room boundary.

    Each edge ID is linked via IfcRelSpaceBoundary to this room's
    IfcSpace entity.
    """

    area_sqft: float = 0.0
    """Room area in square feet, serialized as an IfcQuantityArea
    property on the IfcSpace."""

    # ── Pod placement ────────────────────────────────────────────────────

    has_pod: bool = False
    """True if the DRL Agent placed a pod in this room."""

    pod_sku: str = ""
    """SKU of the placed pod, matching ``ProductPlacement.pod_sku``
    and ``Pod.sku`` in the KG schema.

    Empty when ``has_pod`` is False.
    """

    pod_ifc_entity_type: str = "IfcBuildingElementProxy"
    """IFC entity type for the placed pod.

    IfcBuildingElementProxy is used when no more specific IFC entity
    type applies (e.g., IfcSanitaryTerminal for bathroom pods could
    be used in LOD 400 exports).
    """

    pod_ifc_global_id: str = ""
    """IFC GlobalId assigned to the pod entity.

    Empty when ``has_pod`` is False.
    """

    # ── IFC identifiers ──────────────────────────────────────────────────

    space_ifc_global_id: str = ""
    """IFC GlobalId assigned to the IfcSpace entity."""

    boundary_relation_ifc_global_ids: list[str] = field(default_factory=list)
    """IFC GlobalIds of the IfcRelSpaceBoundary relationships.

    One per boundary_edge_id, same order.
    """


# ══════════════════════════════════════════════════════════════════════════════
# 6. Validation Results
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class ApplicationValidation:
    """Validation result for a single target BIM application.

    Records whether the exported IFC file imports cleanly into the
    target application, along with any warnings or errors encountered.
    """

    application_name: str
    """Target application name (e.g., 'Revit 2024', 'ArchiCAD 27')."""

    application_version: str = ""
    """Specific version tested (e.g., '2024.2', '27.1.0')."""

    status: ValidationStatus = ValidationStatus.NOT_TESTED
    """Import validation result."""

    entity_count_imported: int = 0
    """Number of IFC entities successfully imported.

    Should match ``ExportStatistics.total_ifc_entities`` when
    ``status`` is PASSED.
    """

    warnings: list[str] = field(default_factory=list)
    """Non-critical warning messages from the import process.

    Empty when ``status`` is PASSED or NOT_TESTED.
    """

    errors: list[str] = field(default_factory=list)
    """Critical error messages from the import process.

    Empty unless ``status`` is FAILED.
    """

    notes: str = ""
    """Additional notes about the validation run (e.g., 'tested via
    Revit IFC import API', 'manual import test pending')."""


# ══════════════════════════════════════════════════════════════════════════════
# 7. Export Statistics
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class ExportStatistics:
    """Summary statistics for the IFC export.

    Provides counts of all serialized entities for verification and
    dashboard display.
    """

    total_ifc_entities: int = 0
    """Total number of IFC entities in the exported file.

    Includes walls, openings, spaces, relationships, property sets,
    material layers, and all supporting entities.
    """

    walls_serialized: int = 0
    """Number of IfcWallStandardCase entities written.

    Should match the count of WallAssemblies with
    ``is_serialized=True``.
    """

    walls_skipped: int = 0
    """Number of walls not serialized (not panelizable or no BIM
    family match).

    ``walls_serialized + walls_skipped`` should equal the total
    wall count in the upstream graph.
    """

    openings_attached: int = 0
    """Number of IfcOpeningElement entities written with
    IfcRelVoidsElement relationships."""

    fillings_placed: int = 0
    """Number of IfcDoor / IfcWindow filling entities placed
    (subset of openings_attached where ``has_filling`` is True)."""

    rooms_assigned: int = 0
    """Number of IfcSpace entities written with
    IfcRelSpaceBoundary relationships."""

    pods_placed: int = 0
    """Number of pod IfcProduct entities written within IfcSpace
    containers."""

    space_boundaries_written: int = 0
    """Total number of IfcRelSpaceBoundary relationships written
    across all rooms."""

    property_sets_written: int = 0
    """Total number of IfcPropertySet entities written (fire ratings,
    gauges, SKUs, material specs, etc.)."""

    material_layer_sets_written: int = 0
    """Number of IfcMaterialLayerSet entities written (one per
    unique wall assembly type)."""


# ══════════════════════════════════════════════════════════════════════════════
# 8. Export Metadata
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class IFCExportMetadata:
    """Metadata about the IFC export run.

    Tracks versioning, timestamps, and provenance for reproducibility
    and audit trails.
    """

    generated_at: str = ""
    """ISO 8601 timestamp of when the IFC file was generated.

    Example: '2026-04-02T14:30:00Z'.
    """

    generator_version: str = ""
    """Version identifier of the BIM Transplant Agent that produced
    this output."""

    kg_version: str = ""
    """Version of the Knowledge Graph catalog used for BIM family
    matching.

    Matches ``KnowledgeGraph.version`` from ``schema.py``.
    """

    policy_version: str = ""
    """Version of the DRL policy checkpoint that produced the upstream
    PanelizationResult.

    Carried forward from ``PanelizationResult.policy_version`` for
    full provenance tracking.
    """

    ifc_processor: str = ""
    """Name and version of the IFC serialization library used.

    Example: 'IfcOpenShell 0.8.0'.
    """

    source_pdf_path: str = ""
    """Path to the original PDF floor plan that started the pipeline.

    Carried from the upstream FinalizedGraph for traceability.
    """

    notes: str = ""
    """Free-form notes about the export run."""


# ══════════════════════════════════════════════════════════════════════════════
# 9. Top-Level IFCExportResult
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class IFCExportResult:
    """Complete output of the BIM Transplant Agent (Phase 10).

    Wraps all BIM family matching, 3D assembly, opening attachment,
    room assignment, and IFC serialization results into a single
    structure consumed by the final pipeline output (Phase 12).

    This result answers: "Here is the manufacture-ready BIM model with
    real product SKUs, validated for import into Revit and ArchiCAD."

    **Consumers:**
    - Pipeline CLI (Phase 12) -- presents the IFC file path, glTF
      visualization, validation status, and export statistics to the user.
    - External BIM software (Revit 2024+, ArchiCAD 27+) -- imports the
      IFC-SPF file directly.

    **Must NOT contain:** Cost estimates or feasibility judgments.
    Costs are the BOM Agent's domain (see ``bill_of_materials.py``).
    Feasibility is the Feasibility Agent's domain (see
    ``feasibility_report.py``).

    Reference: AGENTS.md SS BIM Transplant Agent, TASKS.md SS Phase 10.
    """

    source: PanelizationResult
    """The PanelizationResult that was transformed into BIM.

    Retained so downstream consumers can trace wall assemblies and
    room assignments back to the DRL panel/pod decisions and the
    original ClassifiedWallGraph.
    """

    # ── 3D Assemblies ────────────────────────────────────────────────────

    wall_assemblies: list[WallAssembly]
    """3D wall assemblies built from matched BIM families.

    One per wall segment in the upstream graph.  Walls that could not
    be serialized have ``is_serialized=False`` with a
    ``serialization_note``.

    Ordered by ``edge_id`` to match the upstream graph's wall_segments.
    """

    opening_attachments: list[OpeningAttachment]
    """Opening-to-wall attachments via IfcRelVoidsElement.

    One per opening in the upstream FinalizedGraph.openings list.
    Ordered by ``opening_index``.
    """

    room_assignments: list[RoomAssignment]
    """IfcSpace assignments with boundary relationships and pod data.

    One per interior room in the upstream FinalizedGraph.rooms list
    (excluding the exterior boundary room).  Ordered by ``room_id``.
    """

    # ── Output file paths ────────────────────────────────────────────────

    ifc_file_path: str = ""
    """Absolute path to the exported IFC-SPF file.

    The file is in STEP Physical File (SPF) format per
    ISO 10303-21, containing IFC entities per the selected
    ``ifc_schema_version``.
    """

    gltf_file_path: str = ""
    """Optional absolute path to a glTF 2.0 file for web visualization.

    Empty string if glTF export was not requested or not generated.
    The glTF is a lightweight 3D representation derived from the IFC
    model, suitable for browser-based viewers.
    """

    # ── IFC schema ───────────────────────────────────────────────────────

    ifc_schema_version: IFCSchemaVersion = IFCSchemaVersion.IFC4X3
    """IFC schema version used for serialization.

    Default is IFC4X3 (ISO 16739-1:2024), the target standard for
    Axon.  May fall back to IFC4 for broader tool compatibility.
    """

    # ── Validation ───────────────────────────────────────────────────────

    validations: list[ApplicationValidation] = field(default_factory=list)
    """Validation results for each target BIM application.

    Standard pipeline produces two entries: Revit 2024+ and
    ArchiCAD 27+.  Additional targets may be added by configuration.
    """

    # ── Statistics ───────────────────────────────────────────────────────

    statistics: ExportStatistics = field(default_factory=ExportStatistics)
    """Summary counts of all serialized IFC entities."""

    # ── Metadata ─────────────────────────────────────────────────────────

    export_metadata: IFCExportMetadata = field(
        default_factory=IFCExportMetadata
    )
    """Export provenance: timestamps, KG version, policy version,
    IFC processor, and source PDF path."""

    metadata: dict[str, object] = field(default_factory=dict)
    """Additional metadata (processing time, GPU used, etc.)."""
