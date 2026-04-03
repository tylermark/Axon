"""Knowledge Graph query APIs — deterministic lookups for Layer 2 agents.

Provides panel/pod/machine/connection queries and fabrication constraint
validation.  Every function is a pure KG lookup — no ML, no randomness.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from src.knowledge_graph.schema import (
    Connection,
    Machine,
    Panel,
    PanelType,
    Pod,
    RelationshipType,
)

if TYPE_CHECKING:
    from src.knowledge_graph.loader import KnowledgeGraphStore

logger = logging.getLogger(__name__)

# ── Tolerance ───────────────────────────────────────────────────────────────

_LENGTH_TOLERANCE_INCHES: float = 0.25  # ¼″ tolerance for length matching


# ── Result dataclasses ──────────────────────────────────────────────────────


@dataclass
class PanelRecommendation:
    """A recommended panel configuration for a wall segment."""

    panel: Panel
    quantity: int
    cut_lengths_inches: list[float]
    requires_splice: bool
    splice_connections: list[Connection]
    total_material_cost: float
    waste_inches: float
    waste_percentage: float
    score: float  # 0-1, higher = better


@dataclass
class FabricationValidation:
    """Result of fabrication constraint validation."""

    is_valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


# ══════════════════════════════════════════════════════════════════════════════
# 1. Panel Query API (KG-007)
# ══════════════════════════════════════════════════════════════════════════════


def get_valid_panels(
    store: KnowledgeGraphStore,
    wall_length_inches: float,
    wall_type: PanelType | None = None,
    fire_rating_hours: float | None = None,
    gauge: int | None = None,
    stud_depth_inches: float | None = None,
    max_results: int | None = None,
) -> list[Panel]:
    """Find panels compatible with the given wall specifications.

    Filters by:
    - ``wall_length_inches`` must be between panel ``min_length`` and
      ``max_length``.
    - ``wall_type`` must match ``panel.panel_type`` (if specified).
    - ``fire_rating_hours``: panel must meet or exceed (if specified).
    - ``gauge``: exact match (if specified).
    - ``stud_depth_inches``: exact match (if specified).

    Returns panels sorted by best fit (prefer standard lengths, lowest
    cost).
    """
    results: list[Panel] = []
    for panel in store.panels.values():
        if wall_length_inches < panel.min_length_inches - _LENGTH_TOLERANCE_INCHES:
            continue
        if wall_length_inches > panel.max_length_inches + _LENGTH_TOLERANCE_INCHES:
            continue
        if wall_type is not None and panel.panel_type != wall_type:
            continue
        if fire_rating_hours is not None and panel.fire_rating_hours < fire_rating_hours:
            continue
        if gauge is not None and panel.gauge != gauge:
            continue
        if stud_depth_inches is not None and panel.stud_depth_inches != stud_depth_inches:
            continue
        results.append(panel)

    # Sort: best fit score descending, then by cost ascending as tiebreaker
    results.sort(
        key=lambda p: (-_score_panel_fit(p, wall_length_inches), p.unit_cost_per_foot),
    )

    if max_results is not None:
        results = results[:max_results]
    return results


def get_panels_for_wall_segment(
    store: KnowledgeGraphStore,
    wall_length_inches: float,
    wall_type: PanelType,
    fire_rating_hours: float = 0.0,
    preferred_gauge: int | None = None,
) -> list[PanelRecommendation]:
    """Recommend panel configurations for a specific wall segment.

    Returns ranked recommendations including which panel SKU to use, how
    many panels are needed, cut lengths, splice requirements, and cost.
    """
    # Don't filter by wall_length — we handle multi-panel layouts below.
    # Filter only by type, fire rating, and gauge.
    candidates = _get_candidate_panels(
        store,
        wall_type=wall_type,
        fire_rating_hours=fire_rating_hours,
        gauge=preferred_gauge,
    )

    # If preferred gauge yielded nothing, widen the search
    if not candidates and preferred_gauge is not None:
        candidates = _get_candidate_panels(
            store,
            wall_type=wall_type,
            fire_rating_hours=fire_rating_hours,
        )

    recommendations: list[PanelRecommendation] = []
    for panel in candidates:
        available_splices = _get_splice_connections(store, panel)
        quantity, cut_lengths, waste = _compute_panel_layout(
            panel, wall_length_inches, available_splices
        )
        requires_splice = quantity > 1

        # Material cost: sum of each piece's length x cost/ft, converted to feet
        material_cost = sum(cl / 12.0 for cl in cut_lengths) * panel.unit_cost_per_foot
        # Add splice hardware costs
        splice_conns: list[Connection] = []
        if requires_splice:
            splice_conns = available_splices[:1] * (quantity - 1)  # one splice per joint
            material_cost += sum(c.unit_cost for c in splice_conns)

        total_material = sum(cut_lengths)
        waste_pct = (waste / total_material * 100.0) if total_material > 0 else 0.0

        score = _score_panel_fit(panel, wall_length_inches)
        # Penalize splicing
        if requires_splice:
            score *= 0.8
        # Boost preferred gauge
        if preferred_gauge is not None and panel.gauge == preferred_gauge:
            score = min(score * 1.1, 1.0)

        recommendations.append(
            PanelRecommendation(
                panel=panel,
                quantity=quantity,
                cut_lengths_inches=cut_lengths,
                requires_splice=requires_splice,
                splice_connections=splice_conns,
                total_material_cost=round(material_cost, 2),
                waste_inches=round(waste, 2),
                waste_percentage=round(waste_pct, 2),
                score=round(score, 4),
            )
        )

    # Sort by score descending
    recommendations.sort(key=lambda r: -r.score)
    return recommendations


# ══════════════════════════════════════════════════════════════════════════════
# 2. Pod Query API (KG-008)
# ══════════════════════════════════════════════════════════════════════════════


def get_valid_pods(
    store: KnowledgeGraphStore,
    room_width_inches: float,
    room_depth_inches: float,
    room_function: str | None = None,
    required_trades: list[str] | None = None,
) -> list[Pod]:
    """Find pods that fit in the given room dimensions.

    Filters by:
    - Room must be large enough for pod + clearances (checks both
      orientations).
    - ``room_function`` must match ``pod_type`` (if specified).
    - ``required_trades`` must be a subset of ``pod.included_trades``
      (if specified).

    Returns pods sorted by best space utilization.
    """
    results: list[Pod] = []
    for pod in store.pods.values():
        if room_function is not None and pod.pod_type != room_function:
            continue
        if required_trades is not None and not set(required_trades).issubset(
            set(pod.included_trades)
        ):
            continue
        # Check fit in either orientation
        if not _pod_fits(pod, room_width_inches, room_depth_inches):
            continue
        results.append(pod)

    # Sort by space utilization (pod area / room area), descending
    room_area = room_width_inches * room_depth_inches
    results.sort(
        key=lambda p: -(p.width_inches * p.depth_inches) / room_area if room_area > 0 else 0,
    )
    return results


# ══════════════════════════════════════════════════════════════════════════════
# 3. Machine & BIM Queries
# ══════════════════════════════════════════════════════════════════════════════


def get_machine_for_panel(
    store: KnowledgeGraphStore,
    panel_sku: str,
) -> list[Machine]:
    """Get machines capable of fabricating the given panel.

    Uses the ``FABRICATED_BY`` relationship in the KG.
    """
    machine_skus = store.get_neighbors(panel_sku, relationship=RelationshipType.FABRICATED_BY)
    machines: list[Machine] = []
    for sku in machine_skus:
        machine = store.machines.get(sku)
        if machine is not None:
            machines.append(machine)
    return machines


def get_machine_for_spec(
    store: KnowledgeGraphStore,
    gauge: int,
    stud_depth_inches: float,
    length_inches: float,
) -> list[Machine]:
    """Find machines that can produce a panel with the given specs.

    Checks machine capabilities: gauge range, max length, max web depth.
    """
    results: list[Machine] = []
    for machine in store.machines.values():
        # Gauge: lower number = thicker steel.  min_gauge is the thinnest
        # (highest number), max_gauge is the thickest (lowest number).
        if gauge < machine.max_gauge or gauge > machine.min_gauge:
            continue
        if length_inches > machine.max_length_inches + _LENGTH_TOLERANCE_INCHES:
            continue
        if stud_depth_inches > machine.max_web_depth_inches:
            continue
        results.append(machine)
    return results


def get_bim_family(
    store: KnowledgeGraphStore,
    panel_type: PanelType,
    gauge: int,
    stud_depth_inches: float,
    fire_rating_hours: float = 0.0,
) -> Panel | None:
    """Find the exact panel (BIM family) matching the specifications.

    Used by the BIM Transplant agent to map 2D panel slots to 3D families.
    Returns the best matching panel or ``None`` if no exact match.
    """
    best: Panel | None = None
    best_fire_delta: float = float("inf")

    for panel in store.panels.values():
        if panel.panel_type != panel_type:
            continue
        if panel.gauge != gauge:
            continue
        if panel.stud_depth_inches != stud_depth_inches:
            continue
        if panel.fire_rating_hours < fire_rating_hours:
            continue
        # Prefer the closest fire rating that still meets the requirement
        fire_delta = panel.fire_rating_hours - fire_rating_hours
        if fire_delta < best_fire_delta:
            best = panel
            best_fire_delta = fire_delta
    return best


def get_connections_for_panel(
    store: KnowledgeGraphStore,
    panel_sku: str,
) -> list[Connection]:
    """Get all compatible connection hardware for a panel.

    Uses the ``COMPATIBLE_WITH`` relationship.
    """
    conn_skus = store.get_neighbors(panel_sku, relationship=RelationshipType.COMPATIBLE_WITH)
    connections: list[Connection] = []
    for sku in conn_skus:
        conn = store.connections.get(sku)
        if conn is not None:
            connections.append(conn)
    return connections


# ══════════════════════════════════════════════════════════════════════════════
# 4. Fabrication Constraint Validation (KG-009)
# ══════════════════════════════════════════════════════════════════════════════


def validate_panel_fabrication(
    store: KnowledgeGraphStore,
    panel_sku: str,
    required_length_inches: float,
    required_quantity: int = 1,
) -> FabricationValidation:
    """Validate that a panel can actually be fabricated.

    Checks:
    - Panel SKU exists in catalog.
    - Required length is within panel min/max.
    - At least one machine can produce it.
    - Machine can handle the gauge and stud depth.
    - Required connections are available.
    """
    errors: list[str] = []
    warnings: list[str] = []

    panel = store.panels.get(panel_sku)
    if panel is None:
        errors.append(f"Panel SKU '{panel_sku}' not found in catalog")
        return FabricationValidation(is_valid=False, errors=errors, warnings=warnings)

    # Length checks
    if required_length_inches < panel.min_length_inches - _LENGTH_TOLERANCE_INCHES:
        errors.append(
            f'Required length {required_length_inches}" is below panel minimum '
            f'{panel.min_length_inches}"'
        )
    if required_length_inches > panel.max_length_inches + _LENGTH_TOLERANCE_INCHES:
        errors.append(
            f'Required length {required_length_inches}" exceeds panel maximum '
            f'{panel.max_length_inches}"'
        )

    # Machine capability
    machines = get_machine_for_panel(store, panel_sku)
    if not machines:
        errors.append(f"No machine found that can fabricate panel '{panel_sku}'")
    else:
        # Check that at least one machine can handle the required length
        capable = [
            m
            for m in machines
            if required_length_inches <= m.max_length_inches + _LENGTH_TOLERANCE_INCHES
        ]
        if not capable:
            errors.append(
                f"No machine can produce panel '{panel_sku}' at "
                f'{required_length_inches}" length '
                f'(max: {max(m.max_length_inches for m in machines)}")'
            )

    # Connection availability
    connections = get_connections_for_panel(store, panel_sku)
    if not connections:
        warnings.append(f"No compatible connections found for panel '{panel_sku}'")

    if required_quantity > 1:
        splices = [c for c in connections if c.connection_type == "splice"]
        if not splices:
            warnings.append(
                f"Quantity {required_quantity} requested but no splice "
                f"connections available for panel '{panel_sku}'"
            )

    return FabricationValidation(
        is_valid=len(errors) == 0,
        errors=errors,
        warnings=warnings,
    )


def validate_wall_panelization(
    store: KnowledgeGraphStore,
    wall_length_inches: float,
    wall_type: PanelType,
    panel_assignments: list[tuple[str, float]],
) -> FabricationValidation:
    """Validate an entire wall's panel assignment.

    Args:
        store: The Knowledge Graph store.
        wall_length_inches: Total wall length.
        wall_type: Expected panel type for this wall.
        panel_assignments: List of ``(panel_sku, cut_length)`` tuples
            representing the panels placed along the wall.

    Checks:
    - All panel SKUs exist.
    - All cut lengths are within panel min/max.
    - Total panel lengths cover the wall (within tolerance).
    - Panel types match wall type.
    - Splice connections exist where panels meet.
    - No gaps or overlaps.
    """
    errors: list[str] = []
    warnings: list[str] = []

    if not panel_assignments:
        errors.append("No panels assigned to wall")
        return FabricationValidation(is_valid=False, errors=errors, warnings=warnings)

    total_assigned = 0.0
    prev_panel: Panel | None = None

    for i, (sku, cut_length) in enumerate(panel_assignments):
        panel = store.panels.get(sku)
        if panel is None:
            errors.append(f"Panel #{i + 1}: SKU '{sku}' not found in catalog")
            continue

        # Type match
        if panel.panel_type != wall_type:
            errors.append(
                f"Panel #{i + 1} ({sku}): type '{panel.panel_type}' does not "
                f"match wall type '{wall_type}'"
            )

        # Length within panel range
        if cut_length < panel.min_length_inches - _LENGTH_TOLERANCE_INCHES:
            errors.append(
                f'Panel #{i + 1} ({sku}): cut length {cut_length}" is below '
                f'minimum {panel.min_length_inches}"'
            )
        if cut_length > panel.max_length_inches + _LENGTH_TOLERANCE_INCHES:
            errors.append(
                f'Panel #{i + 1} ({sku}): cut length {cut_length}" exceeds '
                f'maximum {panel.max_length_inches}"'
            )

        # Splice check at joints between consecutive panels
        if prev_panel is not None:
            _check_splice_available(store, prev_panel, panel, i, warnings)

        total_assigned += cut_length
        prev_panel = panel

    # Coverage check — allow up to 1" for min-length trim waste
    coverage_diff = abs(total_assigned - wall_length_inches)
    if coverage_diff > 1.0:
        if total_assigned < wall_length_inches:
            errors.append(
                f'Panels cover {total_assigned}" but wall is '
                f'{wall_length_inches}" — gap of {wall_length_inches - total_assigned}"'
            )
        else:
            errors.append(
                f'Panels cover {total_assigned}" but wall is '
                f'{wall_length_inches}" — overlap of {total_assigned - wall_length_inches}"'
            )

    return FabricationValidation(
        is_valid=len(errors) == 0,
        errors=errors,
        warnings=warnings,
    )


def get_fabrication_limits(
    store: KnowledgeGraphStore,
    gauge: int | None = None,
) -> dict[str, float]:
    """Get aggregate fabrication limits across all machines.

    Returns dict with:
    - ``max_length_inches``: longest panel any machine can produce.
    - ``max_web_depth_inches``: deepest stud profile.
    - ``min_gauge`` / ``max_gauge``: gauge range (lower number = thicker).
    - ``max_coil_width_inches``: widest coil any machine accepts.

    If ``gauge`` is specified, filters to machines supporting that gauge.
    """
    machines = list(store.machines.values())
    if gauge is not None:
        machines = [m for m in machines if m.max_gauge <= gauge <= m.min_gauge]

    if not machines:
        return {
            "max_length_inches": 0.0,
            "max_web_depth_inches": 0.0,
            "min_gauge": 0,
            "max_gauge": 0,
            "max_coil_width_inches": 0.0,
        }

    # Gauge numbering is inverted: lower number = thicker steel.
    # Machine.max_gauge = thickest (lowest number), min_gauge = thinnest (highest).
    # Aggregate: thickest across all machines = min(max_gauge), thinnest = max(min_gauge).
    return {
        "max_length_inches": max(m.max_length_inches for m in machines),
        "max_web_depth_inches": max(m.max_web_depth_inches for m in machines),
        "min_gauge": max(m.min_gauge for m in machines),
        "max_gauge": min(m.max_gauge for m in machines),
        "max_coil_width_inches": max(m.coil_width_range_inches[1] for m in machines),
    }


# ══════════════════════════════════════════════════════════════════════════════
# 5. Scoring Helpers
# ══════════════════════════════════════════════════════════════════════════════


def _score_panel_fit(panel: Panel, wall_length_inches: float) -> float:
    """Score how well a panel fits a wall segment (0-1).

    Prefers:
    - Standard lengths (no cutting needed) → higher score
    - Minimal waste → higher score
    - Single panel (no splicing) → higher score
    - Lower cost → slightly higher score
    """
    # Can a single panel cover the wall?
    if wall_length_inches <= panel.max_length_inches:
        # Waste ratio — how much of max capacity is unused
        utilization = wall_length_inches / panel.max_length_inches
        # Base score: high when utilization is high (less waste)
        base = 0.6 + 0.4 * utilization
    else:
        # Multi-panel: penalize more panels needed
        panels_needed = math.ceil(wall_length_inches / panel.max_length_inches)
        base = max(0.1, 0.5 / panels_needed)

    # Cost factor: normalize against a reasonable range ($3-$25/ft)
    cost_factor = 1.0 - min(panel.unit_cost_per_foot / 30.0, 0.9)
    # Blend: 85% fit, 15% cost
    return round(min(0.85 * base + 0.15 * cost_factor, 1.0), 4)


def _compute_panel_layout(
    panel: Panel,
    wall_length_inches: float,
    available_splices: list[Connection],
) -> tuple[int, list[float], float]:
    """Compute how to cover a wall with copies of a given panel.

    Returns:
        ``(quantity, cut_lengths, waste_inches)``

    Strategy:
    - If ``wall_length <= panel.max_length``: one panel, cut to size.
    - If ``wall_length > panel.max_length``: multiple panels with splicing.
    - Minimize waste by distributing length evenly.
    """
    if wall_length_inches <= panel.max_length_inches:
        # Single panel, cut to exact length
        cut_length = max(wall_length_inches, panel.min_length_inches)
        waste = cut_length - wall_length_inches
        return 1, [round(cut_length, 4)], round(waste, 4)

    if not available_splices:
        # No splicing available — use max-length panels and one short closer
        full_panels = int(wall_length_inches // panel.max_length_inches)
        remainder = wall_length_inches - full_panels * panel.max_length_inches
        cut_lengths = [panel.max_length_inches] * full_panels
        if remainder >= panel.min_length_inches:
            cut_lengths.append(round(remainder, 4))
        elif remainder > 0:
            # Remainder too short for a panel — redistribute
            total_panels = full_panels + 1
            even_length = wall_length_inches / total_panels
            if even_length >= panel.min_length_inches:
                cut_lengths = [round(even_length, 4)] * total_panels
            else:
                # Fallback: use max-length panels, accept waste on last
                cut_lengths.append(round(panel.min_length_inches, 4))
        quantity = len(cut_lengths)
        total_material = sum(cut_lengths)
        waste = total_material - wall_length_inches
        return quantity, cut_lengths, round(max(waste, 0.0), 4)

    # With splicing: distribute evenly across minimum number of panels
    panels_needed = math.ceil(wall_length_inches / panel.max_length_inches)
    even_length = wall_length_inches / panels_needed

    if even_length < panel.min_length_inches:
        # Even distribution is below minimum — use fewer longer panels
        even_length = panel.min_length_inches
        panels_needed = math.ceil(wall_length_inches / even_length)

    cut_lengths = [round(even_length, 4)] * panels_needed
    # Adjust last panel to absorb rounding
    assigned = sum(cut_lengths[:-1])
    cut_lengths[-1] = round(wall_length_inches - assigned, 4)

    # Clamp last panel to min
    if cut_lengths[-1] < panel.min_length_inches:
        cut_lengths[-1] = panel.min_length_inches

    total_material = sum(cut_lengths)
    waste = total_material - wall_length_inches
    return panels_needed, cut_lengths, round(max(waste, 0.0), 4)


# ── Internal helpers ────────────────────────────────────────────────────────


def _pod_fits(
    pod: Pod,
    room_width_inches: float,
    room_depth_inches: float,
) -> bool:
    """Check if a pod fits in a room in either orientation."""
    # Normal orientation
    if (
        room_width_inches >= pod.min_room_width_inches
        and room_depth_inches >= pod.min_room_depth_inches
    ):
        return True
    # Rotated 90 degrees
    return (
        room_width_inches >= pod.min_room_depth_inches
        and room_depth_inches >= pod.min_room_width_inches
    )


def _get_splice_connections(
    store: KnowledgeGraphStore,
    panel: Panel,
) -> list[Connection]:
    """Get splice connections compatible with a panel."""
    connections = get_connections_for_panel(store, panel.sku)
    return [c for c in connections if c.connection_type == "splice"]


def _get_candidate_panels(
    store: KnowledgeGraphStore,
    wall_type: PanelType,
    fire_rating_hours: float = 0.0,
    gauge: int | None = None,
) -> list[Panel]:
    """Get panels matching type/fire/gauge without length filtering.

    Used by ``get_panels_for_wall_segment`` which handles multi-panel
    layouts internally.
    """
    results: list[Panel] = []
    for panel in store.panels.values():
        if panel.panel_type != wall_type:
            continue
        if panel.fire_rating_hours < fire_rating_hours:
            continue
        if gauge is not None and panel.gauge != gauge:
            continue
        results.append(panel)
    return results


def _check_splice_available(
    store: KnowledgeGraphStore,
    prev_panel: Panel,
    curr_panel: Panel,
    panel_index: int,
    warnings: list[str],
) -> None:
    """Check that splice hardware exists for a joint between two panels."""
    prev_splices = {c.sku for c in _get_splice_connections(store, prev_panel)}
    curr_splices = {c.sku for c in _get_splice_connections(store, curr_panel)}
    common = prev_splices & curr_splices
    if not common:
        warnings.append(
            f"No common splice connection between panel #{panel_index} "
            f"({prev_panel.sku}) and panel #{panel_index + 1} ({curr_panel.sku})"
        )
