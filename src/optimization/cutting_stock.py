"""Stage 7a: Per-wall 1D cutting stock solver.

For each classified wall segment, finds the minimum-waste partition into
panels from the KG catalog. Uses pattern enumeration over feasible piece
counts for optimal solutions, with First Fit Decreasing (FFD) as a fast
fallback.

The solver handles:
- Walls with openings (split into sub-segments via DRL constraints module)
- Corner thickness deductions at junctions
- KG fabrication constraints (min/max length, gauge, fire rating, machine limits)
- Multiple candidate solutions per wall for the global CP-SAT solver to choose from
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from src.drl.constraints import (
    WallSubSegment,
    compute_junction_map,
    compute_wall_sub_segments,
)
from src.drl.state import fire_rating_to_hours, wall_type_to_panel_type
from src.knowledge_graph.query import (
    _get_candidate_panels,
    _get_splice_connections,
)
from src.knowledge_graph.schema import Panel, PanelType

if TYPE_CHECKING:
    from docs.interfaces.classified_wall_graph import ClassifiedWallGraph
    from docs.interfaces.graph_to_serializer import WallSegment
    from src.knowledge_graph.loader import KnowledgeGraphStore

logger = logging.getLogger(__name__)


# ── Result dataclasses ────────────────────────────────────────────────────────


@dataclass
class CuttingStockSolution:
    """A candidate panel layout for a single wall.

    Multiple solutions may be generated per wall (one per panel type),
    giving the global CP-SAT solver choices for cross-wall optimization.
    """

    wall_edge_id: int
    """Edge ID of the wall this solution covers."""

    panel_sku: str
    """SKU of the panel type used in this solution."""

    panel: Panel
    """The Panel entity from the KG."""

    assignments: list[tuple[str, float]]
    """Ordered list of (sku, cut_length_inches) covering the wall."""

    waste_inches: float
    """Total material waste in inches."""

    waste_percentage: float
    """Waste as a fraction of total material (0-1)."""

    total_cost: float
    """Material + splice hardware cost in USD."""

    requires_splice: bool
    """True if wall needs more than one panel."""

    num_pieces: int
    """Number of panel pieces in this solution."""

    gauge: int
    """Steel gauge of the panels."""

    stud_depth_inches: float
    """Stud depth of the panels."""

    score: float = 0.0
    """Composite quality score (lower waste + fewer pieces = higher)."""


@dataclass
class WallCuttingResult:
    """All candidate solutions for a single wall."""

    wall_edge_id: int
    wall_length_inches: float
    effective_length_inches: float
    panel_type: PanelType
    solutions: list[CuttingStockSolution] = field(default_factory=list)
    is_panelizable: bool = True
    rejection_reason: str = ""


# ── Core solver ───────────────────────────────────────────────────────────────


def solve_wall_cutting_stock(
    wall: WallSegment,
    panel_type: PanelType,
    fire_rating_hours: float,
    effective_length_inches: float,
    store: KnowledgeGraphStore,
    max_solutions: int = 5,
) -> list[CuttingStockSolution]:
    """Solve the 1D cutting stock problem for a single wall.

    Enumerates feasible cutting patterns for each candidate panel type,
    solves the LP relaxation to find the minimum-waste assignment, then
    rounds to integers.

    Args:
        wall: The wall segment.
        panel_type: Required panel type from classifier.
        fire_rating_hours: Required fire rating.
        effective_length_inches: Panelizable length after corner deductions.
        store: Knowledge Graph store.
        max_solutions: Maximum number of candidate solutions to return.

    Returns:
        Ranked list of CuttingStockSolution, best first.
    """
    if effective_length_inches <= 0:
        return []

    candidates = _get_candidate_panels(
        store,
        wall_type=panel_type,
        fire_rating_hours=fire_rating_hours,
    )

    if not candidates:
        logger.warning(
            "No candidate panels for wall %d (type=%s, fire=%.1fh)",
            wall.edge_id, panel_type, fire_rating_hours,
        )
        return []

    solutions: list[CuttingStockSolution] = []
    for panel in candidates:
        sol = _solve_for_panel_type(
            wall.edge_id, panel, effective_length_inches, store,
        )
        if sol is not None:
            solutions.append(sol)

    # Rank by score (higher = better)
    solutions.sort(key=lambda s: -s.score)
    return solutions[:max_solutions]


def _solve_for_panel_type(
    edge_id: int,
    panel: Panel,
    length_inches: float,
    store: KnowledgeGraphStore,
) -> CuttingStockSolution | None:
    """Find the optimal cutting pattern for a single panel type.

    For typical Axon problem sizes (wall < 480", panel max < 240"),
    the number of pieces is small (1-4), so we can solve directly
    without full column generation.
    """
    if length_inches <= 0:
        return None

    # Check if wall is too short for even the minimum panel length
    if length_inches < panel.min_length_inches:
        return None

    # Single panel covers the wall
    if length_inches <= panel.max_length_inches:
        cut_length = max(length_inches, panel.min_length_inches)
        waste = cut_length - length_inches
        cost = (cut_length / 12.0) * panel.unit_cost_per_foot
        return CuttingStockSolution(
            wall_edge_id=edge_id,
            panel_sku=panel.sku,
            panel=panel,
            assignments=[(panel.sku, round(cut_length, 4))],
            waste_inches=round(waste, 4),
            waste_percentage=round(waste / cut_length, 4) if cut_length > 0 else 0.0,
            total_cost=round(cost, 2),
            requires_splice=False,
            num_pieces=1,
            gauge=panel.gauge,
            stud_depth_inches=panel.stud_depth_inches,
            score=_compute_solution_score(waste, cut_length, 1, cost),
        )

    # Multi-panel: need splicing
    splices = _get_splice_connections(store, panel)
    if not splices:
        # No splice hardware — cannot cover this wall with this panel type
        return None

    # Find optimal number of pieces via LP relaxation
    result = _enumerate_cutting_patterns(panel, length_inches)
    if result is None:
        return None

    num_pieces, cut_lengths = result
    total_material = sum(cut_lengths)
    waste = total_material - length_inches

    # Cost: material + splice hardware
    material_cost = (total_material / 12.0) * panel.unit_cost_per_foot
    splice_cost = splices[0].unit_cost * (num_pieces - 1)
    total_cost = material_cost + splice_cost

    assignments = [(panel.sku, round(cl, 4)) for cl in cut_lengths]

    return CuttingStockSolution(
        wall_edge_id=edge_id,
        panel_sku=panel.sku,
        panel=panel,
        assignments=assignments,
        waste_inches=round(waste, 4),
        waste_percentage=round(waste / total_material, 4) if total_material > 0 else 0.0,
        total_cost=round(total_cost, 2),
        requires_splice=True,
        num_pieces=num_pieces,
        gauge=panel.gauge,
        stud_depth_inches=panel.stud_depth_inches,
        score=_compute_solution_score(waste, total_material, num_pieces, total_cost),
    )


def _enumerate_cutting_patterns(
    panel: Panel,
    length_inches: float,
) -> tuple[int, list[float]] | None:
    """Find the minimum-waste cutting pattern by enumeration.

    For the typical Axon problem size (1-5 pieces per wall), enumerates
    feasible piece counts and finds the even distribution that minimizes
    waste, respecting panel min/max length constraints.

    Returns:
        (num_pieces, cut_lengths) or None if infeasible.
    """
    min_len = panel.min_length_inches
    max_len = panel.max_length_inches

    # Find minimum number of pieces needed
    min_pieces = max(1, math.ceil(length_inches / max_len))
    # Find maximum reasonable pieces (don't use more than length / min_len)
    max_pieces = math.floor(length_inches / min_len) if min_len > 0 else min_pieces

    if max_pieces < min_pieces:
        return None

    best_waste = float("inf")
    best_result: tuple[int, list[float]] | None = None

    for n in range(min_pieces, min(max_pieces + 1, min_pieces + 5)):
        # Distribute evenly
        even_length = length_inches / n

        if even_length < min_len:
            # Even distribution too short — can't use n pieces
            break
        if even_length > max_len:
            # Even distribution too long — need more pieces
            continue

        # All pieces at even_length covers exactly the wall
        cut_lengths = [round(even_length, 4)] * n
        # Adjust last piece to absorb rounding errors
        assigned = sum(cut_lengths[:-1])
        cut_lengths[-1] = round(length_inches - assigned, 4)

        # Validate last piece is within bounds
        if cut_lengths[-1] < min_len:
            cut_lengths[-1] = min_len
        elif cut_lengths[-1] > max_len:
            continue  # Can't fix this with n pieces

        total_material = sum(cut_lengths)
        waste = total_material - length_inches

        if waste < best_waste:
            best_waste = waste
            best_result = (n, cut_lengths)

    return best_result


# ── FFD heuristic fallback ────────────────────────────────────────────────────


def solve_ffd(
    panel: Panel,
    length_inches: float,
) -> tuple[int, list[float], float] | None:
    """First Fit Decreasing heuristic for 1D cutting stock.

    Simple greedy: fill max-length panels, then one shorter closer.
    Used as a fast fallback when LP is overkill.

    Returns:
        (num_pieces, cut_lengths, waste_inches) or None if infeasible.
    """
    if length_inches <= 0 or length_inches < panel.min_length_inches:
        return None

    min_len = panel.min_length_inches
    max_len = panel.max_length_inches

    if length_inches <= max_len:
        cut = max(length_inches, min_len)
        return (1, [round(cut, 4)], round(cut - length_inches, 4))

    cut_lengths: list[float] = []
    remaining = length_inches

    while remaining > max_len:
        cut_lengths.append(max_len)
        remaining -= max_len

    if remaining >= min_len:
        cut_lengths.append(round(remaining, 4))
    elif remaining > 0:
        # Remainder too short — redistribute from last full panel
        if cut_lengths:
            # Take from the last panel and create two shorter ones
            last = cut_lengths.pop()
            total = last + remaining
            half = total / 2.0
            if half >= min_len:
                cut_lengths.extend([round(half, 4), round(total - round(half, 4), 4)])
            else:
                # Give up and use min_length (accepting waste)
                cut_lengths.append(last)
                cut_lengths.append(min_len)

    total_material = sum(cut_lengths)
    waste = total_material - length_inches
    return (len(cut_lengths), cut_lengths, round(max(waste, 0.0), 4))


# ── Scoring ───────────────────────────────────────────────────────────────────


def _compute_solution_score(
    waste_inches: float,
    total_material: float,
    num_pieces: int,
    total_cost: float,
) -> float:
    """Score a cutting stock solution (0-1, higher = better).

    Weights:
    - 50% waste efficiency (lower waste = better)
    - 30% splice penalty (fewer pieces = better)
    - 20% cost efficiency
    """
    waste_ratio = waste_inches / total_material if total_material > 0 else 0.0
    waste_score = 1.0 - min(waste_ratio, 1.0)

    # Penalize splicing: 1 piece = 1.0, 2 = 0.7, 3 = 0.5, etc.
    splice_score = 1.0 / (1.0 + 0.3 * (num_pieces - 1))

    # Cost: normalize against $50/ft as "expensive"
    cost_per_foot = (total_cost / (total_material / 12.0)) if total_material > 0 else 0.0
    cost_score = max(0.0, 1.0 - cost_per_foot / 50.0)

    score = 0.5 * waste_score + 0.3 * splice_score + 0.2 * cost_score
    return round(min(max(score, 0.0), 1.0), 4)


# ── Corner deduction helpers ──────────────────────────────────────────────────


def _get_endpoint_deductions(
    wall: WallSegment,
    junction_map: dict,
    walls: list,
    scale: float,
) -> tuple[float, float]:
    """Return ``(start_deduction_inches, end_deduction_inches)`` for a wall.

    Mirrors the per-junction logic of ``get_corner_thickness_deduction`` from
    ``src.drl.constraints``, but returns the start and end values separately so
    callers can apply them to the correct sub-segment of a wall with openings.
    """
    to_inches = scale / 25.4 if scale != 1.0 else 1.0 / 72.0
    wall_lookup = {w.edge_id: w for w in walls}
    deductions = [0.0, 0.0]

    for i, node_id in enumerate((wall.start_node, wall.end_node)):
        junction = junction_map.get(node_id)
        if junction is None or junction.junction_type not in ("corner", "T", "cross"):
            continue
        for adj_eid in junction.wall_edge_ids:
            if adj_eid == wall.edge_id:
                continue
            adj_wall = wall_lookup.get(adj_eid)
            if adj_wall is None:
                continue
            angle_diff = abs(wall.angle - adj_wall.angle) % math.pi
            if abs(angle_diff - math.pi / 2.0) < math.radians(10.0):
                deductions[i] += adj_wall.thickness * to_inches
                break  # one deduction per junction node

    return (deductions[0], deductions[1])


# ── Top-level per-wall API ────────────────────────────────────────────────────


def solve_all_walls(
    classified_graph: ClassifiedWallGraph,
    store: KnowledgeGraphStore,
    max_solutions_per_wall: int = 5,
) -> list[WallCuttingResult]:
    """Solve cutting stock for every wall in the classified graph.

    Handles opening splitting and corner thickness deductions before
    calling the per-wall solver.

    Args:
        classified_graph: Classified wall graph from Layer 1 + classifier.
        store: Knowledge Graph store.
        max_solutions_per_wall: Max candidate solutions per wall.

    Returns:
        One WallCuttingResult per wall segment, in wall_segments order.
    """
    graph = classified_graph.graph
    walls = graph.wall_segments
    classifications = classified_graph.classifications
    openings = graph.openings
    scale = graph.scale_factor

    # Build opening lookup
    opening_map: dict[int, list] = {}
    for opening in openings:
        opening_map.setdefault(opening.wall_edge_id, []).append(opening)

    # Build junction map for corner deductions
    junction_map = compute_junction_map(walls)

    # PDF units to inches
    to_inches = scale / 25.4 if scale != 1.0 else 1.0 / 72.0

    results: list[WallCuttingResult] = []

    for i, wall in enumerate(walls):
        cls = classifications[i]
        panel_type = wall_type_to_panel_type(cls.wall_type, cls.fire_rating)
        fire_hours = fire_rating_to_hours(cls.fire_rating)
        wall_length_inches = wall.length * to_inches

        # Corner thickness deduction — computed per-end so we can apply to
        # the correct boundary sub-segment when the wall has openings.
        start_ded, end_ded = _get_endpoint_deductions(wall, junction_map, walls, scale)
        corner_deduction = start_ded + end_ded
        effective_length = max(0.0, wall_length_inches - corner_deduction)

        # Split around openings
        wall_openings = opening_map.get(wall.edge_id, [])
        sub_segments = compute_wall_sub_segments(wall, wall_openings, scale)

        if len(sub_segments) == 1 and not wall_openings:
            # Simple wall — solve directly on effective length
            solutions = solve_wall_cutting_stock(
                wall, panel_type, fire_hours, effective_length,
                store, max_solutions_per_wall,
            )
            results.append(WallCuttingResult(
                wall_edge_id=wall.edge_id,
                wall_length_inches=round(wall_length_inches, 4),
                effective_length_inches=round(effective_length, 4),
                panel_type=panel_type,
                solutions=solutions,
                is_panelizable=bool(solutions),
                rejection_reason="" if solutions else "No valid panel configuration found.",
            ))
        else:
            # Wall with openings — solve each sub-segment, then combine.
            # Pass endpoint deductions so the first/last sub-segments are
            # shortened correctly at the wall corners.
            combined_solutions = _solve_wall_with_openings(
                wall, sub_segments, panel_type, fire_hours,
                store, max_solutions_per_wall, start_ded, end_ded,
            )
            results.append(WallCuttingResult(
                wall_edge_id=wall.edge_id,
                wall_length_inches=round(wall_length_inches, 4),
                effective_length_inches=round(effective_length, 4),
                panel_type=panel_type,
                solutions=combined_solutions,
                is_panelizable=bool(combined_solutions),
                rejection_reason="" if combined_solutions else "No valid panel configuration for sub-segments.",
            ))

    return results


def _solve_wall_with_openings(
    wall: WallSegment,
    sub_segments: list[WallSubSegment],
    panel_type: PanelType,
    fire_rating_hours: float,
    store: KnowledgeGraphStore,
    max_solutions: int,
    start_deduction: float = 0.0,
    end_deduction: float = 0.0,
) -> list[CuttingStockSolution]:
    """Solve cutting stock for a wall split into sub-segments by openings.

    For each candidate panel type, solve all sub-segments independently
    then combine into a single wall-level solution.

    ``start_deduction`` and ``end_deduction`` are the corner thickness
    deductions (in inches) at the wall's start and end nodes respectively.
    They are applied to the first and last sub-segments that are bounded by
    the wall endpoints (not by openings).
    """
    candidates = _get_candidate_panels(
        store, wall_type=panel_type, fire_rating_hours=fire_rating_hours,
    )
    if not candidates:
        return []

    last_idx = len(sub_segments) - 1
    solutions: list[CuttingStockSolution] = []

    for panel in candidates:
        # Solve each sub-segment with this panel type
        all_assignments: list[tuple[str, float]] = []
        total_waste = 0.0
        total_material = 0.0
        total_cost = 0.0
        total_pieces = 0
        feasible = True

        for j, seg in enumerate(sub_segments):
            # Apply corner deductions to wall-endpoint sub-segments only.
            seg_length = seg.length_inches
            if j == 0 and not seg.left_bounded_by_opening:
                seg_length = max(0.0, seg_length - start_deduction)
            if j == last_idx and not seg.right_bounded_by_opening:
                seg_length = max(0.0, seg_length - end_deduction)

            if seg_length < panel.min_length_inches:
                # Sub-segment too short — skip this panel type
                feasible = False
                break

            sol = _solve_for_panel_type(
                wall.edge_id, panel, seg_length, store,
            )
            if sol is None:
                feasible = False
                break

            all_assignments.extend(sol.assignments)
            total_waste += sol.waste_inches
            total_material += sum(cl for _, cl in sol.assignments)
            total_cost += sol.total_cost
            total_pieces += sol.num_pieces

        if not feasible or not all_assignments:
            continue

        solutions.append(CuttingStockSolution(
            wall_edge_id=wall.edge_id,
            panel_sku=panel.sku,
            panel=panel,
            assignments=all_assignments,
            waste_inches=round(total_waste, 4),
            waste_percentage=round(total_waste / total_material, 4) if total_material > 0 else 0.0,
            total_cost=round(total_cost, 2),
            requires_splice=total_pieces > 1,
            num_pieces=total_pieces,
            gauge=panel.gauge,
            stud_depth_inches=panel.stud_depth_inches,
            score=_compute_solution_score(total_waste, total_material, total_pieces, total_cost),
        ))

    solutions.sort(key=lambda s: -s.score)
    return solutions[:max_solutions]
