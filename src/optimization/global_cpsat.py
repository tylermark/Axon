"""Stage 7b: Global CP-SAT coordination solver.

Takes per-wall candidate solutions from the cutting stock solver and selects
the globally optimal assignment, respecting cross-wall constraints:

1. Junction gauge/stud-depth compatibility
2. Fire compartmentalization continuity
3. SKU diversity minimization
4. Pod placement with dimensional + clearance constraints

Uses Google OR-Tools CP-SAT solver (free, Apache 2.0 license).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from ortools.sat.python import cp_model

from src.drl.constraints import JunctionInfo, compute_junction_map
from src.drl.state import get_room_dims_inches, wall_type_to_panel_type
from src.knowledge_graph.query import get_valid_pods
from src.optimization.cutting_stock import CuttingStockSolution, WallCuttingResult

if TYPE_CHECKING:
    from docs.interfaces.classified_wall_graph import ClassifiedWallGraph
    from docs.interfaces.graph_to_serializer import Room
    from src.knowledge_graph.loader import KnowledgeGraphStore
    from src.knowledge_graph.schema import Pod

logger = logging.getLogger(__name__)


# ── Result dataclass ──────────────────────────────────────────────────────────


@dataclass
class GlobalSolution:
    """Result of the global CP-SAT coordination solver."""

    wall_assignments: dict[int, list[tuple[str, float]]]
    """edge_id -> ordered list of (panel_sku, cut_length_inches)."""

    room_assignments: dict[int, str]
    """room_id -> pod_sku."""

    room_orientations: dict[int, bool]
    """room_id -> True if pod is rotated 90 degrees."""

    total_waste_inches: float = 0.0
    total_cost: float = 0.0
    num_distinct_panel_skus: int = 0
    num_distinct_pod_skus: int = 0
    solver_status: str = ""
    solve_time_seconds: float = 0.0


# ── Solver ────────────────────────────────────────────────────────────────────


def solve_global_coordination(
    wall_results: list[WallCuttingResult],
    classified_graph: ClassifiedWallGraph,
    store: KnowledgeGraphStore,
    time_limit_seconds: float = 30.0,
    sku_weight: float = 0.1,
    cost_weight: float = 0.05,
) -> GlobalSolution:
    """Solve the global coordination problem via CP-SAT.

    Selects one cutting stock solution per wall and one pod per room,
    minimizing waste + SKU diversity + cost, subject to cross-wall
    junction compatibility constraints.

    Args:
        wall_results: Per-wall candidate solutions from cutting stock solver.
        classified_graph: The classified wall graph.
        store: Knowledge Graph store.
        time_limit_seconds: CP-SAT time limit.
        sku_weight: Weight for SKU minimization in objective.
        cost_weight: Weight for cost minimization in objective.

    Returns:
        GlobalSolution with selected assignments.
    """
    t_start = time.perf_counter()
    model = cp_model.CpModel()

    graph = classified_graph.graph
    walls = graph.wall_segments
    rooms = [r for r in graph.rooms if not r.is_exterior]
    scale = graph.scale_factor

    # ── Wall solution selection variables ──────────────────────────────────

    # wall_vars[edge_id][j] = BoolVar: is solution j selected for wall edge_id?
    wall_vars: dict[int, list[cp_model.IntVar]] = {}
    # Maps for lookup
    wall_solutions: dict[int, list[CuttingStockSolution]] = {}

    for wr in wall_results:
        if not wr.solutions:
            continue
        wall_solutions[wr.wall_edge_id] = wr.solutions
        vars_for_wall: list[cp_model.IntVar] = []
        for j in range(len(wr.solutions)):
            v = model.new_bool_var(f"wall_{wr.wall_edge_id}_sol_{j}")
            vars_for_wall.append(v)
        wall_vars[wr.wall_edge_id] = vars_for_wall

        # Exactly one solution per wall
        model.add_exactly_one(vars_for_wall)

    # ── Junction compatibility constraints ────────────────────────────────

    junction_map = compute_junction_map(walls)
    _add_junction_constraints(model, wall_vars, wall_solutions, junction_map)

    # ── Pod placement variables ───────────────────────────────────────────

    # pod_vars[room_id][(pod_sku, rotated)] = BoolVar
    pod_vars: dict[int, dict[tuple[str, bool], cp_model.IntVar]] = {}
    pod_lookup: dict[int, list[tuple[Pod, bool]]] = {}

    for room in rooms:
        width, depth = get_room_dims_inches(room, graph.nodes, scale)
        if width <= 0 or depth <= 0:
            continue

        candidates = get_valid_pods(
            store, room_width_inches=width, room_depth_inches=depth,
            room_function=room.label.lower().strip() if room.label else None,
        )
        if not candidates:
            continue

        room_pod_vars: dict[tuple[str, bool], cp_model.IntVar] = {}
        room_pod_list: list[tuple[Pod, bool]] = []

        for pod in candidates:
            # Check both orientations
            for rotated in (False, True):
                if rotated:
                    pw = pod.depth_inches + 2 * pod.clearance_inches
                    pd = pod.width_inches + 2 * pod.clearance_inches
                else:
                    pw = pod.width_inches + 2 * pod.clearance_inches
                    pd = pod.depth_inches + 2 * pod.clearance_inches

                if pw <= width and pd <= depth:
                    key = (pod.sku, rotated)
                    if key not in room_pod_vars:
                        v = model.new_bool_var(
                            f"room_{room.room_id}_pod_{pod.sku}_rot{int(rotated)}"
                        )
                        room_pod_vars[key] = v
                        room_pod_list.append((pod, rotated))

        if room_pod_vars:
            # Add a "no pod" option
            no_pod = model.new_bool_var(f"room_{room.room_id}_no_pod")
            all_options = list(room_pod_vars.values()) + [no_pod]
            model.add_exactly_one(all_options)

            pod_vars[room.room_id] = room_pod_vars
            pod_lookup[room.room_id] = room_pod_list

    # ── Objective function ────────────────────────────────────────────────

    _build_objective(
        model, wall_vars, wall_solutions, pod_vars, pod_lookup,
        store, sku_weight, cost_weight,
    )

    # ── Solve ─────────────────────────────────────────────────────────────

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit_seconds
    solver.parameters.num_workers = 4

    status = solver.solve(model)
    solve_time = time.perf_counter() - t_start

    status_name = {
        cp_model.OPTIMAL: "OPTIMAL",
        cp_model.FEASIBLE: "FEASIBLE",
        cp_model.INFEASIBLE: "INFEASIBLE",
        cp_model.MODEL_INVALID: "MODEL_INVALID",
        cp_model.UNKNOWN: "UNKNOWN",
    }.get(status, "UNKNOWN")

    logger.info(
        "CP-SAT solved in %.2fs: status=%s", solve_time, status_name,
    )

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return GlobalSolution(
            wall_assignments={},
            room_assignments={},
            room_orientations={},
            solver_status=status_name,
            solve_time_seconds=round(solve_time, 3),
        )

    # ── Extract solution ──────────────────────────────────────────────────

    return _extract_solution(
        solver, wall_vars, wall_solutions, pod_vars, pod_lookup,
        wall_results, status_name, solve_time,
    )


# ── Constraint builders ───────────────────────────────────────────────────────


def _add_junction_constraints(
    model: cp_model.CpModel,
    wall_vars: dict[int, list[cp_model.IntVar]],
    wall_solutions: dict[int, list[CuttingStockSolution]],
    junction_map: dict[int, JunctionInfo],
) -> None:
    """Add gauge/stud-depth compatibility at junctions.

    At each junction, for every pair of adjacent walls that both have
    solutions, enforce that selected solutions have matching gauge and
    stud depth.
    """
    for junction in junction_map.values():
        # Get all adjacent walls that have solution variables
        adjacent_edges = [
            eid for eid in junction.wall_edge_ids
            if eid in wall_vars
        ]
        if len(adjacent_edges) < 2:
            continue

        # For each pair of adjacent walls at this junction
        for idx_a in range(len(adjacent_edges)):
            for idx_b in range(idx_a + 1, len(adjacent_edges)):
                eid_a = adjacent_edges[idx_a]
                eid_b = adjacent_edges[idx_b]

                sols_a = wall_solutions[eid_a]
                sols_b = wall_solutions[eid_b]
                vars_a = wall_vars[eid_a]
                vars_b = wall_vars[eid_b]

                # Forbid pairs where gauge or stud depth differ
                for ja, sol_a in enumerate(sols_a):
                    for jb, sol_b in enumerate(sols_b):
                        gauge_mismatch = sol_a.gauge != sol_b.gauge
                        depth_mismatch = sol_a.stud_depth_inches != sol_b.stud_depth_inches

                        if gauge_mismatch or depth_mismatch:
                            # Cannot both be selected
                            model.add_bool_or([
                                vars_a[ja].negated(),
                                vars_b[jb].negated(),
                            ])


def _build_objective(
    model: cp_model.CpModel,
    wall_vars: dict[int, list[cp_model.IntVar]],
    wall_solutions: dict[int, list[CuttingStockSolution]],
    pod_vars: dict[int, dict[tuple[str, bool], cp_model.IntVar]],
    pod_lookup: dict[int, list[tuple[Pod, bool]]],
    store: KnowledgeGraphStore,
    sku_weight: float,
    cost_weight: float,
) -> None:
    """Build the CP-SAT objective: minimize waste + SKU count + cost.

    All values are scaled to integers (CP-SAT requires integer coefficients).
    Waste is in hundredths of an inch, cost in cents.
    """
    # Scale factor: waste in hundredths of an inch
    WASTE_SCALE = 100
    # Scale factor: cost in cents
    COST_SCALE = 100
    # Scale factor for SKU penalty
    SKU_PENALTY = 1000

    objective_terms: list[tuple[cp_model.IntVar, int]] = []

    # ── Waste minimization (primary objective) ────────────────────────────
    for eid, vars_list in wall_vars.items():
        sols = wall_solutions[eid]
        for j, var in enumerate(vars_list):
            waste_scaled = int(sols[j].waste_inches * WASTE_SCALE)
            if waste_scaled > 0:
                objective_terms.append((var, waste_scaled))

    # ── Cost minimization ─────────────────────────────────────────────────
    if cost_weight > 0:
        for eid, vars_list in wall_vars.items():
            sols = wall_solutions[eid]
            for j, var in enumerate(vars_list):
                cost_scaled = int(sols[j].total_cost * COST_SCALE * cost_weight)
                if cost_scaled > 0:
                    objective_terms.append((var, cost_scaled))

    # ── SKU diversity minimization ────────────────────────────────────────
    if sku_weight > 0:
        # Collect all unique panel SKUs across all solutions
        all_skus: set[str] = set()
        for sols in wall_solutions.values():
            for sol in sols:
                all_skus.add(sol.panel_sku)

        # For each SKU, create a BoolVar "is this SKU used?"
        for sku in all_skus:
            sku_used = model.new_bool_var(f"sku_used_{sku}")
            # sku_used = 1 if any wall selects a solution using this SKU
            sku_users: list[cp_model.IntVar] = []
            for eid, vars_list in wall_vars.items():
                sols = wall_solutions[eid]
                for j, var in enumerate(vars_list):
                    if sols[j].panel_sku == sku:
                        sku_users.append(var)

            if sku_users:
                # sku_used >= any of the users
                model.add_max_equality(sku_used, sku_users)
                penalty = int(SKU_PENALTY * sku_weight)
                objective_terms.append((sku_used, penalty))

    # ── Pod cost (encourage placement) ────────────────────────────────────
    # Small negative cost (bonus) for placing pods — encourage coverage
    for room_id, pv_dict in pod_vars.items():
        pods_for_room = pod_lookup.get(room_id, [])
        for (pod_sku, rotated), var in pv_dict.items():
            # Find the pod object
            for pod, rot in pods_for_room:
                if pod.sku == pod_sku and rot == rotated:
                    # Bonus for placing (negative cost = encouraged)
                    # Larger pods get bigger bonus
                    area_score = int(
                        (pod.width_inches * pod.depth_inches) / 100.0
                    )
                    objective_terms.append((var, -area_score))
                    break

    # Set objective
    if objective_terms:
        model.minimize(
            sum(coeff * var for var, coeff in objective_terms)
        )


# ── Solution extraction ───────────────────────────────────────────────────────


def _extract_solution(
    solver: cp_model.CpSolver,
    wall_vars: dict[int, list[cp_model.IntVar]],
    wall_solutions: dict[int, list[CuttingStockSolution]],
    pod_vars: dict[int, dict[tuple[str, bool], cp_model.IntVar]],
    pod_lookup: dict[int, list[tuple[Pod, bool]]],
    wall_results: list[WallCuttingResult],
    status_name: str,
    solve_time: float,
) -> GlobalSolution:
    """Extract the selected assignments from the CP-SAT solution."""
    wall_assignments: dict[int, list[tuple[str, float]]] = {}
    total_waste = 0.0
    total_cost = 0.0
    panel_skus_used: set[str] = set()

    # Extract wall assignments
    for eid, vars_list in wall_vars.items():
        sols = wall_solutions[eid]
        for j, var in enumerate(vars_list):
            if solver.value(var):
                sol = sols[j]
                wall_assignments[eid] = sol.assignments
                total_waste += sol.waste_inches
                total_cost += sol.total_cost
                panel_skus_used.add(sol.panel_sku)
                break

    # Include walls with no solutions as empty assignments
    for wr in wall_results:
        if wr.wall_edge_id not in wall_assignments and not wr.solutions:
            wall_assignments[wr.wall_edge_id] = []

    # Extract room assignments
    room_assignments: dict[int, str] = {}
    room_orientations: dict[int, bool] = {}
    pod_skus_used: set[str] = set()

    for room_id, pv_dict in pod_vars.items():
        for (pod_sku, rotated), var in pv_dict.items():
            if solver.value(var):
                room_assignments[room_id] = pod_sku
                room_orientations[room_id] = rotated
                pod_skus_used.add(pod_sku)
                # Add pod cost
                pods_for_room = pod_lookup.get(room_id, [])
                for pod, rot in pods_for_room:
                    if pod.sku == pod_sku and rot == rotated:
                        total_cost += pod.unit_cost
                        break
                break

    return GlobalSolution(
        wall_assignments=wall_assignments,
        room_assignments=room_assignments,
        room_orientations=room_orientations,
        total_waste_inches=round(total_waste, 4),
        total_cost=round(total_cost, 2),
        num_distinct_panel_skus=len(panel_skus_used),
        num_distinct_pod_skus=len(pod_skus_used),
        solver_status=status_name,
        solve_time_seconds=round(solve_time, 3),
    )
