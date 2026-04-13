"""Top-level optimization orchestrator for panelization and pod placement.

Wires the cutting stock solver (Stage 7a) and global CP-SAT solver (Stage 7b)
together, producing a PanelizationResult identical to what the DRL agent
produces. Falls back to DRL greedy policy when OR solver is infeasible or
for very large instances.

Usage:
    from src.optimization import optimize_panelization, OptimizationConfig

    result = optimize_panelization(classified_graph, kg_store)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

from src.optimization.cutting_stock import solve_all_walls
from src.optimization.global_cpsat import solve_global_coordination
from src.optimization.result_builder import build_panelization_result

if TYPE_CHECKING:
    from docs.interfaces.classified_wall_graph import ClassifiedWallGraph
    from docs.interfaces.drl_output import PanelizationResult
    from src.knowledge_graph.loader import KnowledgeGraphStore

logger = logging.getLogger(__name__)


@dataclass
class OptimizationConfig:
    """Configuration for the OR-based optimization pipeline.

    This is a lightweight dataclass for direct construction. The pipeline
    also provides ``PanelizationOptimizationConfig`` (Pydantic BaseModel)
    in ``src.pipeline.config`` which is automatically bridged to this class.
    """

    solver_backend: str = "cpsat"
    """Solver to use: ``"cpsat"`` (OR-Tools CP-SAT) or ``"drl"`` (DRL fallback)."""

    cpsat_time_limit_seconds: float = 30.0
    """CP-SAT solver time limit in seconds."""

    sku_minimization_weight: float = 0.1
    """Weight for SKU diversity minimization in objective (0 = disabled)."""

    cost_weight: float = 0.05
    """Weight for cost minimization in objective (0 = disabled)."""

    max_solutions_per_wall: int = 5
    """Maximum candidate cutting stock solutions per wall."""

    drl_fallback_threshold: int = 500
    """Wall count above which DRL fallback is used instead of CP-SAT."""

    @classmethod
    def from_pydantic(cls, pydantic_config: object) -> OptimizationConfig:
        """Create from a PanelizationOptimizationConfig Pydantic model."""
        return cls(
            solver_backend=getattr(pydantic_config, "solver_backend", "cpsat"),
            cpsat_time_limit_seconds=getattr(pydantic_config, "cpsat_time_limit_seconds", 30.0),
            sku_minimization_weight=getattr(pydantic_config, "sku_minimization_weight", 0.1),
            cost_weight=getattr(pydantic_config, "cost_weight", 0.05),
            max_solutions_per_wall=getattr(pydantic_config, "max_solutions_per_wall", 5),
            drl_fallback_threshold=getattr(pydantic_config, "drl_fallback_threshold", 500),
        )


def optimize_panelization(
    classified_graph: ClassifiedWallGraph,
    store: KnowledgeGraphStore,
    config: OptimizationConfig | None = None,
) -> PanelizationResult:
    """Run the OR-based optimization pipeline.

    1. Solve per-wall cutting stock (Stage 7a)
    2. Solve global CP-SAT coordination (Stage 7b)
    3. Build PanelizationResult from the solution

    Falls back to DRL greedy policy if:
    - ``config.solver_backend == "drl"``
    - Wall count exceeds ``config.drl_fallback_threshold``
    - CP-SAT returns INFEASIBLE

    Args:
        classified_graph: Classified wall graph from Layer 1 + classifier.
        store: Knowledge Graph store.
        config: Optimization configuration. Uses defaults if None.

    Returns:
        PanelizationResult matching the interface contract in
        ``docs/interfaces/drl_output.py``.
    """
    if config is None:
        config = OptimizationConfig()

    n_walls = len(classified_graph.graph.wall_segments)

    # Check if DRL fallback is requested or needed
    if config.solver_backend == "drl":
        logger.info("[optimization] Using DRL fallback (configured)")
        return _run_drl_fallback(classified_graph, store)

    if n_walls > config.drl_fallback_threshold:
        logger.info(
            "[optimization] %d walls exceeds threshold %d — using DRL fallback",
            n_walls, config.drl_fallback_threshold,
        )
        return _run_drl_fallback(classified_graph, store)

    # ── Stage 7a: Per-wall cutting stock ──────────────────────────────────
    t_start = time.perf_counter()
    logger.info("[optimization] Stage 7a: Solving cutting stock for %d walls", n_walls)

    wall_results = solve_all_walls(
        classified_graph, store,
        max_solutions_per_wall=config.max_solutions_per_wall,
    )

    panelizable_count = sum(1 for wr in wall_results if wr.is_panelizable)
    logger.info(
        "[optimization] Cutting stock done: %d/%d walls panelizable",
        panelizable_count, n_walls,
    )

    # ── Stage 7b: Global CP-SAT coordination ──────────────────────────────
    logger.info("[optimization] Stage 7b: Solving global coordination via CP-SAT")

    global_solution = solve_global_coordination(
        wall_results=wall_results,
        classified_graph=classified_graph,
        store=store,
        time_limit_seconds=config.cpsat_time_limit_seconds,
        sku_weight=config.sku_minimization_weight,
        cost_weight=config.cost_weight,
    )

    total_time = time.perf_counter() - t_start

    if global_solution.solver_status in ("INFEASIBLE", "MODEL_INVALID", "UNKNOWN"):
        logger.warning(
            "[optimization] CP-SAT returned %s — falling back to DRL",
            global_solution.solver_status,
        )
        return _run_drl_fallback(classified_graph, store)

    logger.info(
        "[optimization] CP-SAT %s in %.2fs: waste=%.1f\", %d panel SKUs, %d pod SKUs",
        global_solution.solver_status,
        global_solution.solve_time_seconds,
        global_solution.total_waste_inches,
        global_solution.num_distinct_panel_skus,
        global_solution.num_distinct_pod_skus,
    )

    # ── Build PanelizationResult ──────────────────────────────────────────
    result = build_panelization_result(
        classified_graph=classified_graph,
        wall_assignments=global_solution.wall_assignments,
        room_assignments=global_solution.room_assignments,
        room_orientations=global_solution.room_orientations,
        total_material_cost=global_solution.total_cost,
        solver_name=f"or_cpsat_{global_solution.solver_status.lower()}",
        solve_time_seconds=total_time,
    )

    logger.info(
        "[optimization] Done: SPUR=%.4f, coverage=%.1f%%, waste=%.1f%%, pods=%.1f%%",
        result.spur_score,
        result.coverage_percentage,
        result.waste_percentage,
        result.pod_placement_rate,
    )

    return result


def _run_drl_fallback(
    classified_graph: ClassifiedWallGraph,
    store: KnowledgeGraphStore,
) -> PanelizationResult:
    """Run the DRL greedy policy as a fallback.

    Imports the DRL env and greedy policy only when needed to avoid
    unnecessary Gymnasium/SB3 imports in the OR-only path.
    """
    from src.drl.env import PanelizationEnv
    from src.drl.train import greedy_policy

    logger.info("[optimization] Running DRL greedy policy fallback")

    env = PanelizationEnv(classified_graph=classified_graph, store=store)
    _obs, _info = env.reset()
    terminated = truncated = False
    while not (terminated or truncated):
        action = greedy_policy(env)
        _obs, _reward, terminated, truncated, _info = env.step(action)

    env_results = env.get_results()
    wall_assignments: dict[int, list[tuple[str, float]]] = env_results["wall_assignments"]
    room_assignments: dict[int, str] = env_results["room_assignments"]

    return build_panelization_result(
        classified_graph=classified_graph,
        wall_assignments=wall_assignments,
        room_assignments=room_assignments,
        solver_name="drl_greedy",
    )
