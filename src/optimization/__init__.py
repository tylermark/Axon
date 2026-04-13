"""OR-based optimization for wall panelization and pod placement.

Replaces the DRL agent (Stage 7) with classical operations research:
- Stage 7a: Per-wall 1D cutting stock (Gilmore-Gomory / FFD)
- Stage 7b: Global CP-SAT coordination (cross-wall constraints, pod placement)

The DRL module (src/drl/) is preserved as a fallback for very large instances.
"""

from src.optimization.solver import OptimizationConfig, optimize_panelization

__all__ = ["OptimizationConfig", "optimize_panelization"]
