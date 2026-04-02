"""NeSy SAT constraint solver — differentiable architectural axioms.

Provides four geometric axioms (ARCHITECTURE.md §Stage 4), a
configurable registry, a differentiable SAT solver, Betti-0
regularization, geometric projector, and a top-level constraint
pipeline for composite constraint evaluation during diffusion
denoising.
"""

from src.constraints.axioms import (
    Axiom,
    AxiomRegistry,
    JunctionClosureAxiom,
    OrthogonalIntegrityAxiom,
    ParallelPairConstancyAxiom,
    SpatialNonIntersectionAxiom,
    compute_edge_angles,
    compute_edge_directions,
    edges_from_adjacency,
)
from src.constraints.projector import GeometricProjector
from src.constraints.sat_solver import (
    BettiRegularization,
    ConstraintSolver,
    DifferentiableSATSolver,
)

__all__ = [
    "Axiom",
    "AxiomRegistry",
    "BettiRegularization",
    "ConstraintSolver",
    "DifferentiableSATSolver",
    "GeometricProjector",
    "JunctionClosureAxiom",
    "OrthogonalIntegrityAxiom",
    "ParallelPairConstancyAxiom",
    "SpatialNonIntersectionAxiom",
    "compute_edge_angles",
    "compute_edge_directions",
    "edges_from_adjacency",
]
