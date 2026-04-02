"""Interface contract: Physics Agent → Training loss composition.

Defines the PhysicsLoss dataclass — the output of Stage 6
(Physics Validation via PINN / FEA).

Mathematical basis (MODEL_SPEC.md §Physics-Informed Structural Viability):

    L_PINN = L_data + λ_BC · L_boundary + λ_phys · L_PDE    [EQ-12]

    FEA: K · u = F  (stiffness × displacement = force)
    Adjoint method for gradient computation w.r.t. wall node coordinates.

Architecture:
    - MITC4 quadrilateral shell elements (2D plane stress)
    - 1D Euler-Bernoulli beam-column elements (slender walls)
    - PE-PINN with sin activations (not ReLU) for spectral bias mitigation
    - JAX-SSO for differentiable FEA solving
    [ARCHITECTURE.md §Stage 6]
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import torch


class ViabilityStatus(str, Enum):
    """Structural viability classification for the overall layout."""

    VIABLE = "viable"
    """All structural checks pass within allowable limits."""

    MARGINAL = "marginal"
    """Some elements near allowable limits but within tolerance."""

    FAILED = "failed"
    """One or more structural checks exceed allowable limits."""


@dataclass
class ElementStress:
    """Stress analysis results for a single structural element (wall segment).

    Each wall segment is discretized into FEA elements. This captures
    the peak stress/displacement values for that element.
    """

    element_id: int
    """Index of the wall segment (edge) in the structural graph."""

    max_displacement: float
    """Maximum nodal displacement in the element, in length units."""

    max_shear_stress: float
    """Maximum shear stress (τ_max) in the element, in force/area units."""

    max_bearing_pressure: float
    """Maximum bearing pressure at the element's foundation contact, in force/area units."""

    utilization_ratio: float
    """Ratio of actual stress to allowable stress, dimensionless.

    Values > 1.0 indicate structural failure. Values in [0.8, 1.0]
    are marginal. Values < 0.8 are safe.
    """


@dataclass
class PhysicsLoss:
    """Physics validation loss from differentiable FEA computation.

    Quantifies the structural viability of the predicted wall layout
    by solving the static equilibrium equations and checking that
    displacements, stresses, and load paths are within allowable limits.

    The physics loss backpropagates via the adjoint method, forcing the
    diffusion engine to adjust wall node geometry toward mechanical
    equilibrium.

    Reference: MODEL_SPEC.md §Physics-Informed Structural Viability,
               ARCHITECTURE.md §Stage 6.
    """

    total_loss: torch.Tensor
    """Composite PINN loss, shape () float32.

    L_PINN = L_data + λ_BC · L_boundary + λ_phys · L_PDE

    Differentiable via adjoint method. Backpropagates through
    JAX-SSO solver to wall node coordinates.
    """

    pde_loss: torch.Tensor
    """PDE residual loss, shape () float32.

    MSE of the equilibrium equation residual: K·u - F.
    Measures how far the current geometry is from satisfying
    static equilibrium under applied loads.
    """

    boundary_loss: torch.Tensor
    """Boundary condition loss, shape () float32.

    Penalizes violations of prescribed boundary conditions
    (fixed supports, roller constraints, etc.).
    """

    data_loss: torch.Tensor
    """Data-fitting loss against reference FEA solution, shape () float32.

    MSE(u_predicted, u_reference) where u_reference comes from
    a high-fidelity traditional FEA solver.
    """

    displacement_field: torch.Tensor
    """Nodal displacement vector, shape (N_fem_nodes, 2) float32.

    Solution of K·u = F. Each row is (dx, dy) displacement at a FEM node.
    """

    max_displacement: float
    """Global maximum displacement magnitude across all nodes."""

    max_shear_stress: float
    """Global maximum shear stress across all elements."""

    max_bearing_pressure: float
    """Global maximum bearing pressure across all foundation contacts."""

    allowable_displacement: float
    """Code-prescribed allowable displacement limit (e.g., L/360)."""

    allowable_shear_stress: float
    """Material-specific allowable shear stress (σ_allowable)."""

    viability_status: ViabilityStatus
    """Overall structural viability classification."""

    element_results: list[ElementStress]
    """Per-element stress analysis results.

    One entry per wall segment in the structural graph.
    """

    load_paths: torch.Tensor | None = None
    """Computed load transfer paths, shape (N_paths, N_nodes_per_path) int64.

    Each row traces a load path from roof/floor level down to foundation.
    Identifies the chain of wall segments carrying vertical loads.
    """

    dead_load_total: float = 0.0
    """Total applied dead load (self-weight), in force units."""

    live_load_total: float = 0.0
    """Total applied live load, in force units.

    Default: 40 psf residential per ARCHITECTURE.md §Stage 6.
    """

    node_position_gradients: torch.Tensor | None = None
    """Gradient of physics loss w.r.t. wall node positions, shape (B, N, 2) float32.

    Computed via adjoint method. These gradients indicate how each node
    should move to improve structural viability. Backpropagated into
    the diffusion engine.
    """

    lambda_bc: float = 1.0
    """Weight for boundary condition loss term."""

    lambda_phys: float = 1.0
    """Weight for PDE loss term."""

    metadata: dict[str, object] = field(default_factory=dict)
    """Additional diagnostic data (solver iterations, convergence, element counts, etc.)."""
