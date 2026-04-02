"""Interface contract: Constraint Agent → Diffusion Agent (gradient feedback).

Defines the ConstraintGradients dataclass — the output of Stage 4
(Differentiable Constraint Enforcement). Provides per-axiom loss values,
total constraint loss, and optionally projected (snapped) geometry.

Mathematical basis (MODEL_SPEC.md §NeSy Constraint SAT, Table 2):

    L_ortho   = Σ (1 - |cos(θ_e1, θ_e2)|²)               [EQ-07]
    L_parallel = Σ max(0, |d(e1,e2) - μ_thickness| - IQR/2)  [EQ-08]
    L_junction = λ · ||L · x||²  (Laplacian penalty)       [EQ-09]
    L_intersect = Σ ReLU(overlap_area(e_i, e_j))           [EQ-10]

    L_SAT = w1·L_ortho + w2·L_parallel + w3·L_junction + w4·L_intersect

Key design:
    - Soft constraints during training (smooth penalties)
    - Hard projection/snap during inference (exact geometry)
    [ARCHITECTURE.md §Stage 4]
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class AxiomResult:
    """Result of evaluating a single architectural axiom.

    Each axiom produces a scalar loss, per-element violation scores,
    and optionally gradient vectors for projection.
    """

    name: str
    """Axiom identifier: 'orthogonal', 'parallel_pair', 'junction_closure',
    'non_intersection', or custom-registered axiom name."""

    loss: torch.Tensor
    """Scalar loss value for this axiom, shape () float32.

    Differentiable w.r.t. node positions and edge logits.
    """

    violation_mask: torch.Tensor
    """Per-element violation indicator, shape varies by axiom.

    For edge-pair axioms (orthogonal, parallel): shape (N_pairs,) bool.
    For node axioms (junction_closure): shape (N_nodes,) bool.
    True where the axiom is violated beyond tolerance.
    """

    violation_scores: torch.Tensor
    """Per-element violation magnitude, shape matches violation_mask, float32.

    Continuous measure of how far each element is from compliance.
    Zero when fully satisfied.
    """

    weight: torch.Tensor
    """Learned weight for this axiom in the composite loss, shape () float32.

    Optimized via meta-learning on validation set.
    Reference: ARCHITECTURE.md §Stage 4 Key Design Decisions.
    """


@dataclass
class ConstraintGradients:
    """Constraint evaluation output from the NeSy SAT solver.

    Contains per-axiom results, the composite constraint loss L_SAT,
    and optionally projected geometry for inference-time hard snapping.

    This dataclass flows back to the Diffusion Agent at each reverse
    diffusion step t, providing gradient signals that guide the denoising
    toward geometrically valid wall configurations.

    Reference: MODEL_SPEC.md §Differentiable NeSy Constraint Satisfaction,
               ARCHITECTURE.md §Stage 4.
    """

    axiom_results: list[AxiomResult]
    """Individual results for each evaluated axiom.

    Standard set: [orthogonal, parallel_pair, junction_closure, non_intersection].
    May include additional custom axioms from the configurable registry.
    """

    total_loss: torch.Tensor
    """Composite constraint loss L_SAT, shape () float32.

    L_SAT = Σ_i (w_i · L_i) where w_i are learned axiom weights.
    Differentiable end-to-end for backpropagation through the diffusion loop.
    """

    projected_positions: torch.Tensor | None
    """Hard-projected node positions after geometric snapping, shape (B, N, 2) float32.

    Non-None only during inference (denoising_step close to 0).
    During training, constraints are enforced via soft loss only.

    Snap operations:
    - Near-orthogonal angles → exact 90° when within tolerance
    - Near-parallel pairs → exact parallel when within tolerance
    - Near-closed junctions → exact closure

    Reference: ARCHITECTURE.md §Stage 4 Key Design Decisions.
    """

    projected_adjacency: torch.Tensor | None = None
    """Hard-projected adjacency after topological correction, shape (B, N, N) float32.

    Non-None only during inference. Removes spurious edges from
    non-intersection violations and adds edges for junction closure.
    """

    wall_thickness_estimates: torch.Tensor | None = None
    """Estimated wall thickness per parallel pair, shape (N_walls,) float32.

    Derived from the parallel pair constancy axiom's distance computation.
    Used downstream by the Physics Agent for FEA mesh discretization
    and by the Serializer for IFC SweptSolid shape representation.
    """

    edge_angles: torch.Tensor | None = None
    """Computed edge angles in radians, shape (E_batch,) float32.

    Angle of each edge relative to the positive x-axis, in [0, π).
    Pre-computed during orthogonal integrity evaluation. Useful for
    downstream modules.
    """

    parallel_pairs: torch.Tensor | None = None
    """Detected parallel edge pairs, shape (N_pairs, 2) int64.

    Each row [i, j] indexes into edge_index, indicating edges i and j
    are identified as a parallel wall pair.
    """

    is_inference: bool = False
    """Whether hard projection was applied (True) or soft loss only (False)."""

    denoising_step: int = 0
    """Current denoising step at which constraints were evaluated."""

    metadata: dict[str, object] = field(default_factory=dict)
    """Additional diagnostic data (violation counts, convergence info, etc.)."""
