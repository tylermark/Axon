"""Interface contract: Topology Agent → Training loss composition.

Defines the TopologyLoss dataclass — the output of Stage 5
(Topological Integrity via Persistent Homology).

Mathematical basis (MODEL_SPEC.md §Topology-Aware Optimization):

    Persistence diagrams: Dgm(f) tracking birth/death of features.
    Wasserstein-1 distance via Sinkhorn-Knopp optimal transport:
        W₁(Dgm_pred, Dgm_target)

    TAFL = α · W₁ + β · |Betti₀_pred - Betti₀_gt| + γ · |Betti₁_pred - Betti₁_gt|

    Integrated into composite loss:
        L_total = L_diffusion + λ_SAT · L_logic + λ_topo · W_p(Dgm_pred, Dgm_target)
        [EQ-11]

Key design:
    - Cubical complex (not simplicial) for grid-aligned filtration efficiency
    - Sinkhorn (not Hungarian) for differentiability
    - Betti counts as auxiliary loss alongside Wasserstein
    [ARCHITECTURE.md §Stage 5]
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class PersistenceDiagram:
    """Persistence diagram tracking topological feature lifetimes.

    Each row represents a topological feature (connected component or loop)
    with its birth and death filtration values.

    Reference: MODEL_SPEC.md §Topology-Aware Optimization.
    """

    points: torch.Tensor
    """Birth-death pairs, shape (K, 2) float32.

    Each row [birth, death] describes a topological feature.
    Features with death >> birth are more persistent (robust).
    Points on the diagonal (birth ≈ death) represent noise.
    """

    homology_dim: int
    """Homology dimension this diagram tracks.

    0 = connected components (Betti-0): tracks merging of components.
    1 = loops/holes (Betti-1): tracks enclosed regions (rooms).
    """


@dataclass
class TopologyLoss:
    """Topological integrity loss from persistent homology computation.

    Quantifies the structural connectivity difference between the predicted
    graph and ground truth using algebraic topology. This loss ensures that:
    - Rooms are properly enclosed (correct Betti-1 count)
    - The building envelope is continuous (correct Betti-0 count)
    - Microscopic gaps don't shatter wall topology

    Reference: MODEL_SPEC.md §Topology-Aware Optimization, ARCHITECTURE.md §Stage 5.
    """

    total_loss: torch.Tensor
    """TAFL (Topology-Aware Focal Loss), shape () float32.

    TAFL = α · W₁ + β · |Betti₀_pred - Betti₀_gt| + γ · |Betti₁_pred - Betti₁_gt|

    Differentiable via Sinkhorn-Knopp optimal transport.
    """

    wasserstein_distance: torch.Tensor
    """Wasserstein-1 distance between persistence diagrams, shape () float32.

    W₁(Dgm_pred, Dgm_target) computed via entropy-regularized optimal transport
    using the Sinkhorn-Knopp algorithm for differentiability.
    """

    betti_0_predicted: int
    """Predicted Betti-0 number (connected components).

    For a valid floor plan, this should equal the ground truth.
    Typically 1 for a single building, more for detached structures.
    """

    betti_0_target: int
    """Ground-truth Betti-0 number."""

    betti_1_predicted: int
    """Predicted Betti-1 number (enclosed loops/holes).

    Each enclosed room contributes one loop. The building exterior
    contributes an additional boundary loop.
    """

    betti_1_target: int
    """Ground-truth Betti-1 number."""

    betti_0_error: torch.Tensor
    """Absolute Betti-0 error, shape () float32.

    |Betti₀_pred - Betti₀_gt|, used as auxiliary loss term.
    """

    betti_1_error: torch.Tensor
    """Absolute Betti-1 error, shape () float32.

    |Betti₁_pred - Betti₁_gt|, used as auxiliary loss term.
    """

    persistence_diagram_pred: list[PersistenceDiagram]
    """Predicted persistence diagrams, one per homology dimension.

    Typically [dim-0 diagram, dim-1 diagram].
    """

    persistence_diagram_target: list[PersistenceDiagram]
    """Ground-truth persistence diagrams, one per homology dimension."""

    transport_plan: torch.Tensor | None = None
    """Optimal transport plan from Sinkhorn, shape (K_pred, K_target) float32.

    The soft assignment matrix between predicted and target persistence
    diagram points. Retained for visualization and debugging.
    """

    alpha: float = 1.0
    """Weight for Wasserstein distance in TAFL."""

    beta: float = 0.5
    """Weight for Betti-0 error in TAFL."""

    gamma: float = 0.5
    """Weight for Betti-1 error in TAFL."""

    sinkhorn_iterations: int = 100
    """Number of Sinkhorn-Knopp iterations used for OT computation."""

    sinkhorn_epsilon: float = 0.01
    """Entropy regularization parameter for Sinkhorn-Knopp."""

    metadata: dict[str, object] = field(default_factory=dict)
    """Additional diagnostic data (convergence info, per-feature lifetimes, etc.)."""
