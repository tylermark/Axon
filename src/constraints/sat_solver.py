"""Differentiable SAT solver, Betti regularization, and top-level constraint pipeline.

Implements three modules:

    DifferentiableSATSolver  — Combines axiom losses with learned weights (C-005)
    BettiRegularization      — Spectral Betti-0 regularization for room enclosure (C-006)
    ConstraintSolver         — Top-level pipeline wiring SAT + Betti + Projector (C-005/6/7)

The solver evaluates all registered axioms at each denoising step, producing
a composite loss L_SAT = sum_i softplus(w_i) * L_i that flows back to the
diffusion loop as gradient signals.

Reference:
    MODEL_SPEC.md  Table 2, EQ-07
    ARCHITECTURE.md  Stage 4
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

from docs.interfaces.constraint_signals import AxiomResult, ConstraintGradients
from src.constraints.axioms import (
    AxiomRegistry,
    compute_edge_angles,
    compute_edge_directions,
    find_parallel_pairs,
)

if TYPE_CHECKING:
    from src.pipeline.config import ConstraintConfig


# ---------------------------------------------------------------------------
# C-005: Differentiable SAT Solver
# ---------------------------------------------------------------------------


class DifferentiableSATSolver(nn.Module):
    """Differentiable Boolean SAT solver via convex decomposition.

    Combines all axiom losses into a composite constraint loss L_SAT.
    Axiom weights are learned via meta-learning (when ``config.learn_weights=True``).

    .. math::

        L_{SAT} = \\sum_i \\text{softplus}(w_i) \\cdot L_i

    Args:
        config: Constraint configuration providing tolerances, weights,
            and meta-learning flag.

    Reference:
        MODEL_SPEC.md  Table 2
        ARCHITECTURE.md  Stage 4
    """

    def __init__(self, config: ConstraintConfig) -> None:
        super().__init__()
        self.config = config
        self.registry = AxiomRegistry.create_default(config)

    def forward(
        self,
        node_positions: torch.Tensor,
        adjacency: torch.Tensor,
        edge_index: torch.Tensor,
        node_mask: torch.Tensor | None = None,
        denoising_step: int = 0,
        total_steps: int = 1000,
    ) -> ConstraintGradients:
        """Evaluate all constraints and return composite loss + diagnostics.

        Args:
            node_positions: Junction coordinates, ``(B, N, 2)``.
            adjacency: Adjacency logits or binary, ``(B, N, N)``.
            edge_index: COO edge indices, ``(2, E)``.
            node_mask: Valid-node mask, ``(B, N)`` bool.
            denoising_step: Current reverse-diffusion step *t*.
            total_steps: Total diffusion timesteps *T*.

        Returns:
            ``ConstraintGradients`` with per-axiom results, total loss,
            and auxiliary geometry diagnostics.
        """
        device = node_positions.device

        # --- Evaluate every registered axiom ---
        axiom_results: list[AxiomResult] = self.registry.evaluate_all(
            node_positions, adjacency, edge_index, node_mask
        )

        # --- Composite loss with softplus-positive weights ---
        total_loss = torch.tensor(0.0, device=device, requires_grad=True)
        for result in axiom_results:
            weighted = torch.nn.functional.softplus(result.weight) * result.loss
            total_loss = total_loss + weighted

        # --- Auxiliary diagnostics ---
        edge_angles: torch.Tensor | None = None
        parallel_pairs: torch.Tensor | None = None
        wall_thickness_estimates: torch.Tensor | None = None

        if edge_index.shape[1] > 0:
            directions = compute_edge_directions(node_positions, edge_index)
            edge_angles = compute_edge_angles(directions)

            # Extract parallel pair info from the parallel_pair axiom result.
            pp_result = _find_result(axiom_results, "parallel_pair")
            if pp_result is not None and pp_result.violation_mask.numel() > 0:
                parallel_pairs, wall_thickness_estimates = _extract_parallel_info(
                    node_positions, edge_index, directions, edge_angles, self.config
                )

        return ConstraintGradients(
            axiom_results=axiom_results,
            total_loss=total_loss,
            projected_positions=None,
            projected_adjacency=None,
            wall_thickness_estimates=wall_thickness_estimates,
            edge_angles=edge_angles,
            parallel_pairs=parallel_pairs,
            is_inference=False,
            denoising_step=denoising_step,
        )


# ---------------------------------------------------------------------------
# C-006: Betti Regularization
# ---------------------------------------------------------------------------


class BettiRegularization(nn.Module):
    """Lightweight Betti-0 regularization for room enclosure.

    Encourages the predicted graph to form a single connected component
    (Betti-0 = 1) using a differentiable spectral approximation of the
    graph Laplacian.

    The number of connected components equals the multiplicity of the
    zero eigenvalue of the Laplacian.  We approximate this count via
    the sum of ``sigmoid(-eigenvalue / temperature)`` for the *k* smallest
    eigenvalues, then penalize deviation from ``target_betti_0``.

    Args:
        target_betti_0: Desired number of connected components (1 for a
            single connected floor plan).
        d_model: Not used for learnable parameters; reserved for interface
            consistency with other modules.

    Reference:
        ARCHITECTURE.md  Stage 4 — "Lightweight Betti number regularization
        for room enclosure (training only)"
    """

    def __init__(self, target_betti_0: int = 1, d_model: int = 64) -> None:
        super().__init__()
        self.target_betti_0 = target_betti_0
        # Temperature controls the sharpness of the soft zero-eigenvalue count.
        self.temperature = nn.Parameter(torch.tensor(0.1), requires_grad=False)

    # NOTE(constrain): k_cap limits the eigsh subspace to the smallest k_cap×k_cap
    # block of each Laplacian. For Betti-0 (counting near-zero eigenvalues = connected
    # components) only the smallest few eigenvalues matter, so capping at 128 is safe
    # for typical target_betti_0 values (1–4). If target_betti_0 is set above 64,
    # increase k_cap accordingly.
    _K_CAP: int = 128

    def forward(
        self,
        adjacency: torch.Tensor,
        node_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute Betti-0 regularization loss.

        Batched implementation: a single ``torch.linalg.eigvalsh`` call on a
        ``(B, k, k)`` tensor replaces the original per-sample Python loop,
        eliminating B−1 redundant CUDA-sync round-trips and CPU-side overhead.

        Args:
            adjacency: Adjacency logits or binary, ``(B, N, N)``.
            node_mask: Valid-node mask, ``(B, N)`` bool.

        Returns:
            Scalar loss penalizing deviation from ``target_betti_0``.
        """
        if adjacency.dim() == 2:
            adjacency = adjacency.unsqueeze(0)

        # Soft adjacency from logits.
        adj = torch.sigmoid(adjacency) if adjacency.requires_grad else adjacency
        # Symmetrise (adjacency should already be symmetric, but ensure it).
        adj = (adj + adj.transpose(-1, -2)) * 0.5

        batch_size, n, _ = adj.shape
        device = adj.device

        # Cap k for speed: only the smallest k eigenvalues matter for Betti-0.
        k = min(n, self._K_CAP)

        if k < 2:
            # Every sample is trivially 1 component or empty.
            # Return a zero loss that still carries a grad path through adj via
            # a dummy trace-based term that is always 0 but connects the graph.
            # NOTE(constrain): the dummy sum is scaled to 0 so it doesn't bias
            # the loss, but it keeps adj in the computation graph for gradient flow.
            dummy = (adj * 0.0).sum() * 0.0
            zero = torch.zeros((), device=device)
            return zero + dummy

        # Apply node mask before building the Laplacian.
        # Invalid nodes get their adj rows/cols zeroed, then their Laplacian diagonal
        # is set to a large sentinel value so their eigenvalues land far above zero
        # and sigmoid(-λ/T)→0, effectively excluding them from the soft component count.
        # NOTE(constrain): this strategy lets all samples use the same k=min(n,_K_CAP)
        # subspace regardless of how many valid nodes each sample has, which is required
        # for batching eigvalsh over heterogeneous node_mask counts.
        _SENTINEL = 1e4
        if node_mask is not None:
            mask_f = node_mask.float()  # (B, N)
            adj = adj * mask_f.unsqueeze(1) * mask_f.unsqueeze(2)
            invalid_mask = ~node_mask  # (B, N) — True where node is padding
        else:
            invalid_mask = None

        # Build batched Laplacian for the [:k, :k] subspace: (B, k, k).
        adj_k = adj[:, :k, :k]
        degree_k = adj_k.sum(dim=-1)  # (B, k)
        laplacian_k = torch.diag_embed(degree_k) - adj_k  # (B, k, k)

        # Inject sentinel on the diagonal of padding nodes so their eigenvalues
        # are pushed far above zero.
        if invalid_mask is not None:
            inv_k = invalid_mask[:, :k].float() * _SENTINEL  # (B, k)
            # diag_embed produces (B, k, k); adding to laplacian diagonal:
            laplacian_k = laplacian_k + torch.diag_embed(inv_k)

        # Single batched eigendecomposition — one CUDA kernel instead of B.
        eigenvalues = torch.linalg.eigvalsh(laplacian_k)  # (B, k)

        # Soft count of near-zero eigenvalues per sample.
        soft_counts = torch.sigmoid(-eigenvalues / self.temperature).sum(dim=-1)  # (B,)

        # Loss: mean deviation from target across the batch.
        return (soft_counts - self.target_betti_0).abs().mean()


# ---------------------------------------------------------------------------
# Top-level ConstraintSolver
# ---------------------------------------------------------------------------


class ConstraintSolver(nn.Module):
    """Complete constraint evaluation pipeline.

    Wires together:

    1. ``DifferentiableSATSolver`` — axiom evaluation + composite loss
    2. ``BettiRegularization`` — topological regularization (training)
    3. ``GeometricProjector`` — hard snapping (inference)

    Args:
        config: Constraint configuration.
    """

    def __init__(self, config: ConstraintConfig) -> None:
        super().__init__()
        self.config = config
        self.sat_solver = DifferentiableSATSolver(config)
        self.betti = BettiRegularization()

        # Import here to avoid circular dependency at module level.
        from src.constraints.projector import GeometricProjector

        self.projector = GeometricProjector(config)

    def forward(
        self,
        node_positions: torch.Tensor,
        adjacency: torch.Tensor,
        edge_index: torch.Tensor,
        node_mask: torch.Tensor | None = None,
        denoising_step: int = 0,
        total_steps: int = 1000,
        is_inference: bool = False,
    ) -> ConstraintGradients:
        """Full constraint evaluation.

        During training: soft SAT loss + Betti regularization.
        During inference: soft loss + hard projection when
        ``denoising_step <= config.snap_at_step``.

        Args:
            node_positions: Junction coordinates, ``(B, N, 2)``.
            adjacency: Adjacency logits or binary, ``(B, N, N)``.
            edge_index: COO edge indices, ``(2, E)``.
            node_mask: Valid-node mask, ``(B, N)`` bool.
            denoising_step: Current reverse-diffusion step *t*.
            total_steps: Total diffusion timesteps *T*.
            is_inference: Whether to apply hard projection.

        Returns:
            ``ConstraintGradients`` with combined losses and optional
            projected geometry.
        """
        # --- SAT solver ---
        gradients = self.sat_solver(
            node_positions,
            adjacency,
            edge_index,
            node_mask,
            denoising_step,
            total_steps,
        )

        # --- Betti regularization (training path) ---
        betti_loss = self.betti(adjacency, node_mask)
        total_loss = gradients.total_loss + betti_loss

        # Store Betti loss in metadata for diagnostics.
        metadata: dict[str, object] = {"betti_loss": betti_loss.detach().item()}

        # --- Hard projection (inference, near final step) ---
        projected_positions: torch.Tensor | None = None
        projected_adjacency: torch.Tensor | None = None

        if is_inference and denoising_step <= self.config.snap_at_step:
            projected_positions, projected_adjacency = self.projector.project(
                node_positions,
                adjacency,
                edge_index,
                node_mask,
                axiom_results=gradients.axiom_results,
            )

        return ConstraintGradients(
            axiom_results=gradients.axiom_results,
            total_loss=total_loss,
            projected_positions=projected_positions,
            projected_adjacency=projected_adjacency,
            wall_thickness_estimates=gradients.wall_thickness_estimates,
            edge_angles=gradients.edge_angles,
            parallel_pairs=gradients.parallel_pairs,
            is_inference=is_inference and denoising_step <= self.config.snap_at_step,
            denoising_step=denoising_step,
            metadata=metadata,
        )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _find_result(results: list[AxiomResult], name: str) -> AxiomResult | None:
    """Find an axiom result by name."""
    for r in results:
        if r.name == name:
            return r
    return None


def _extract_parallel_info(
    node_positions: torch.Tensor,
    edge_index: torch.Tensor,
    directions: torch.Tensor,
    edge_angles: torch.Tensor,
    config: ConstraintConfig,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Extract parallel pair indices and wall thickness estimates.

    Re-uses the angle threshold from the parallel pair axiom to identify
    parallel edges and measure perpendicular distance (wall thickness).

    Returns:
        ``(parallel_pairs, wall_thickness_estimates)`` or ``(None, None)``
        if no parallel pairs are found.
    """
    device = node_positions.device
    num_edges = edge_index.shape[1]
    if num_edges < 2:
        return None, None

    angle_threshold = math.radians(5.0)  # consistent with axiom default

    # O(E log E) parallel pair detection via sort + scan.
    ei, ej = find_parallel_pairs(edge_angles, angle_threshold)
    if ei.numel() == 0:
        return None, None

    parallel_pairs = torch.stack([ei, ej], dim=1)  # (N_pairs, 2)

    # Wall thickness = perpendicular distance between parallel edges.
    positions = node_positions.reshape(-1, 2) if node_positions.dim() == 3 else node_positions
    src_j, dst_j = edge_index[0, ej], edge_index[1, ej]
    mid_j = (positions[src_j] + positions[dst_j]) * 0.5

    d_i = directions[ei]
    normal = torch.stack([-d_i[:, 1], d_i[:, 0]], dim=-1)
    delta = mid_j - positions[edge_index[0, ei]]
    wall_thickness = (delta * normal).sum(dim=-1).abs()

    return parallel_pairs, wall_thickness
