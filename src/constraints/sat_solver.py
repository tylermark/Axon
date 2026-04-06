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

    def forward(
        self,
        adjacency: torch.Tensor,
        node_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute Betti-0 regularization loss.

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

        losses: list[torch.Tensor] = []

        for b in range(batch_size):
            a = adj[b]  # (N, N)

            # Apply node mask: zero out rows/cols of invalid nodes.
            if node_mask is not None:
                mask = node_mask[b].float()  # (N,)
                a = a * mask.unsqueeze(0) * mask.unsqueeze(1)
                n_valid = mask.sum().clamp(min=1.0)
            else:
                n_valid = torch.tensor(float(n), device=device)

            # Graph Laplacian: L = D - A.
            degree = a.sum(dim=-1)
            laplacian = torch.diag_embed(degree) - a

            # Number of eigenvalues to check (at most n_valid, capped for speed).
            k = min(int(n_valid.item()), n)
            if k < 2:
                # Trivially 1 component or empty graph.
                losses.append(torch.zeros(1, device=device, requires_grad=True).squeeze())
                continue

            # Eigenvalues (symmetric real matrix → real eigenvalues).
            eigenvalues = torch.linalg.eigvalsh(laplacian[:k, :k])

            # Soft count of near-zero eigenvalues.
            # sigmoid(-eigenvalue / temperature) ≈ 1 when eigenvalue ≈ 0.
            soft_count = torch.sigmoid(-eigenvalues / self.temperature).sum()

            # Loss: deviation from target.
            losses.append((soft_count - self.target_betti_0).abs())

        return torch.stack(losses).mean()


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

    # Chunked pairwise angle difference to avoid O(E²) memory.
    CHUNK = 4096
    ei_list: list[torch.Tensor] = []
    ej_list: list[torch.Tensor] = []
    for start in range(0, num_edges, CHUNK):
        end = min(start + CHUNK, num_edges)
        diff = (edge_angles[start:end].unsqueeze(1) - edge_angles.unsqueeze(0)).abs()
        diff = torch.min(diff, math.pi - diff)
        row_idx = torch.arange(start, end, device=device).unsqueeze(1)
        col_idx = torch.arange(num_edges, device=device).unsqueeze(0)
        mask = (diff < angle_threshold) & (col_idx > row_idx)
        ri, ci = torch.where(mask)
        ei_list.append(ri + start)
        ej_list.append(ci)

    if not ei_list or sum(t.numel() for t in ei_list) == 0:
        return None, None

    ei = torch.cat(ei_list)
    ej = torch.cat(ej_list)
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
