"""Forward diffusion process for joint continuous-discrete graphs.

Adds noise to node coordinates (Gaussian) and adjacency matrices
(absorbing-state categorical) over T timesteps.  Used during training
to produce noised inputs (G_t) for the denoising network.

Continuous:
    X_t = sqrt(ᾱ_t) * X_0  +  sqrt(1 - ᾱ_t) * epsilon,   epsilon ~ N(0, I)

Discrete (absorbing-state):
    Each edge kept with probability ᾱ_t, replaced by a uniform Bernoulli(0.5)
    sample with probability (1 - ᾱ_t).

Reference:
    MODEL_SPEC.md  -- EQ-06
    ARCHITECTURE.md -- Stage 3: Graph Diffusion Engine
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch import nn

if TYPE_CHECKING:
    from src.diffusion.scheduler import DiffusionScheduler


@dataclass
class ForwardDiffusionOutput:
    """Output of the forward diffusion process."""

    x_t: torch.Tensor
    """(B, N, 2) noised node coordinates."""

    a_t: torch.Tensor
    """(B, N, N) noised adjacency matrix."""

    epsilon: torch.Tensor
    """(B, N, 2) Gaussian noise added to coordinates."""

    t: torch.Tensor
    """(B,) timesteps used."""

    x_0: torch.Tensor
    """(B, N, 2) original clean coordinates."""

    a_0: torch.Tensor
    """(B, N, N) original clean adjacency."""

    alpha_bar: torch.Tensor
    """(B,) ᾱ_t values used (squeezed from scheduler)."""


class ForwardDiffusion(nn.Module):
    """Forward diffusion for joint continuous-discrete structural graphs.

    Args:
        scheduler: Precomputed noise schedule.
    """

    def __init__(self, scheduler: DiffusionScheduler) -> None:
        super().__init__()
        self.scheduler = scheduler

    # ------------------------------------------------------------------
    # Continuous noise (node coordinates)
    # ------------------------------------------------------------------

    def noise_coordinates(
        self,
        x_0: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Add Gaussian noise to node coordinates.

        X_t = sqrt(ᾱ_t) * X_0 + sqrt(1 - ᾱ_t) * epsilon

        Args:
            x_0: (B, N, 2) clean coordinates.
            t: (B,) integer timesteps.
            noise: Optional pre-sampled noise (B, N, 2). Sampled if *None*.

        Returns:
            ``(x_t, epsilon)`` — noised coordinates and the noise that was added.
        """
        if noise is None:
            noise = torch.randn_like(x_0)

        sqrt_alpha_bar = self.scheduler.get_sqrt_alpha_bar(t)  # (B,1,1)
        sqrt_one_minus = self.scheduler.get_sqrt_one_minus_alpha_bar(t)  # (B,1,1)

        x_t = sqrt_alpha_bar * x_0 + sqrt_one_minus * noise
        return x_t, noise

    # ------------------------------------------------------------------
    # Discrete noise (adjacency matrix)
    # ------------------------------------------------------------------

    def noise_adjacency(
        self,
        a_0: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """Add absorbing-state categorical noise to adjacency.

        Each edge is kept with probability ᾱ_t and replaced by a uniform
        Bernoulli(0.5) sample with probability (1 - ᾱ_t).  The result is
        symmetrised so the graph remains undirected.

        Args:
            a_0: (B, N, N) clean adjacency in {0, 1} (float).
            t: (B,) integer timesteps.

        Returns:
            (B, N, N) noised adjacency (float, values in {0, 1}).
        """
        alpha_bar = self.scheduler.get_alpha_bar(t)  # (B, 1, 1) for broadcasting

        # Bernoulli mask: True where the original edge value is kept.
        keep_mask = torch.rand_like(a_0) < alpha_bar

        # Random replacement values (uniform Bernoulli).
        random_adj = (torch.rand_like(a_0) > 0.5).float()

        a_t = torch.where(keep_mask, a_0, random_adj)

        # Enforce symmetry: use upper triangle, mirror to lower.
        upper = torch.triu(a_t, diagonal=1)
        a_t = upper + upper.transpose(-2, -1)

        # Preserve zero diagonal (no self-loops).
        a_t = a_t * (1.0 - torch.eye(a_0.size(-1), device=a_0.device, dtype=a_0.dtype))

        return a_t

    # ------------------------------------------------------------------
    # Joint forward pass
    # ------------------------------------------------------------------

    def forward(
        self,
        x_0: torch.Tensor,
        a_0: torch.Tensor,
        t: torch.Tensor,
        node_mask: torch.Tensor | None = None,
    ) -> ForwardDiffusionOutput:
        """Run forward diffusion on coordinates and adjacency jointly.

        Args:
            x_0: (B, N, 2) clean node coordinates.
            a_0: (B, N, N) clean adjacency.
            t: (B,) integer timesteps.
            node_mask: (B, N) bool — *True* for valid nodes.

        Returns:
            :class:`ForwardDiffusionOutput` with noised graph and bookkeeping.
        """
        x_t, epsilon = self.noise_coordinates(x_0, t)
        a_t = self.noise_adjacency(a_0, t)

        # Zero out padded positions so they don't contribute to loss.
        if node_mask is not None:
            coord_mask = node_mask.unsqueeze(-1).float()  # (B, N, 1)
            x_t = x_t * coord_mask
            epsilon = epsilon * coord_mask

            # Zero rows and columns of padded nodes in adjacency.
            adj_mask = node_mask.unsqueeze(-1) & node_mask.unsqueeze(-2)  # (B, N, N)
            a_t = a_t * adj_mask.float()

        # Squeeze alpha_bar for the output (B,).
        alpha_bar = self.scheduler.get_alpha_bar(t).view(-1)

        return ForwardDiffusionOutput(
            x_t=x_t,
            a_t=a_t,
            epsilon=epsilon,
            t=t,
            x_0=x_0,
            a_0=a_0,
            alpha_bar=alpha_bar,
        )
