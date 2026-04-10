"""Noise schedules for the forward diffusion process.

Implements the cosine schedule from Nichol & Dhariwal (2021) and a linear
baseline. The scheduler precomputes all diffusion coefficients (betas,
cumulative alphas, posterior terms) so they can be looked up by timestep
without recomputation during training or inference.

Reference:
    MODEL_SPEC.md  -- EQ-06, forward diffusion math
    ARCHITECTURE.md -- Stage 3: Graph Diffusion Engine
"""

from __future__ import annotations

import math

import torch
from torch import nn


class DiffusionScheduler(nn.Module):
    """Cosine / linear noise schedule for graph diffusion.

    Precomputes and registers as buffers:
        betas, alphas, alphas_cumprod, alphas_cumprod_prev,
        sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod,
        posterior_mean_coef1, posterior_mean_coef2, posterior_variance.

    Args:
        num_timesteps: Total diffusion steps T.
        schedule_type: ``"cosine"`` or ``"linear"``.
        s: Cosine schedule offset (prevents beta near 0 at t=0).
        beta_min: Linear schedule minimum beta.
        beta_max: Linear schedule maximum beta.
    """

    def __init__(
        self,
        num_timesteps: int = 1000,
        schedule_type: str = "cosine",
        s: float = 0.008,
        beta_min: float = 1e-4,
        beta_max: float = 0.02,
    ) -> None:
        super().__init__()
        self.num_timesteps = num_timesteps
        self.schedule_type = schedule_type

        if schedule_type == "cosine":
            betas = self._cosine_betas(num_timesteps, s)
        elif schedule_type == "linear":
            betas = torch.linspace(beta_min, beta_max, num_timesteps, dtype=torch.float64)
        else:
            raise ValueError(f"Unknown schedule_type={schedule_type!r}. Use 'cosine' or 'linear'.")

        # Clamp betas to (0, 0.999) for numerical safety.
        betas = betas.clamp(min=1e-8, max=0.999)

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1, dtype=torch.float64), alphas_cumprod[:-1]])

        # Posterior q(x_{t-1} | x_t, x_0).
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        # At t=0 the denominator is 0; clamp to avoid NaN.
        posterior_variance = posterior_variance.clamp(min=1e-7)

        posterior_mean_coef1 = betas * alphas_cumprod_prev.sqrt() / (1.0 - alphas_cumprod)
        posterior_mean_coef2 = (1.0 - alphas_cumprod_prev) * alphas.sqrt() / (1.0 - alphas_cumprod)

        # Register all as float32 buffers (float64 only needed during computation).
        def _reg(name: str, tensor: torch.Tensor) -> None:
            self.register_buffer(name, tensor.float())

        _reg("betas", betas)
        _reg("alphas", alphas)
        _reg("alphas_cumprod", alphas_cumprod)
        _reg("alphas_cumprod_prev", alphas_cumprod_prev)
        _reg("sqrt_alphas_cumprod", alphas_cumprod.sqrt())
        _reg("sqrt_one_minus_alphas_cumprod", (1.0 - alphas_cumprod).sqrt())
        _reg("posterior_mean_coef1", posterior_mean_coef1)
        _reg("posterior_mean_coef2", posterior_mean_coef2)
        _reg("posterior_variance", posterior_variance)
        _reg("posterior_log_variance_clipped", posterior_variance.log())

    # ------------------------------------------------------------------
    # Schedule construction
    # ------------------------------------------------------------------

    @staticmethod
    def _cosine_betas(num_steps: int, s: float) -> torch.Tensor:
        """Derive betas from the cosine alpha-bar schedule.

        ᾱ_t = f(t) / f(0), where f(t) = cos²((t/T + s) / (1+s) * pi/2).
        β_t  = 1 - ᾱ_t / ᾱ_{t-1}.
        """
        steps = torch.arange(num_steps + 1, dtype=torch.float64)
        f = torch.cos(((steps / num_steps) + s) / (1 + s) * (math.pi / 2)) ** 2
        alpha_bar = f / f[0]
        betas = 1.0 - alpha_bar[1:] / alpha_bar[:-1]
        return betas

    # ------------------------------------------------------------------
    # Lookup helpers
    # ------------------------------------------------------------------

    def _gather(self, buf: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Index *buf* at positions *t* and reshape for broadcasting.

        Args:
            buf: (T,) precomputed schedule buffer.
            t: (B,) integer timesteps.

        Returns:
            (B, 1, 1) tensor — broadcastable against (B, N, D) graph tensors.
        """
        out = buf.gather(0, t.long())
        return out.view(-1, 1, 1)

    def get_alpha_bar(self, t: torch.Tensor) -> torch.Tensor:
        """Return ᾱ_t for batch of timesteps. Shape: (B, 1, 1)."""
        return self._gather(self.alphas_cumprod, t)

    def get_sqrt_alpha_bar(self, t: torch.Tensor) -> torch.Tensor:
        """Return sqrt(ᾱ_t). Shape: (B, 1, 1)."""
        return self._gather(self.sqrt_alphas_cumprod, t)

    def get_sqrt_one_minus_alpha_bar(self, t: torch.Tensor) -> torch.Tensor:
        """Return sqrt(1 - ᾱ_t). Shape: (B, 1, 1)."""
        return self._gather(self.sqrt_one_minus_alphas_cumprod, t)

    def sample_timesteps(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Sample timesteps uniformly from [0, T-1].

        Returns:
            (batch_size,) int64 tensor on *device*.
        """
        return torch.randint(0, self.num_timesteps, (batch_size,), device=device, dtype=torch.long)
