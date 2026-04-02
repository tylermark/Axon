"""Unit tests for src/diffusion/scheduler.py.

Tests DiffusionScheduler: cosine and linear noise schedules,
coefficient properties, timestep sampling, and edge cases.

Q-005: diffusion unit tests (scheduler).
"""

from __future__ import annotations

import torch

from src.diffusion.scheduler import DiffusionScheduler

# ---------------------------------------------------------------------------
# Cosine schedule
# ---------------------------------------------------------------------------


class TestCosineSchedule:
    """Tests for the cosine noise schedule."""

    def test_betas_positive(self):
        sched = DiffusionScheduler(num_timesteps=100, schedule_type="cosine")
        assert (sched.betas > 0).all()

    def test_betas_upper_bounded(self):
        sched = DiffusionScheduler(num_timesteps=100, schedule_type="cosine")
        assert (sched.betas <= 0.999).all()

    def test_alphas_cumprod_monotonically_decreasing(self):
        sched = DiffusionScheduler(num_timesteps=100, schedule_type="cosine")
        diffs = sched.alphas_cumprod[1:] - sched.alphas_cumprod[:-1]
        assert (diffs < 0).all(), "alphas_cumprod must be monotonically decreasing"

    def test_alphas_cumprod_start_near_one(self):
        sched = DiffusionScheduler(num_timesteps=100, schedule_type="cosine")
        assert sched.alphas_cumprod[0] > 0.99

    def test_alphas_cumprod_end_near_zero(self):
        sched = DiffusionScheduler(num_timesteps=1000, schedule_type="cosine")
        assert sched.alphas_cumprod[-1] < 0.05


# ---------------------------------------------------------------------------
# Linear schedule
# ---------------------------------------------------------------------------


class TestLinearSchedule:
    """Tests for the linear noise schedule."""

    def test_betas_linearly_spaced(self):
        sched = DiffusionScheduler(
            num_timesteps=100,
            schedule_type="linear",
            beta_min=1e-4,
            beta_max=0.02,
        )
        expected = torch.linspace(1e-4, 0.02, 100)
        torch.testing.assert_close(sched.betas, expected, atol=1e-6, rtol=1e-5)

    def test_betas_positive(self):
        sched = DiffusionScheduler(num_timesteps=100, schedule_type="linear")
        assert (sched.betas > 0).all()

    def test_alphas_cumprod_monotonically_decreasing(self):
        sched = DiffusionScheduler(num_timesteps=100, schedule_type="linear")
        diffs = sched.alphas_cumprod[1:] - sched.alphas_cumprod[:-1]
        assert (diffs < 0).all()


# ---------------------------------------------------------------------------
# Lookup helpers
# ---------------------------------------------------------------------------


class TestSchedulerLookups:
    """Tests for get_alpha_bar, sample_timesteps, and coefficient consistency."""

    def test_get_alpha_bar_shape(self):
        sched = DiffusionScheduler(num_timesteps=100)
        t = torch.tensor([0, 50, 99])
        ab = sched.get_alpha_bar(t)
        assert ab.shape == (3, 1, 1)

    def test_get_alpha_bar_values_match_buffer(self):
        sched = DiffusionScheduler(num_timesteps=100)
        t = torch.tensor([10, 50, 90])
        ab = sched.get_alpha_bar(t)
        for i, ti in enumerate(t):
            torch.testing.assert_close(ab[i, 0, 0], sched.alphas_cumprod[ti.item()])

    def test_sample_timesteps_range(self):
        sched = DiffusionScheduler(num_timesteps=100)
        t = sched.sample_timesteps(1000, device=torch.device("cpu"))
        assert t.min() >= 0
        assert t.max() <= 99
        assert t.shape == (1000,)

    def test_sample_timesteps_dtype(self):
        sched = DiffusionScheduler(num_timesteps=100)
        t = sched.sample_timesteps(4, device=torch.device("cpu"))
        assert t.dtype == torch.long

    def test_sqrt_coefficients_consistent(self):
        """sqrt_alphas_cumprod^2 + sqrt_one_minus_alphas_cumprod^2 ≈ 1."""
        sched = DiffusionScheduler(num_timesteps=100)
        sum_sq = sched.sqrt_alphas_cumprod**2 + sched.sqrt_one_minus_alphas_cumprod**2
        torch.testing.assert_close(sum_sq, torch.ones_like(sum_sq), atol=1e-5, rtol=1e-5)

    def test_posterior_variance_non_negative(self):
        sched = DiffusionScheduler(num_timesteps=100)
        assert (sched.posterior_variance >= 0).all()


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestSchedulerEdgeCases:
    """Edge cases and multiple timestep counts."""

    def test_t100_works(self):
        sched = DiffusionScheduler(num_timesteps=100, schedule_type="cosine")
        assert sched.betas.shape == (100,)

    def test_t1000_works(self):
        sched = DiffusionScheduler(num_timesteps=1000, schedule_type="cosine")
        assert sched.betas.shape == (1000,)

    def test_unknown_schedule_raises(self):
        import pytest

        with pytest.raises(ValueError, match="Unknown schedule_type"):
            DiffusionScheduler(num_timesteps=100, schedule_type="quadratic")

    def test_buffers_are_float32(self):
        sched = DiffusionScheduler(num_timesteps=100)
        assert sched.betas.dtype == torch.float32
        assert sched.alphas_cumprod.dtype == torch.float32
        assert sched.posterior_variance.dtype == torch.float32
