"""Unit tests for src/diffusion/reverse.py.

Tests TimestepEmbedding, GraphTransformerBlock, GraphTransformerBackbone,
ReverseDiffusion, DiffusionLoss, and GraphDiffusionModel.

Q-005: diffusion unit tests (reverse process).
"""

from __future__ import annotations

import torch

from src.diffusion.forward import ForwardDiffusion
from src.diffusion.hdse import HDSE
from src.diffusion.reverse import (
    DiffusionLoss,
    GraphDiffusionModel,
    GraphTransformerBackbone,
    GraphTransformerBlock,
    ReverseDiffusion,
    TimestepEmbedding,
)
from src.diffusion.scheduler import DiffusionScheduler
from tests.fixtures.diffusion_helpers import (
    create_small_diffusion_config,
    create_synthetic_graph,
)

B, N, D = 2, 8, 64
N_CTX, D_CTX = 6, 256


def _make_context() -> tuple[torch.Tensor, torch.Tensor]:
    """Synthetic cross-modal context."""
    ctx = torch.randn(B, N_CTX, D_CTX)
    ctx_mask = torch.ones(B, N_CTX, dtype=torch.bool)
    return ctx, ctx_mask


# ---------------------------------------------------------------------------
# TimestepEmbedding
# ---------------------------------------------------------------------------


class TestTimestepEmbedding:
    """Tests for TimestepEmbedding."""

    def test_output_shape(self):
        te = TimestepEmbedding(d_model=D)
        t = torch.tensor([0, 50])
        out = te(t)
        assert out.shape == (2, D)

    def test_different_timesteps_different_embeddings(self):
        te = TimestepEmbedding(d_model=D)
        t = torch.tensor([0, 99])
        out = te(t)
        assert not torch.allclose(out[0], out[1]), (
            "Different timesteps must produce different embeddings"
        )

    def test_same_timestep_same_embedding(self):
        te = TimestepEmbedding(d_model=D)
        t = torch.tensor([42, 42])
        out = te(t)
        torch.testing.assert_close(out[0], out[1])


# ---------------------------------------------------------------------------
# GraphTransformerBlock
# ---------------------------------------------------------------------------


class TestGraphTransformerBlock:
    """Tests for GraphTransformerBlock."""

    def test_output_shape(self):
        block = GraphTransformerBlock(d_model=D, n_heads=4, d_context=D_CTX, dropout=0.0)
        x = torch.randn(B, N, D)
        out = block(x, hdse_bias=None)
        assert out.shape == (B, N, D)

    def test_hdse_bias_changes_output(self):
        block = GraphTransformerBlock(d_model=D, n_heads=4, d_context=D_CTX, dropout=0.0)
        block.eval()
        x = torch.randn(B, N, D)
        out_no_bias = block(x, hdse_bias=None)
        bias = torch.randn(B, 4, N, N) * 0.5
        out_with_bias = block(x, hdse_bias=bias)
        assert not torch.allclose(out_no_bias, out_with_bias, atol=1e-5)

    def test_context_changes_output(self):
        block = GraphTransformerBlock(d_model=D, n_heads=4, d_context=D_CTX, dropout=0.0)
        block.eval()
        x = torch.randn(B, N, D)
        ctx, ctx_mask = _make_context()
        out_no_ctx = block(x, hdse_bias=None, context=None)
        out_with_ctx = block(x, hdse_bias=None, context=ctx, context_mask=ctx_mask)
        assert not torch.allclose(out_no_ctx, out_with_ctx, atol=1e-5)

    def test_with_node_mask(self):
        block = GraphTransformerBlock(d_model=D, n_heads=4, d_context=D_CTX, dropout=0.0)
        x = torch.randn(B, N, D)
        mask = torch.ones(B, N, dtype=torch.bool)
        mask[:, -2:] = False
        out = block(x, hdse_bias=None, node_mask=mask)
        assert out.shape == (B, N, D)


# ---------------------------------------------------------------------------
# GraphTransformerBackbone
# ---------------------------------------------------------------------------


class TestGraphTransformerBackbone:
    """Tests for GraphTransformerBackbone."""

    def test_epsilon_pred_shape(self):
        cfg = create_small_diffusion_config(use_hdse=False)
        backbone = GraphTransformerBackbone(cfg)
        x_t = torch.randn(B, N, 2)
        a_t = torch.rand(B, N, N)
        t = torch.tensor([5, 50])
        eps, _edge_log = backbone(x_t, a_t, t, hdse_output=None, context=None, node_mask=None)
        assert eps.shape == (B, N, 2)

    def test_edge_logits_shape(self):
        cfg = create_small_diffusion_config(use_hdse=False)
        backbone = GraphTransformerBackbone(cfg)
        x_t = torch.randn(B, N, 2)
        a_t = torch.rand(B, N, N)
        t = torch.tensor([5, 50])
        _, edge_log = backbone(x_t, a_t, t, hdse_output=None, context=None, node_mask=None)
        assert edge_log.shape == (B, N, N)

    def test_edge_logits_symmetric(self):
        cfg = create_small_diffusion_config(use_hdse=False)
        backbone = GraphTransformerBackbone(cfg)
        x_t = torch.randn(B, N, 2)
        a_t = torch.rand(B, N, N)
        t = torch.tensor([5, 50])
        _, edge_log = backbone(x_t, a_t, t, hdse_output=None, context=None, node_mask=None)
        torch.testing.assert_close(edge_log, edge_log.transpose(-2, -1))

    def test_edge_logits_zero_diagonal(self):
        cfg = create_small_diffusion_config(use_hdse=False)
        backbone = GraphTransformerBackbone(cfg)
        x_t = torch.randn(B, N, 2)
        a_t = torch.rand(B, N, N)
        t = torch.tensor([5, 50])
        _, edge_log = backbone(x_t, a_t, t, hdse_output=None, context=None, node_mask=None)
        for b in range(B):
            assert (edge_log[b].diag() == 0).all()

    def test_with_hdse_output(self):
        cfg = create_small_diffusion_config(use_hdse=True)
        backbone = GraphTransformerBackbone(cfg)
        hdse = HDSE(cfg)
        x_t = torch.randn(B, N, 2)
        a_t = torch.rand(B, N, N)
        t = torch.tensor([5, 50])
        hdse_out = hdse(a_t)
        eps, _edge_log = backbone(x_t, a_t, t, hdse_output=hdse_out, context=None, node_mask=None)
        assert eps.shape == (B, N, 2)

    def test_with_context(self):
        cfg = create_small_diffusion_config(use_hdse=False)
        backbone = GraphTransformerBackbone(cfg)
        x_t = torch.randn(B, N, 2)
        a_t = torch.rand(B, N, N)
        t = torch.tensor([5, 50])
        ctx, ctx_mask = _make_context()
        eps, _ = backbone(
            x_t,
            a_t,
            t,
            hdse_output=None,
            context=ctx,
            node_mask=None,
            context_mask=ctx_mask,
        )
        assert eps.shape == (B, N, 2)

    def test_node_mask_zeros_padded(self):
        cfg = create_small_diffusion_config(use_hdse=False)
        backbone = GraphTransformerBackbone(cfg)
        x_t = torch.randn(B, N, 2)
        a_t = torch.rand(B, N, N)
        t = torch.tensor([5, 50])
        mask = torch.ones(B, N, dtype=torch.bool)
        mask[:, -2:] = False
        eps, edge_log = backbone(x_t, a_t, t, hdse_output=None, context=None, node_mask=mask)
        assert (eps[:, -2:, :] == 0).all()
        assert (edge_log[:, -2:, :] == 0).all()
        assert (edge_log[:, :, -2:] == 0).all()


# ---------------------------------------------------------------------------
# ReverseDiffusion.denoise_step
# ---------------------------------------------------------------------------


class TestDenoiseStep:
    """Tests for ReverseDiffusion.denoise_step."""

    def test_output_shapes(self):
        cfg = create_small_diffusion_config(use_hdse=False)
        sched = DiffusionScheduler(num_timesteps=100)
        backbone = GraphTransformerBackbone(cfg)
        rev = ReverseDiffusion(backbone, sched)
        x_t = torch.randn(B, N, 2)
        a_t = (torch.rand(B, N, N) > 0.5).float()
        t = torch.tensor([50, 50])
        x_prev, a_prev = rev.denoise_step(x_t, a_t, t, None, None, None)
        assert x_prev.shape == (B, N, 2)
        assert a_prev.shape == (B, N, N)


# ---------------------------------------------------------------------------
# ReverseDiffusion.compute_vlb_loss
# ---------------------------------------------------------------------------


class TestComputeVlbLoss:
    """Tests for ReverseDiffusion.compute_vlb_loss."""

    def test_returns_diffusion_loss(self):
        cfg = create_small_diffusion_config(use_hdse=False)
        sched = DiffusionScheduler(num_timesteps=100)
        backbone = GraphTransformerBackbone(cfg)
        rev = ReverseDiffusion(backbone, sched)
        fwd = ForwardDiffusion(sched)
        x_0, a_0, mask = create_synthetic_graph(B, N)
        loss = rev.compute_vlb_loss(x_0, a_0, fwd, hdse=None, context=None, node_mask=mask)
        assert isinstance(loss, DiffusionLoss)
        assert loss.total.shape == ()

    def test_loss_differentiable(self):
        cfg = create_small_diffusion_config(use_hdse=False)
        sched = DiffusionScheduler(num_timesteps=100)
        backbone = GraphTransformerBackbone(cfg)
        rev = ReverseDiffusion(backbone, sched)
        fwd = ForwardDiffusion(sched)
        x_0, a_0, mask = create_synthetic_graph(B, N)
        loss = rev.compute_vlb_loss(x_0, a_0, fwd, hdse=None, context=None, node_mask=mask)
        loss.total.backward()
        # At least one parameter should have a gradient
        has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in backbone.parameters())
        assert has_grad, "Loss must be differentiable — expected gradients on backbone params"

    def test_loss_positive_for_random(self):
        cfg = create_small_diffusion_config(use_hdse=False)
        sched = DiffusionScheduler(num_timesteps=100)
        backbone = GraphTransformerBackbone(cfg)
        rev = ReverseDiffusion(backbone, sched)
        fwd = ForwardDiffusion(sched)
        x_0 = torch.randn(B, N, 2)
        a_0 = (torch.rand(B, N, N) > 0.5).float()
        loss = rev.compute_vlb_loss(x_0, a_0, fwd, hdse=None, context=None, node_mask=None)
        assert loss.total.item() > 0


# ---------------------------------------------------------------------------
# DiffusionLoss
# ---------------------------------------------------------------------------


class TestDiffusionLoss:
    """Tests for DiffusionLoss dataclass."""

    def test_fields_present(self):
        loss = DiffusionLoss(
            total=torch.tensor(1.0),
            coordinate_loss=torch.tensor(0.6),
            adjacency_loss=torch.tensor(0.4),
            t=torch.tensor([10, 50]),
        )
        torch.testing.assert_close(loss.total, torch.tensor(1.0))
        torch.testing.assert_close(loss.coordinate_loss, torch.tensor(0.6))
        torch.testing.assert_close(loss.adjacency_loss, torch.tensor(0.4))


# ---------------------------------------------------------------------------
# GraphDiffusionModel
# ---------------------------------------------------------------------------


class TestGraphDiffusionModel:
    """Tests for GraphDiffusionModel (top-level module)."""

    def test_forward_returns_loss(self):
        cfg = create_small_diffusion_config(use_hdse=False)
        model = GraphDiffusionModel(cfg)
        x_0, a_0, mask = create_synthetic_graph(B, N)
        loss = model(x_0, a_0, node_mask=mask)
        assert isinstance(loss, DiffusionLoss)
        assert loss.total.shape == ()
        assert torch.isfinite(loss.total)

    def test_forward_loss_backward(self):
        cfg = create_small_diffusion_config(use_hdse=False)
        model = GraphDiffusionModel(cfg)
        x_0, a_0, mask = create_synthetic_graph(B, N)
        loss = model(x_0, a_0, node_mask=mask)
        loss.total.backward()
        has_grad = any(p.grad is not None for p in model.parameters())
        assert has_grad

    def test_forward_with_context(self):
        cfg = create_small_diffusion_config(use_hdse=False)
        model = GraphDiffusionModel(cfg)
        x_0, a_0, mask = create_synthetic_graph(B, N)
        ctx, ctx_mask = _make_context()
        loss = model(x_0, a_0, context=ctx, node_mask=mask, context_mask=ctx_mask)
        assert torch.isfinite(loss.total)

    def test_forward_with_hdse(self):
        cfg = create_small_diffusion_config(use_hdse=True)
        model = GraphDiffusionModel(cfg)
        x_0, a_0, mask = create_synthetic_graph(B, N)
        loss = model(x_0, a_0, node_mask=mask)
        assert torch.isfinite(loss.total)

    def test_sample_returns_refined_graph(self):
        from docs.interfaces.diffusion_output import RefinedStructuralGraph

        cfg = create_small_diffusion_config(use_hdse=False)
        model = GraphDiffusionModel(cfg)
        model.eval()
        result = model.sample(num_nodes=N, batch_size=B, num_steps=5)
        assert isinstance(result, RefinedStructuralGraph)
        assert result.node_positions.shape == (B, N, 2)
        assert result.adjacency_logits.shape == (B, N, N)
        assert result.node_mask.shape == (B, N)
        assert result.node_features.shape == (B, N, cfg.d_model)

    def test_sample_edge_index_shape(self):
        cfg = create_small_diffusion_config(use_hdse=False)
        model = GraphDiffusionModel(cfg)
        model.eval()
        result = model.sample(num_nodes=N, batch_size=1, num_steps=5)
        assert result.edge_index.shape[0] == 2  # (2, E)
        assert result.edge_index.dtype == torch.long

    def test_ddim_deterministic_eta0(self):
        """eta=0 DDIM should be deterministic given fixed noise seed."""
        cfg = create_small_diffusion_config(use_hdse=False)
        model = GraphDiffusionModel(cfg)
        model.eval()
        torch.manual_seed(123)
        r1 = model.sample(num_nodes=N, batch_size=1, num_steps=5)
        torch.manual_seed(123)
        r2 = model.sample(num_nodes=N, batch_size=1, num_steps=5)
        torch.testing.assert_close(r1.node_positions, r2.node_positions)

    def test_ddim_5_steps_works(self):
        cfg = create_small_diffusion_config(use_hdse=False)
        model = GraphDiffusionModel(cfg)
        model.eval()
        result = model.sample(num_nodes=N, batch_size=1, num_steps=5)
        assert torch.isfinite(result.node_positions).all()
