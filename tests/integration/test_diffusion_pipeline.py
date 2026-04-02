"""Integration tests for the diffusion pipeline.

End-to-end: synthetic clean graph → model.forward() (training loss)
           → model.sample() (DDIM inference) → RefinedStructuralGraph.

Q-006: diffusion integration tests.
"""

from __future__ import annotations

import pytest
import torch

from docs.interfaces.diffusion_output import RefinedStructuralGraph
from src.diffusion.reverse import DiffusionLoss, GraphDiffusionModel
from tests.fixtures.diffusion_helpers import (
    create_small_diffusion_config,
    create_synthetic_graph,
)

B, N = 2, 8
N_CTX, D_CTX = 6, 256


@pytest.fixture(scope="module")
def diffusion_model() -> GraphDiffusionModel:
    """Module-scoped model to avoid repeated init overhead."""
    cfg = create_small_diffusion_config(use_hdse=True)
    model = GraphDiffusionModel(cfg)
    model.eval()
    return model


@pytest.fixture(scope="module")
def clean_graph() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Module-scoped clean graph fixture."""
    return create_synthetic_graph(B, N)


@pytest.fixture(scope="module")
def context_tensors() -> tuple[torch.Tensor, torch.Tensor]:
    """Synthetic cross-modal context matching tokenizer output shape."""
    torch.manual_seed(0)
    ctx = torch.randn(B, N_CTX, D_CTX)
    ctx_mask = torch.ones(B, N_CTX, dtype=torch.bool)
    return ctx, ctx_mask


# ---------------------------------------------------------------------------
# Training forward pass
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestDiffusionTrainingPipeline:
    """Integration: clean graph → model.forward() → DiffusionLoss."""

    def test_forward_produces_loss(self, diffusion_model, clean_graph, context_tensors):
        model = diffusion_model
        model.train()
        x_0, a_0, mask = clean_graph
        ctx, ctx_mask = context_tensors

        loss = model(x_0, a_0, context=ctx, node_mask=mask, context_mask=ctx_mask)
        assert isinstance(loss, DiffusionLoss)
        assert loss.total.shape == ()
        assert torch.isfinite(loss.total)
        assert loss.coordinate_loss.item() >= 0
        assert loss.adjacency_loss.item() >= 0

    def test_loss_backward_succeeds(self, clean_graph, context_tensors):
        # Fresh model so grads accumulate cleanly
        cfg = create_small_diffusion_config(use_hdse=True)
        model = GraphDiffusionModel(cfg)
        x_0, a_0, mask = clean_graph
        ctx, ctx_mask = context_tensors

        loss = model(x_0, a_0, context=ctx, node_mask=mask, context_mask=ctx_mask)
        loss.total.backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert len(grads) > 0, "Expected gradients after backward"
        assert all(torch.isfinite(g).all() for g in grads), "Gradients must be finite"


# ---------------------------------------------------------------------------
# DDIM sampling / inference
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestDiffusionSamplingPipeline:
    """Integration: model.sample() → RefinedStructuralGraph with valid fields."""

    def test_sample_returns_refined_graph(self, diffusion_model, context_tensors):
        model = diffusion_model
        model.eval()
        ctx, ctx_mask = context_tensors

        result = model.sample(
            num_nodes=N,
            batch_size=B,
            num_steps=5,
            context=ctx,
            context_mask=ctx_mask,
        )
        assert isinstance(result, RefinedStructuralGraph)

    def test_node_positions_shape_and_range(self, diffusion_model, context_tensors):
        model = diffusion_model
        model.eval()
        ctx, ctx_mask = context_tensors

        result = model.sample(
            num_nodes=N,
            batch_size=B,
            num_steps=5,
            context=ctx,
            context_mask=ctx_mask,
        )
        assert result.node_positions.shape == (B, N, 2)
        assert torch.isfinite(result.node_positions).all()
        # Untrained model won't produce [0,1] positions, but values should
        # be finite (not NaN/Inf). Magnitude can be large with random weights.
        assert result.node_positions.abs().max() < 1e6, "Positions should not explode to inf"

    def test_adjacency_logits_shape(self, diffusion_model, context_tensors):
        model = diffusion_model
        model.eval()
        ctx, ctx_mask = context_tensors

        result = model.sample(
            num_nodes=N,
            batch_size=B,
            num_steps=5,
            context=ctx,
            context_mask=ctx_mask,
        )
        assert result.adjacency_logits.shape == (B, N, N)
        assert torch.isfinite(result.adjacency_logits).all()

    def test_node_mask_shape(self, diffusion_model, context_tensors):
        model = diffusion_model
        model.eval()
        ctx, ctx_mask = context_tensors

        result = model.sample(
            num_nodes=N,
            batch_size=B,
            num_steps=5,
            context=ctx,
            context_mask=ctx_mask,
        )
        assert result.node_mask.shape == (B, N)
        assert result.node_mask.dtype == torch.bool
        assert result.node_mask.all(), "Default mask should be all-True"

    def test_edge_index_format(self, diffusion_model, context_tensors):
        model = diffusion_model
        model.eval()
        ctx, ctx_mask = context_tensors

        result = model.sample(
            num_nodes=N,
            batch_size=B,
            num_steps=5,
            context=ctx,
            context_mask=ctx_mask,
        )
        assert result.edge_index.shape[0] == 2
        assert result.edge_index.dtype == torch.long

    def test_node_features_shape(self, diffusion_model, context_tensors):
        model = diffusion_model
        model.eval()
        ctx, ctx_mask = context_tensors

        result = model.sample(
            num_nodes=N,
            batch_size=B,
            num_steps=5,
            context=ctx,
            context_mask=ctx_mask,
        )
        cfg = create_small_diffusion_config()
        assert result.node_features.shape == (B, N, cfg.d_model)

    def test_junction_types_populated(self, diffusion_model, context_tensors):
        model = diffusion_model
        model.eval()
        ctx, ctx_mask = context_tensors

        result = model.sample(
            num_nodes=N,
            batch_size=B,
            num_steps=5,
            context=ctx,
            context_mask=ctx_mask,
        )
        assert len(result.junction_types) == B
        for batch_jt in result.junction_types:
            assert len(batch_jt) == N

    def test_context_embeddings_passed_through(self, diffusion_model, context_tensors):
        model = diffusion_model
        model.eval()
        ctx, ctx_mask = context_tensors

        result = model.sample(
            num_nodes=N,
            batch_size=B,
            num_steps=5,
            context=ctx,
            context_mask=ctx_mask,
        )
        assert result.context_embeddings is not None
        torch.testing.assert_close(result.context_embeddings, ctx)

    def test_sample_without_context(self, diffusion_model):
        """Sampling works without cross-modal context."""
        model = diffusion_model
        model.eval()
        result = model.sample(num_nodes=N, batch_size=1, num_steps=5)
        assert isinstance(result, RefinedStructuralGraph)
        assert torch.isfinite(result.node_positions).all()
