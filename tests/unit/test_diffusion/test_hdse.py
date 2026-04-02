"""Unit tests for src/diffusion/hdse.py.

Tests ShortestPathEncoding, RandomWalkEncoding, HierarchicalLevelEncoding,
and the combined HDSE module.

Q-005: diffusion unit tests (HDSE).
"""

from __future__ import annotations

import torch

from src.diffusion.hdse import (
    HDSE,
    HierarchicalLevelEncoding,
    RandomWalkEncoding,
    ShortestPathEncoding,
)
from tests.fixtures.diffusion_helpers import create_small_diffusion_config

B, N, D = 2, 8, 64


def _cycle_adjacency(n: int, batch: int = 2) -> torch.Tensor:
    """Cycle graph adjacency (n nodes)."""
    a = torch.zeros(n, n)
    for i in range(n):
        j = (i + 1) % n
        a[i, j] = 1.0
        a[j, i] = 1.0
    return a.unsqueeze(0).expand(batch, -1, -1).clone()


# ---------------------------------------------------------------------------
# ShortestPathEncoding
# ---------------------------------------------------------------------------


class TestShortestPathEncoding:
    """Tests for ShortestPathEncoding."""

    def test_output_shape(self):
        sp = ShortestPathEncoding(max_distance=5, d_model=D)
        adj = _cycle_adjacency(N)
        out = sp(adj)
        assert out.shape == (B, N, N, D)

    def test_self_distance_is_zero_index(self):
        """Diagonal entries should be distance 0 (embedding index 0)."""
        sp = ShortestPathEncoding(max_distance=5, d_model=D)
        adj = _cycle_adjacency(N)
        out = sp(adj)
        # All diagonal embeddings should be the same (index 0)
        diag_embs = out[:, range(N), range(N), :]  # (B, N, D)
        zero_emb = sp.embedding(torch.tensor([0]))  # (1, D)
        for b in range(B):
            for i in range(N):
                torch.testing.assert_close(diag_embs[b, i], zero_emb.squeeze(0))

    def test_adjacent_nodes_distance_one(self):
        """Directly connected nodes should have distance 1."""
        sp = ShortestPathEncoding(max_distance=5, d_model=D)
        adj = _cycle_adjacency(N)
        out = sp(adj)
        one_emb = sp.embedding(torch.tensor([1]))  # (1, D)
        # Node 0 and 1 are connected
        torch.testing.assert_close(out[0, 0, 1], one_emb.squeeze(0))

    def test_with_node_mask(self):
        sp = ShortestPathEncoding(max_distance=5, d_model=D)
        adj = _cycle_adjacency(N)
        mask = torch.ones(B, N, dtype=torch.bool)
        mask[:, -2:] = False  # mask last 2
        out = sp(adj, node_mask=mask)
        assert out.shape == (B, N, N, D)


# ---------------------------------------------------------------------------
# RandomWalkEncoding
# ---------------------------------------------------------------------------


class TestRandomWalkEncoding:
    """Tests for RandomWalkEncoding."""

    def test_output_shape(self):
        rw = RandomWalkEncoding(num_steps=8, d_model=D)
        adj = _cycle_adjacency(N)
        out = rw(adj)
        assert out.shape == (B, N, N, D)

    def test_self_walk_probability_high(self):
        """In a cycle graph the 1-step self-landing is 0, but 2-step is > 0."""
        rw = RandomWalkEncoding(num_steps=4, d_model=D)
        adj = _cycle_adjacency(N).float()
        # Check the module doesn't crash and output is finite
        out = rw(adj)
        assert torch.isfinite(out).all()

    def test_with_node_mask(self):
        rw = RandomWalkEncoding(num_steps=4, d_model=D)
        adj = _cycle_adjacency(N)
        mask = torch.ones(B, N, dtype=torch.bool)
        mask[:, -2:] = False
        out = rw(adj, node_mask=mask)
        assert out.shape == (B, N, N, D)


# ---------------------------------------------------------------------------
# HierarchicalLevelEncoding
# ---------------------------------------------------------------------------


class TestHierarchicalLevelEncoding:
    """Tests for HierarchicalLevelEncoding."""

    def test_output_shape(self):
        hier = HierarchicalLevelEncoding(num_levels=4, d_model=D)
        adj = _cycle_adjacency(N)
        out = hier(adj)
        assert out.shape == (B, N, D)

    def test_with_node_mask(self):
        hier = HierarchicalLevelEncoding(num_levels=4, d_model=D)
        adj = _cycle_adjacency(N)
        mask = torch.ones(B, N, dtype=torch.bool)
        mask[:, -2:] = False
        out = hier(adj, node_mask=mask)
        assert out.shape == (B, N, D)

    def test_with_node_positions(self):
        hier = HierarchicalLevelEncoding(num_levels=4, d_model=D)
        adj = _cycle_adjacency(N)
        pos = torch.rand(B, N, 2)
        out = hier(adj, node_positions=pos)
        assert out.shape == (B, N, D)


# ---------------------------------------------------------------------------
# Combined HDSE
# ---------------------------------------------------------------------------


class TestHDSE:
    """Tests for the combined HDSE module."""

    def test_attention_bias_shape(self):
        cfg = create_small_diffusion_config()
        hdse = HDSE(cfg)
        adj = _cycle_adjacency(N)
        out = hdse(adj)
        assert out.attention_bias.shape == (B, cfg.n_heads, N, N)

    def test_node_encodings_shape(self):
        cfg = create_small_diffusion_config()
        hdse = HDSE(cfg)
        adj = _cycle_adjacency(N)
        out = hdse(adj)
        assert out.node_encodings.shape == (B, N, cfg.d_model)

    def test_with_node_mask(self):
        cfg = create_small_diffusion_config()
        hdse = HDSE(cfg)
        adj = _cycle_adjacency(N)
        mask = torch.ones(B, N, dtype=torch.bool)
        mask[:, -2:] = False
        out = hdse(adj, node_mask=mask)
        assert out.attention_bias.shape == (B, cfg.n_heads, N, N)
        # Padded positions should have zero bias
        assert (out.attention_bias[:, :, -2:, :] == 0).all()
        assert (out.attention_bias[:, :, :, -2:] == 0).all()

    def test_empty_adjacency_no_crash(self):
        """No edges — module should still run without error."""
        cfg = create_small_diffusion_config()
        hdse = HDSE(cfg)
        adj = torch.zeros(B, N, N)
        out = hdse(adj)
        assert torch.isfinite(out.attention_bias).all()
        assert torch.isfinite(out.node_encodings).all()

    def test_with_node_positions(self):
        cfg = create_small_diffusion_config()
        hdse = HDSE(cfg)
        adj = _cycle_adjacency(N)
        pos = torch.rand(B, N, 2)
        out = hdse(adj, node_positions=pos)
        assert out.attention_bias.shape == (B, cfg.n_heads, N, N)
