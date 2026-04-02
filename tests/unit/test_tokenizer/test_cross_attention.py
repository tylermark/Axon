"""Unit tests for src/tokenizer/cross_attention.py.

Tests VisionToVectorAttention, VectorToVisionAttention, SpatialAttentionWindow,
TokenizedEarlyFusion, VectorOnlyFallback, and Tokenizer (top-level).

Q-003: tokenizer unit tests (cross-attention and fusion path).
"""

from __future__ import annotations

import pytest
import torch

from src.pipeline.config import TokenizerConfig
from src.tokenizer.cross_attention import (
    SpatialAttentionWindow,
    TokenizedEarlyFusion,
    Tokenizer,
    VectorOnlyFallback,
    VectorToVisionAttention,
    VisionToVectorAttention,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_raw_features(
    batch_size: int = 1,
    seq_len: int = 10,
) -> dict[str, torch.Tensor]:
    """Create dummy raw feature tensors for Tokenizer.forward()."""
    return {
        "operator_type": torch.randint(0, 4, (batch_size, seq_len)),
        "coordinates": torch.rand(batch_size, seq_len, 4),
        "stroke_width": torch.rand(batch_size, seq_len),
        "dash_hash": torch.randint(0, 64, (batch_size, seq_len)),
        "color_rgb": torch.rand(batch_size, seq_len, 3),
        "confidence_wall": torch.rand(batch_size, seq_len),
        "attention_mask": torch.ones(batch_size, seq_len, dtype=torch.bool),
        "raw_coordinates": torch.rand(batch_size, seq_len, 4).double(),
    }


# ---------------------------------------------------------------------------
# VisionToVectorAttention
# ---------------------------------------------------------------------------


class TestVisionToVectorAttention:
    """Tests for VisionToVectorAttention (T-004)."""

    def test_output_shape(self):
        attn = VisionToVectorAttention(d_model=64, n_heads=4, dropout=0.0)
        vec = torch.randn(2, 10, 64)
        vis = torch.randn(2, 20, 64)
        out, _weights = attn(vec, vis)
        assert out.shape == (2, 10, 64)

    def test_attention_weights_shape(self):
        attn = VisionToVectorAttention(d_model=64, n_heads=4, dropout=0.0)
        vec = torch.randn(1, 5, 64)
        vis = torch.randn(1, 8, 64)
        _out, weights = attn(vec, vis)
        # average_attn_weights=False → (B, n_heads, N_q, N_kv)
        assert weights.shape == (1, 4, 5, 8)

    def test_with_spatial_attention_mask(self):
        attn = VisionToVectorAttention(d_model=64, n_heads=4, dropout=0.0)
        vec = torch.randn(1, 5, 64)
        vis = torch.randn(1, 8, 64)
        # attn_mask: (B*n_heads, N_q, N_kv) where True = masked out
        mask = torch.zeros(4, 5, 8, dtype=torch.bool)
        mask[:, :, 4:] = True  # mask out last 4 visual features
        out, _ = attn(vec, vis, attn_mask=mask)
        assert out.shape == (1, 5, 64)


# ---------------------------------------------------------------------------
# VectorToVisionAttention
# ---------------------------------------------------------------------------


class TestVectorToVisionAttention:
    """Tests for VectorToVisionAttention (T-005)."""

    def test_output_shape(self):
        attn = VectorToVisionAttention(d_model=64, n_heads=4, dropout=0.0)
        vis = torch.randn(2, 20, 64)
        vec = torch.randn(2, 10, 64)
        out, _weights = attn(vis, vec)
        assert out.shape == (2, 20, 64)

    def test_attention_weights_shape(self):
        attn = VectorToVisionAttention(d_model=64, n_heads=4, dropout=0.0)
        vis = torch.randn(1, 8, 64)
        vec = torch.randn(1, 5, 64)
        _out, weights = attn(vis, vec)
        # default average_attn_weights=True for this module → (B, N_q, N_kv)
        assert weights is not None


# ---------------------------------------------------------------------------
# SpatialAttentionWindow
# ---------------------------------------------------------------------------


class TestSpatialAttentionWindow:
    """Tests for SpatialAttentionWindow (T-007)."""

    def test_mask_blocks_features_outside_radius(self):
        window = SpatialAttentionWindow(attention_radius_fraction=0.1)
        token_pos = torch.tensor([[[0.0, 0.0]]])
        visual_pos = torch.tensor([[[0.0, 0.0], [0.5, 0.5]]])
        mask = window.compute_mask(token_pos, visual_pos)
        # (0,0)→(0,0): dist=0 < 0.1 → not masked
        assert not mask[0, 0, 0].item()
        # (0,0)→(0.5,0.5): dist≈0.707 > 0.1 → masked
        assert mask[0, 0, 1].item()

    def test_mask_allows_features_inside_radius(self):
        window = SpatialAttentionWindow(attention_radius_fraction=0.2)
        token_pos = torch.tensor([[[0.5, 0.5]]])
        visual_pos = torch.tensor([[[0.5, 0.5], [0.55, 0.55], [0.6, 0.6]]])
        mask = window.compute_mask(token_pos, visual_pos)
        # (0.5,0.5)→(0.5,0.5): dist=0 → not masked
        assert not mask[0, 0, 0].item()
        # (0.5,0.5)→(0.55,0.55): dist≈0.071 < 0.2 → not masked
        assert not mask[0, 0, 1].item()
        # (0.5,0.5)→(0.6,0.6): dist≈0.141 < 0.2 → not masked
        assert not mask[0, 0, 2].item()

    def test_large_radius_unmasks_nearby_positions(self):
        window = SpatialAttentionWindow(attention_radius_fraction=1.0)
        # All positions in [0, 0.5] — max dist = sqrt(0.5) ≈ 0.707 < 1.0
        token_pos = torch.rand(1, 5, 2) * 0.5
        visual_pos = torch.rand(1, 10, 2) * 0.5
        mask = window.compute_mask(token_pos, visual_pos)
        assert not mask.any()

    def test_zero_radius_masks_distinct_positions(self):
        window = SpatialAttentionWindow(attention_radius_fraction=0.0)
        token_pos = torch.tensor([[[0.1, 0.2], [0.5, 0.5]]])
        visual_pos = torch.tensor([[[0.3, 0.4], [0.7, 0.8]]])
        mask = window.compute_mask(token_pos, visual_pos)
        # r_sq=0 → any non-zero distance is masked
        assert mask.all()

    def test_mask_shape(self):
        window = SpatialAttentionWindow(attention_radius_fraction=0.05)
        token_pos = torch.rand(3, 10, 2)
        visual_pos = torch.rand(3, 20, 2)
        mask = window.compute_mask(token_pos, visual_pos)
        assert mask.shape == (3, 10, 20)
        assert mask.dtype == torch.bool


# ---------------------------------------------------------------------------
# TokenizedEarlyFusion
# ---------------------------------------------------------------------------


class TestTokenizedEarlyFusion:
    """Tests for TokenizedEarlyFusion (T-006)."""

    def test_output_shape(self):
        tef = TokenizedEarlyFusion(d_model=64, n_heads=4, n_layers=1, dropout=0.0)
        vec = torch.randn(1, 10, 64)
        vis = torch.randn(1, 20, 64)
        out, _weights = tef(vec, vis)
        assert out.shape == (1, 10, 64)

    def test_with_multiple_layers(self):
        tef = TokenizedEarlyFusion(d_model=64, n_heads=4, n_layers=2, dropout=0.0)
        vec = torch.randn(1, 8, 64)
        vis = torch.randn(1, 16, 64)
        out, _weights = tef(vec, vis)
        assert out.shape == (1, 8, 64)

    def test_with_spatial_mask(self):
        tef = TokenizedEarlyFusion(d_model=64, n_heads=4, n_layers=1, dropout=0.0)
        vec = torch.randn(1, 5, 64)
        vis = torch.randn(1, 8, 64)
        spatial_mask = torch.zeros(1, 5, 8, dtype=torch.bool)
        spatial_mask[:, :, 4:] = True  # mask out last 4 visual features
        out, _weights = tef(vec, vis, spatial_mask=spatial_mask)
        assert out.shape == (1, 5, 64)

    def test_with_token_padding_mask(self):
        tef = TokenizedEarlyFusion(d_model=64, n_heads=4, n_layers=1, dropout=0.0)
        vec = torch.randn(1, 8, 64)
        vis = torch.randn(1, 10, 64)
        padding_mask = torch.tensor([[True] * 5 + [False] * 3])  # 5 valid + 3 padding
        out, _ = tef(vec, vis, token_padding_mask=padding_mask)
        assert out.shape == (1, 8, 64)


# ---------------------------------------------------------------------------
# VectorOnlyFallback
# ---------------------------------------------------------------------------


class TestVectorOnlyFallback:
    """Tests for VectorOnlyFallback (T-008)."""

    def test_output_shape_matches_input(self):
        fallback = VectorOnlyFallback(d_model=64, n_heads=4, n_layers=1, dropout=0.0)
        tokens = torch.randn(2, 10, 64)
        out = fallback(tokens)
        assert out.shape == tokens.shape

    def test_with_attention_mask(self):
        fallback = VectorOnlyFallback(d_model=64, n_heads=4, n_layers=1, dropout=0.0)
        tokens = torch.randn(1, 8, 64)
        mask = torch.tensor([[True] * 5 + [False] * 3])
        out = fallback(tokens, attention_mask=mask)
        assert out.shape == (1, 8, 64)

    def test_multiple_layers(self):
        fallback = VectorOnlyFallback(d_model=64, n_heads=4, n_layers=3, dropout=0.0)
        tokens = torch.randn(1, 6, 64)
        out = fallback(tokens)
        assert out.shape == (1, 6, 64)


# ---------------------------------------------------------------------------
# Tokenizer (top-level) — slow tests, loads timm VisionBackbone
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestTokenizer:
    """Tests for the top-level Tokenizer module."""

    @pytest.fixture(scope="class")
    def tokenizer(self):
        from src.pipeline.config import VisionBackbone as VisionBackboneType

        config = TokenizerConfig(
            d_model=128,
            n_heads=4,
            vision_backbone=VisionBackboneType.HRNET_W32,
            dropout=0.0,
        )
        tok = Tokenizer(config, n_tef_layers=1)
        tok.eval()
        return tok

    def test_forward_with_images(self, tokenizer):
        features = _make_raw_features(batch_size=1, seq_len=10)
        images = torch.randn(1, 3, 256, 256)
        with torch.no_grad():
            result = tokenizer(features, images=images)
        assert result.token_embeddings.shape == (1, 10, 128)
        assert result.attention_mask.shape == (1, 10)
        assert result.is_vector_only is False
        assert result.raster_features is not None

    def test_forward_without_images_vector_only(self, tokenizer):
        features = _make_raw_features(batch_size=1, seq_len=10)
        with torch.no_grad():
            result = tokenizer(features, images=None)
        assert result.token_embeddings.shape == (1, 10, 128)
        assert result.is_vector_only is True
        assert result.raster_features is None

    def test_all_output_shapes_consistent(self, tokenizer):
        features = _make_raw_features(batch_size=2, seq_len=8)
        with torch.no_grad():
            result = tokenizer(features, images=None)
        assert result.token_embeddings.shape == (2, 8, 128)
        assert result.attention_mask.shape == (2, 8)
        assert result.position_encodings.shape == (2, 8, 128)
        assert result.raw_coordinates.shape == (2, 8, 4)
        assert result.confidence_wall.shape == (2, 8)
        assert result.edge_indices is not None

    def test_raster_features_none_in_vector_only_mode(self, tokenizer):
        features = _make_raw_features(batch_size=1, seq_len=5)
        with torch.no_grad():
            result = tokenizer(features, images=None)
        assert result.raster_features is None
        assert result.vision_to_vector_weights is None
