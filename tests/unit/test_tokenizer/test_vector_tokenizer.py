"""Unit tests for src/tokenizer/vector_tokenizer.py.

Tests VectorTokenEmbedding, LearnedPositionalEncoding2D, VectorTokenizer,
graph_to_token_features(), and collate_graphs().

Q-003: tokenizer unit tests (vector tokenization path).
"""

from __future__ import annotations

import math

import pytest
import torch

from src.pipeline.config import TokenizerConfig
from src.tokenizer.vector_tokenizer import (
    LearnedPositionalEncoding2D,
    VectorTokenEmbedding,
    VectorTokenizer,
    collate_graphs,
    graph_to_token_features,
)
from tests.fixtures.graph_factory import create_synthetic_raw_graph

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_raw_features(
    batch_size: int = 1,
    seq_len: int = 10,
) -> dict[str, torch.Tensor]:
    """Create dummy raw feature tensors compatible with VectorTokenEmbedding."""
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
# VectorTokenEmbedding
# ---------------------------------------------------------------------------


class TestVectorTokenEmbedding:
    """Tests for VectorTokenEmbedding (T-001)."""

    def test_output_shape_valid_input(self):
        emb = VectorTokenEmbedding(d_model=64, d_operator=16, d_dash=16)
        features = _make_raw_features(batch_size=2, seq_len=8)
        out = emb(features)
        assert out.shape == (2, 8, 64)
        assert out.dtype == torch.float32

    def test_different_operator_types_produce_different_embeddings(self):
        emb = VectorTokenEmbedding(d_model=64, d_operator=16, d_dash=16)
        features = _make_raw_features(batch_size=1, seq_len=2)
        features["operator_type"] = torch.tensor([[0, 1]])  # moveto vs lineto
        out = emb(features)
        assert not torch.allclose(out[0, 0], out[0, 1])

    def test_different_coordinates_produce_different_embeddings(self):
        emb = VectorTokenEmbedding(d_model=64, d_operator=16, d_dash=16)
        f1 = _make_raw_features(batch_size=1, seq_len=1)
        f2 = {k: v.clone() for k, v in f1.items()}
        f1["coordinates"] = torch.tensor([[[0.1, 0.2, 0.3, 0.4]]])
        f2["coordinates"] = torch.tensor([[[0.9, 0.8, 0.7, 0.6]]])
        out1 = emb(f1)
        out2 = emb(f2)
        assert not torch.allclose(out1, out2)

    def test_batch_size_one(self):
        emb = VectorTokenEmbedding(d_model=64, d_operator=16, d_dash=16)
        features = _make_raw_features(batch_size=1, seq_len=5)
        out = emb(features)
        assert out.shape == (1, 5, 64)

    def test_batch_size_multiple(self):
        emb = VectorTokenEmbedding(d_model=64, d_operator=16, d_dash=16)
        features = _make_raw_features(batch_size=4, seq_len=5)
        out = emb(features)
        assert out.shape == (4, 5, 64)

    @pytest.mark.parametrize("d_model", [96, 128, 256])
    def test_respects_d_model(self, d_model: int):
        emb = VectorTokenEmbedding(d_model=d_model, d_operator=16, d_dash=16)
        features = _make_raw_features(batch_size=1, seq_len=3)
        out = emb(features)
        assert out.shape[-1] == d_model


# ---------------------------------------------------------------------------
# LearnedPositionalEncoding2D
# ---------------------------------------------------------------------------


class TestLearnedPositionalEncoding2D:
    """Tests for LearnedPositionalEncoding2D (T-002)."""

    def test_output_shape(self):
        pe = LearnedPositionalEncoding2D(d_model=64, dropout=0.0)
        coords = torch.rand(2, 10, 4)
        out = pe(coords)
        assert out.shape == (2, 10, 64)
        assert out.dtype == torch.float32

    def test_different_positions_produce_different_encodings(self):
        pe = LearnedPositionalEncoding2D(d_model=64, dropout=0.0)
        coords = torch.zeros(1, 2, 4)
        coords[0, 0] = torch.tensor([0.0, 0.0, 0.0, 0.0])  # midpoint (0, 0)
        coords[0, 1] = torch.tensor([1.0, 1.0, 1.0, 1.0])  # midpoint (1, 1)
        out = pe(coords)
        assert not torch.allclose(out[0, 0], out[0, 1])

    def test_deterministic_for_same_input(self):
        pe = LearnedPositionalEncoding2D(d_model=64, dropout=0.0)
        pe.eval()
        coords = torch.rand(1, 5, 4)
        out1 = pe(coords)
        out2 = pe(coords)
        torch.testing.assert_close(out1, out2)


# ---------------------------------------------------------------------------
# VectorTokenizer (combined)
# ---------------------------------------------------------------------------


class TestVectorTokenizer:
    """Tests for VectorTokenizer — embedding + positional encoding sum."""

    def test_output_shape(self):
        config = TokenizerConfig(d_model=128, n_heads=4, dropout=0.0)
        vt = VectorTokenizer(config)
        features = _make_raw_features(batch_size=2, seq_len=8)
        out = vt(features)
        assert out.shape == (2, 8, 128)
        assert out.dtype == torch.float32

    def test_output_shape_matches_components(self):
        """Output shape must be consistent with embedding + position encoding."""
        config = TokenizerConfig(d_model=128, n_heads=4, dropout=0.0)
        vt = VectorTokenizer(config)
        vt.eval()
        features = _make_raw_features(batch_size=1, seq_len=5)
        out = vt(features)
        emb_out = vt.embedding(features)
        pos_out = vt.positional_encoding(features["coordinates"])
        assert out.shape == emb_out.shape == pos_out.shape


# ---------------------------------------------------------------------------
# graph_to_token_features
# ---------------------------------------------------------------------------


class TestGraphToTokenFeatures:
    """Tests for graph_to_token_features()."""

    def test_output_has_all_expected_keys(self):
        graph = create_synthetic_raw_graph(num_edges=5)
        features = graph_to_token_features(graph)
        expected_keys = {
            "operator_type",
            "coordinates",
            "stroke_width",
            "dash_hash",
            "color_rgb",
            "confidence_wall",
            "attention_mask",
            "raw_coordinates",
        }
        assert set(features.keys()) == expected_keys

    def test_coordinates_normalized_to_unit_range(self):
        graph = create_synthetic_raw_graph(num_edges=5)
        features = graph_to_token_features(graph)
        mask = features["attention_mask"]
        valid_coords = features["coordinates"][mask]
        assert valid_coords.min() >= 0.0
        assert valid_coords.max() <= 1.0

    def test_stroke_width_normalized_by_page_diagonal(self):
        graph = create_synthetic_raw_graph(num_edges=5, page_width=612.0, page_height=792.0)
        features = graph_to_token_features(graph)
        page_diag = math.sqrt(612.0**2 + 792.0**2)
        expected_sw = 1.5 / page_diag  # factory uses stroke_width=1.5
        mask = features["attention_mask"]
        valid_sw = features["stroke_width"][mask]
        torch.testing.assert_close(
            valid_sw,
            torch.full_like(valid_sw, expected_sw),
        )

    def test_attention_mask_shape_no_padding(self):
        graph = create_synthetic_raw_graph(num_edges=7)
        features = graph_to_token_features(graph)
        assert features["attention_mask"].shape == (7,)
        assert features["attention_mask"].all()

    def test_max_tokens_padding(self):
        graph = create_synthetic_raw_graph(num_edges=5)
        features = graph_to_token_features(graph, max_tokens=10)
        assert features["operator_type"].shape == (10,)
        assert features["coordinates"].shape == (10, 4)
        assert features["attention_mask"][:5].all()
        assert not features["attention_mask"][5:].any()

    def test_max_tokens_truncation(self):
        graph = create_synthetic_raw_graph(num_edges=10)
        features = graph_to_token_features(graph, max_tokens=5)
        assert features["operator_type"].shape == (5,)
        assert features["attention_mask"].all()

    def test_raw_coordinates_preserved_in_pdf_units(self):
        graph = create_synthetic_raw_graph(num_edges=3)
        features = graph_to_token_features(graph)
        raw = features["raw_coordinates"]
        assert raw.dtype == torch.float64
        assert raw.shape == (3, 4)
        # Raw coords should not be in [0, 1] — they're in PDF user units
        mask = features["attention_mask"]
        valid_raw = raw[mask]
        # At least some coordinates should be > 1 (page is 612x792)
        assert valid_raw.abs().max() > 1.0


# ---------------------------------------------------------------------------
# collate_graphs
# ---------------------------------------------------------------------------


class TestCollateGraphs:
    """Tests for collate_graphs() — batching with padding."""

    def test_batches_different_sizes_with_padding(self):
        g1 = create_synthetic_raw_graph(num_edges=3, seed=1)
        g2 = create_synthetic_raw_graph(num_edges=7, seed=2)
        batch = collate_graphs([g1, g2])
        assert batch["operator_type"].shape == (2, 7)
        assert batch["coordinates"].shape == (2, 7, 4)

    def test_attention_mask_marks_padding_correctly(self):
        g1 = create_synthetic_raw_graph(num_edges=3, seed=1)
        g2 = create_synthetic_raw_graph(num_edges=7, seed=2)
        batch = collate_graphs([g1, g2])
        # g1: 3 valid, 4 padded
        assert batch["attention_mask"][0, :3].all()
        assert not batch["attention_mask"][0, 3:].any()
        # g2: 7 valid
        assert batch["attention_mask"][1].all()

    def test_batch_dimension_correct(self):
        graphs = [create_synthetic_raw_graph(num_edges=5, seed=i) for i in range(4)]
        batch = collate_graphs(graphs)
        assert batch["operator_type"].shape[0] == 4

    def test_empty_list_raises_value_error(self):
        with pytest.raises(ValueError, match="empty"):
            collate_graphs([])

    def test_max_tokens_applied(self):
        g1 = create_synthetic_raw_graph(num_edges=10, seed=1)
        g2 = create_synthetic_raw_graph(num_edges=15, seed=2)
        batch = collate_graphs([g1, g2], max_tokens=8)
        assert batch["operator_type"].shape == (2, 8)
        assert batch["attention_mask"].all()  # both truncated, all valid
