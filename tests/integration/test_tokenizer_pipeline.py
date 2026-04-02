"""Integration test: RawGraph -> EnrichedTokenSequence end-to-end.

Q-004: Full tokenizer pipeline integration test covering both
raster-fused and vector-only fallback paths.
"""

from __future__ import annotations

import pytest
import torch

from docs.interfaces.tokenizer_to_diffusion import EnrichedTokenSequence
from src.pipeline.config import TokenizerConfig
from src.pipeline.config import VisionBackbone as VisionBackboneType
from src.tokenizer.cross_attention import Tokenizer
from src.tokenizer.vector_tokenizer import collate_graphs
from tests.fixtures.graph_factory import create_synthetic_raw_graph


@pytest.mark.slow
class TestTokenizerPipeline:
    """End-to-end integration tests for the tokenizer pipeline."""

    @pytest.fixture(scope="class")
    def tokenizer(self):
        config = TokenizerConfig(
            d_model=256,
            n_heads=8,
            vision_backbone=VisionBackboneType.HRNET_W32,
            dropout=0.0,
        )
        tok = Tokenizer(config, n_tef_layers=1)
        tok.eval()
        return tok

    def test_e2e_with_raster_image(self, tokenizer):
        """Full pipeline: RawGraph + raster → EnrichedTokenSequence."""
        graph = create_synthetic_raw_graph(num_edges=10)
        features = collate_graphs([graph])
        images = torch.randn(1, 3, 256, 256)

        with torch.no_grad():
            result = tokenizer(features, images=images)

        assert isinstance(result, EnrichedTokenSequence)
        assert result.token_embeddings.shape == (1, 10, 256)
        assert result.attention_mask.shape == (1, 10)
        assert result.position_encodings.shape == (1, 10, 256)
        assert result.raw_coordinates.shape == (1, 10, 4)
        assert result.confidence_wall.shape == (1, 10)
        assert result.edge_indices is not None
        assert result.is_vector_only is False
        assert result.raster_features is not None

    def test_e2e_vector_only_fallback(self, tokenizer):
        """Full pipeline: RawGraph without raster → vector-only path."""
        graph = create_synthetic_raw_graph(num_edges=10)
        features = collate_graphs([graph])

        with torch.no_grad():
            result = tokenizer(features, images=None)

        assert isinstance(result, EnrichedTokenSequence)
        assert result.token_embeddings.shape == (1, 10, 256)
        assert result.attention_mask.shape == (1, 10)
        assert result.position_encodings.shape == (1, 10, 256)
        assert result.raw_coordinates.shape == (1, 10, 4)
        assert result.confidence_wall.shape == (1, 10)
        assert result.is_vector_only is True
        assert result.raster_features is None

    def test_e2e_batch_with_padding(self, tokenizer):
        """Batch of 2 different-sized graphs: verify padding and masking."""
        g1 = create_synthetic_raw_graph(num_edges=5, seed=1)
        g2 = create_synthetic_raw_graph(num_edges=12, seed=2)
        features = collate_graphs([g1, g2])

        with torch.no_grad():
            result = tokenizer(features, images=None)

        assert result.token_embeddings.shape == (2, 12, 256)
        assert result.attention_mask.shape == (2, 12)
        assert result.position_encodings.shape == (2, 12, 256)
        assert result.raw_coordinates.shape == (2, 12, 4)
        assert result.confidence_wall.shape == (2, 12)

        # g1 has 5 valid tokens → first 5 True, rest False
        assert result.attention_mask[0, :5].all()
        assert not result.attention_mask[0, 5:].any()
        # g2 has 12 valid tokens → all True
        assert result.attention_mask[1].all()

    def test_e2e_output_is_enriched_token_sequence(self, tokenizer):
        """Verify all EnrichedTokenSequence fields are populated."""
        graph = create_synthetic_raw_graph(num_edges=8)
        features = collate_graphs([graph])

        with torch.no_grad():
            result = tokenizer(features, images=None)

        assert result.d_model == 256
        assert result.n_heads == 8
        assert result.stroke_features.shape == (1, 8, 4)  # sw + RGB
        assert result.token_embeddings.dtype == torch.float32
        assert result.attention_mask.dtype == torch.bool
