"""Unit tests for src/tokenizer/vision_backbone.py.

Tests preprocess_image() (fast) and VisionBackbone (slow, loads pretrained weights).

Q-003: tokenizer unit tests (vision path).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.tokenizer.vision_backbone import preprocess_image

# ---------------------------------------------------------------------------
# preprocess_image (fast — no timm needed)
# ---------------------------------------------------------------------------


class TestPreprocessImage:
    """Tests for preprocess_image()."""

    def test_numpy_input_to_tensor(self):
        img = np.random.randint(0, 256, (100, 150, 3), dtype=np.uint8)
        out = preprocess_image(img)
        assert out.shape == (1, 3, 512, 512)
        assert out.dtype == torch.float32

    def test_output_is_normalized_not_raw_0_255(self):
        img = np.random.randint(0, 256, (100, 150, 3), dtype=np.uint8)
        out = preprocess_image(img)
        # ImageNet-normalized values are roughly in [-2.5, 2.8]
        assert out.max() < 10.0
        assert out.min() > -10.0

    def test_custom_target_size(self):
        img = np.random.randint(0, 256, (200, 300, 3), dtype=np.uint8)
        out = preprocess_image(img, target_size=(256, 256))
        assert out.shape == (1, 3, 256, 256)

    def test_different_input_sizes_resize_correctly(self):
        for h, w in [(50, 80), (300, 200), (1024, 768)]:
            img = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
            out = preprocess_image(img, target_size=(128, 128))
            assert out.shape == (1, 3, 128, 128)

    def test_tensor_input_3d(self):
        tensor = torch.randint(0, 256, (3, 100, 150), dtype=torch.uint8)
        out = preprocess_image(tensor)
        assert out.shape == (1, 3, 512, 512)

    def test_tensor_input_4d_already_batched(self):
        tensor = torch.randint(0, 256, (1, 3, 100, 150), dtype=torch.uint8)
        out = preprocess_image(tensor)
        assert out.shape == (1, 3, 512, 512)


# ---------------------------------------------------------------------------
# VisionBackbone (slow — loads timm model)
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestVisionBackbone:
    """Tests for VisionBackbone — requires timm pretrained model download."""

    @pytest.fixture(scope="class")
    def backbone(self):
        from src.pipeline.config import TokenizerConfig
        from src.pipeline.config import VisionBackbone as VisionBackboneType
        from src.tokenizer.vision_backbone import VisionBackbone

        config = TokenizerConfig(
            d_model=64,
            n_heads=4,
            vision_backbone=VisionBackboneType.HRNET_W32,
            dropout=0.0,
        )
        model = VisionBackbone(config)
        model.eval()
        return model

    def test_forward_produces_vision_features(self, backbone):
        images = torch.randn(1, 3, 256, 256)
        with torch.no_grad():
            vf = backbone(images)
        assert vf.flat_features.ndim == 3
        assert vf.flat_features.shape[0] == 1
        assert vf.flat_features.shape[2] == 64  # d_model

    def test_feature_maps_projected_to_d_model(self, backbone):
        images = torch.randn(1, 3, 256, 256)
        with torch.no_grad():
            vf = backbone(images)
        for fm in vf.feature_maps:
            assert fm.shape[1] == 64  # channel dim = d_model

    def test_flat_features_shape_consistent(self, backbone):
        images = torch.randn(2, 3, 256, 256)
        with torch.no_grad():
            vf = backbone(images)
        assert vf.flat_features.shape[0] == 2
        assert vf.flat_features.shape[2] == 64

    def test_spatial_positions_in_unit_range(self, backbone):
        images = torch.randn(1, 3, 256, 256)
        with torch.no_grad():
            vf = backbone(images)
        assert vf.spatial_positions.shape == (1, vf.flat_features.shape[1], 2)
        assert vf.spatial_positions.min() >= 0.0
        assert vf.spatial_positions.max() <= 1.0

    def test_flat_features_and_spatial_positions_match(self, backbone):
        images = torch.randn(1, 3, 256, 256)
        with torch.no_grad():
            vf = backbone(images)
        # N_visual must match between features and positions
        assert vf.flat_features.shape[1] == vf.spatial_positions.shape[1]
