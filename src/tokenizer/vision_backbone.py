"""Multi-scale vision backbone for raster feature extraction.

Wraps HRNet and Swin Transformer backbones via timm to extract
multi-scale feature maps from rendered PDF page images. Features
are projected to a common dimension for cross-attention fusion
with vector tokens.

Reference: ARCHITECTURE.md §Stage 2, MODEL_SPEC.md §Cross-Modal Feature Alignment.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as f

try:
    import timm

    _TIMM_AVAILABLE = True
except ImportError:
    _TIMM_AVAILABLE = False

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.pipeline.config import TokenizerConfig

logger = logging.getLogger(__name__)

# ImageNet normalization constants
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass
class VisionFeatures:
    """Multi-scale visual features extracted by the backbone."""

    feature_maps: list[torch.Tensor]
    """Feature maps at each scale, each (B, d_model, H_i, W_i)."""

    flat_features: torch.Tensor
    """All features flattened and concatenated, shape (B, N_visual, d_model).
    Used as K,V in cross-attention with vector tokens."""

    spatial_positions: torch.Tensor
    """Normalized (x, y) position of each spatial feature, shape (B, N_visual, 2).
    Used for spatial attention windowing."""

    scales: list[tuple[int, int]]
    """(H_i, W_i) for each feature map scale."""


class VisionBackbone(nn.Module):
    """Multi-scale raster feature extractor using timm backbones.

    Wraps HRNet or Swin Transformer to extract feature maps at multiple
    spatial scales, projects them to a common channel dimension via 1x1
    convolutions, and flattens into a single sequence for cross-attention.

    Args:
        config: TokenizerConfig with vision_backbone, d_model settings.

    Raises:
        RuntimeError: If timm is not installed.
    """

    def __init__(self, config: TokenizerConfig) -> None:
        super().__init__()

        if not _TIMM_AVAILABLE:
            raise RuntimeError(
                "timm is required for VisionBackbone. Install with: pip install timm"
            )

        self.d_model = config.d_model
        self.backbone_name = config.vision_backbone.value

        # Create backbone with intermediate feature extraction
        try:
            self.backbone = timm.create_model(
                self.backbone_name,
                pretrained=True,
                features_only=True,
            )
        except Exception:
            logger.warning(
                "Failed to load pretrained %s, falling back to random init",
                self.backbone_name,
            )
            self.backbone = timm.create_model(
                self.backbone_name,
                pretrained=False,
                features_only=True,
            )

        # Freeze backbone by default — fine-tuned later via unfreeze()
        for param in self.backbone.parameters():
            param.requires_grad = False

        # 1x1 conv projections: per-scale channels -> d_model
        channels = self.backbone.feature_info.channels()
        self.projections = nn.ModuleList(
            [nn.Conv2d(c, self.d_model, kernel_size=1, bias=False) for c in channels]
        )

        self._num_scales = len(channels)

    def forward(self, images: torch.Tensor) -> VisionFeatures:
        """Extract multi-scale features from raster images.

        Args:
            images: (B, 3, H, W) float32, normalized RGB images.

        Returns:
            VisionFeatures containing projected feature maps and
            flattened spatial sequence for cross-attention.
        """
        raw_features = self.backbone(images)

        projected: list[torch.Tensor] = []
        scales: list[tuple[int, int]] = []
        for feat, proj in zip(raw_features, self.projections, strict=True):
            p = proj(feat)  # (B, d_model, H_i, W_i)
            projected.append(p)
            scales.append((p.shape[2], p.shape[3]))

        # Flatten each scale and concatenate into a single sequence
        batch_size = images.shape[0]
        flat_list: list[torch.Tensor] = []
        pos_list: list[torch.Tensor] = []

        for p, (h, w) in zip(projected, scales, strict=True):
            # (B, d_model, H, W) -> (B, H*W, d_model)
            flat_list.append(p.flatten(2).transpose(1, 2))

            # Normalized (x, y) positions for spatial attention windowing
            ys = torch.linspace(0.0, 1.0, h, device=images.device, dtype=images.dtype)
            xs = torch.linspace(0.0, 1.0, w, device=images.device, dtype=images.dtype)
            grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
            positions = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=-1)
            pos_list.append(positions.unsqueeze(0).expand(batch_size, -1, -1))

        flat_features = torch.cat(flat_list, dim=1)  # (B, N_visual, d_model)
        spatial_positions = torch.cat(pos_list, dim=1)  # (B, N_visual, 2)

        return VisionFeatures(
            feature_maps=projected,
            flat_features=flat_features,
            spatial_positions=spatial_positions,
            scales=scales,
        )

    def unfreeze(self) -> None:
        """Unfreeze backbone parameters for fine-tuning."""
        for param in self.backbone.parameters():
            param.requires_grad = True


def preprocess_image(
    image: np.ndarray | torch.Tensor,
    target_size: tuple[int, int] = (512, 512),
) -> torch.Tensor:
    """Preprocess a raster image for the vision backbone.

    Resizes and normalizes with ImageNet statistics.

    Args:
        image: RGB image as (H, W, 3) uint8 numpy array or (3, H, W) tensor.
        target_size: Resize target (H, W).

    Returns:
        Normalized tensor (1, 3, H, W) float32 with ImageNet normalization.
    """
    if isinstance(image, np.ndarray):
        # (H, W, 3) uint8 -> (1, 3, H, W) float32 in [0, 1]
        tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    else:
        tensor = image.float()
        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)
        if tensor.max() > 1.0:
            tensor = tensor / 255.0

    # Resize to target
    tensor = f.interpolate(tensor, size=target_size, mode="bilinear", align_corners=False)

    # ImageNet normalization
    mean = torch.tensor(IMAGENET_MEAN, device=tensor.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=tensor.device).view(1, 3, 1, 1)
    return (tensor - mean) / std


def render_pdf_page(
    pdf_path: str | Path,
    page_index: int = 0,
    dpi: int = 300,
) -> np.ndarray:
    """Render a PDF page to a raster image using PyMuPDF.

    Args:
        pdf_path: Path to PDF file.
        page_index: Page number (0-indexed).
        dpi: Render resolution (300 default from TokenizerConfig).

    Returns:
        RGB image as (H, W, 3) uint8 numpy array.

    Raises:
        FileNotFoundError: If pdf_path does not exist.
        IndexError: If page_index is out of range.
    """
    import fitz  # PyMuPDF — lazy import to avoid hard dependency

    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        msg = f"PDF not found: {pdf_path}"
        raise FileNotFoundError(msg)

    doc = fitz.open(pdf_path)
    try:
        if page_index >= len(doc):
            msg = f"Page {page_index} out of range (document has {len(doc)} pages)"
            raise IndexError(msg)

        page = doc[page_index]
        zoom = dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, alpha=False)

        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, 3)
        return img.copy()  # Own the memory before doc closes
    finally:
        doc.close()
