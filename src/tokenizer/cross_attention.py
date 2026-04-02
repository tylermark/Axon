"""Bidirectional cross-attention fusion and spatial windowing for Stage 2.

Implements Tokenized Early Fusion (TEF): vector tokens and raster features
exchange information via bidirectional cross-attention, bounded by spatial
attention windows. Includes vector-only fallback when raster is unavailable.

Tasks: T-004 (VisionToVectorAttention), T-005 (VectorToVisionAttention),
       T-006 (TokenizedEarlyFusion), T-007 (SpatialAttentionWindow),
       T-008 (VectorOnlyFallback / Tokenizer fallback path).
Reference: ARCHITECTURE.md §Stage 2, MODEL_SPEC.md §Cross-Modal Feature Alignment.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch import nn

if TYPE_CHECKING:
    from docs.interfaces.parser_to_tokenizer import RawGraph
    from src.pipeline.config import TokenizerConfig

from docs.interfaces.tokenizer_to_diffusion import EnrichedTokenSequence
from src.tokenizer.vector_tokenizer import VectorTokenizer, graph_to_token_features
from src.tokenizer.vision_backbone import VisionBackbone, preprocess_image

logger = logging.getLogger(__name__)

_DEFAULT_TEF_LAYERS = 2


# ---------------------------------------------------------------------------
# T-004: Vision-to-Vector Cross-Attention
# ---------------------------------------------------------------------------


class VisionToVectorAttention(nn.Module):
    """Cross-attention: vector tokens query visual features.

    Attention(Q=vector_tokens, K=visual_features, V=visual_features)

    Enriches each vector token embedding with localized semantic context
    from the raster feature map via scaled dot-product attention with
    residual connection and layer normalization.

    Reference: MODEL_SPEC.md §Cross-Modal Feature Alignment, EQ-05.
    """

    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        vector_tokens: torch.Tensor,
        visual_features: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        key_padding_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Enrich vector tokens via cross-attention over visual features.

        Args:
            vector_tokens: (B, N_tokens, d_model) vector token embeddings.
            visual_features: (B, N_visual, d_model) visual feature sequence.
            attn_mask: (B * n_heads, N_tokens, N_visual) bool mask where
                True means the position is masked out (not attended).
            key_padding_mask: (B, N_visual) bool, True = ignore position.

        Returns:
            Enriched vector tokens (B, N_tokens, d_model) with residual
            connection + LayerNorm, and per-head attention weights
            (B, n_heads, N_tokens, N_visual).
        """
        out, weights = self.attn(
            query=vector_tokens,
            key=visual_features,
            value=visual_features,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            average_attn_weights=False,
        )
        return self.norm(vector_tokens + out), weights


# ---------------------------------------------------------------------------
# T-005: Vector-to-Vision Cross-Attention
# ---------------------------------------------------------------------------


class VectorToVisionAttention(nn.Module):
    """Cross-attention: visual features query vector tokens.

    Attention(Q=visual_features, K=vector_tokens, V=vector_tokens)

    Enriches visual features with precise geometric information from
    the vector token sequence.
    """

    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        visual_features: torch.Tensor,
        vector_tokens: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        key_padding_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Enrich visual features via cross-attention over vector tokens.

        Args:
            visual_features: (B, N_visual, d_model) visual feature sequence.
            vector_tokens: (B, N_tokens, d_model) vector token embeddings.
            attn_mask: (B * n_heads, N_visual, N_tokens) bool mask where
                True means the position is masked out.
            key_padding_mask: (B, N_tokens) bool, True = ignore position.

        Returns:
            Enriched visual features (B, N_visual, d_model) with residual
            connection + LayerNorm, and attention weights.
        """
        out, weights = self.attn(
            query=visual_features,
            key=vector_tokens,
            value=vector_tokens,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
        )
        return self.norm(visual_features + out), weights


# ---------------------------------------------------------------------------
# T-007: Spatial Attention Window
# ---------------------------------------------------------------------------


class SpatialAttentionWindow:
    """Computes spatial attention masks to bound cross-attention radius.

    Each vector token only attends to visual features within a distance
    of ``attention_radius_fraction`` in normalized page coordinate space.
    This prevents global attention blowup on large architectural sheets.

    Uses chunked pairwise distance computation to avoid materializing
    huge dense distance matrices for large sequences.

    Reference: ARCHITECTURE.md §Stage 2 Key Design Decisions.
    """

    def __init__(
        self,
        attention_radius_fraction: float = 0.05,
        chunk_size: int = 1024,
    ) -> None:
        """Initialize spatial attention window.

        Args:
            attention_radius_fraction: Maximum normalized distance for
                attention. Positions beyond this radius are masked out.
            chunk_size: Number of query positions processed per chunk
                when computing pairwise distances.
        """
        self.radius = attention_radius_fraction
        self.chunk_size = chunk_size

    def compute_mask(
        self,
        token_positions: torch.Tensor,
        visual_positions: torch.Tensor,
    ) -> torch.Tensor:
        """Compute spatial attention mask for cross-attention.

        Args:
            token_positions: (B, N_tokens, 2) normalized (x, y) positions.
            visual_positions: (B, N_visual, 2) normalized (x, y) positions.

        Returns:
            Bool mask (B, N_tokens, N_visual) where True means the position
            is MASKED OUT (beyond attention radius). Compatible with
            PyTorch ``nn.MultiheadAttention`` attn_mask convention.
        """
        b, n_t, _ = token_positions.shape
        _, n_v, _ = visual_positions.shape
        r_sq = self.radius * self.radius

        if n_t <= self.chunk_size:
            diff = token_positions.unsqueeze(2) - visual_positions.unsqueeze(1)
            return (diff * diff).sum(dim=-1) > r_sq

        # Chunked computation for large token sequences
        mask = torch.ones(b, n_t, n_v, dtype=torch.bool, device=token_positions.device)
        for start in range(0, n_t, self.chunk_size):
            end = min(start + self.chunk_size, n_t)
            chunk = token_positions[:, start:end]
            diff = chunk.unsqueeze(2) - visual_positions.unsqueeze(1)
            mask[:, start:end] = (diff * diff).sum(dim=-1) > r_sq
        return mask


def _expand_mask_for_heads(mask: torch.Tensor, n_heads: int) -> torch.Tensor:
    """Expand (B, N_q, N_kv) bool mask to (B * n_heads, N_q, N_kv).

    Required by ``nn.MultiheadAttention`` which expects 3D attn_mask
    with shape ``(batch * num_heads, query_len, key_len)``.

    Args:
        mask: Bool attention mask, shape (B, N_q, N_kv).
        n_heads: Number of attention heads.

    Returns:
        Expanded mask (B * n_heads, N_q, N_kv).
    """
    b, n_q, n_kv = mask.shape
    return mask.unsqueeze(1).expand(b, n_heads, n_q, n_kv).reshape(b * n_heads, n_q, n_kv)


# ---------------------------------------------------------------------------
# T-006: Tokenized Early Fusion
# ---------------------------------------------------------------------------


class TokenizedEarlyFusion(nn.Module):
    """TEF: Bidirectional cross-modal fusion combining both attention directions.

    Each TEF layer performs:
    1. VisionToVectorAttention -- vector tokens absorb visual semantics
    2. VectorToVisionAttention -- visual features absorb geometric precision
    3. Gated fusion -- sigmoid gate modulates projected token representations
    4. Feed-forward network with residual connection + LayerNorm

    Multiple TEF layers are stacked for deeper cross-modal entanglement.

    Reference: ARCHITECTURE.md §Stage 2, MODEL_SPEC.md §Cross-Modal Feature Alignment.
    """

    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = _DEFAULT_TEF_LAYERS,
        dropout: float = 0.1,
    ) -> None:
        """Initialize stacked TEF fusion layers.

        Args:
            d_model: Token embedding dimension.
            n_heads: Number of attention heads for cross-attention.
            n_layers: Number of stacked TEF layers.
            dropout: Dropout rate in attention and FFN layers.
        """
        super().__init__()
        self.n_layers = n_layers
        self.n_heads = n_heads

        self.v2v_attn = nn.ModuleList()
        self.vec2vis_attn = nn.ModuleList()
        self.gate_sigmoid = nn.ModuleList()
        self.gate_proj = nn.ModuleList()
        self.ffn = nn.ModuleList()
        self.ffn_norm = nn.ModuleList()

        for _ in range(n_layers):
            self.v2v_attn.append(VisionToVectorAttention(d_model, n_heads, dropout))
            self.vec2vis_attn.append(VectorToVisionAttention(d_model, n_heads, dropout))
            self.gate_sigmoid.append(nn.Linear(d_model, d_model))
            self.gate_proj.append(nn.Linear(d_model, d_model))
            self.ffn.append(
                nn.Sequential(
                    nn.Linear(d_model, d_model * 4),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(d_model * 4, d_model),
                    nn.Dropout(dropout),
                )
            )
            self.ffn_norm.append(nn.LayerNorm(d_model))

    def forward(
        self,
        vector_tokens: torch.Tensor,
        visual_features: torch.Tensor,
        spatial_mask: torch.Tensor | None = None,
        token_padding_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Run stacked TEF fusion layers.

        Args:
            vector_tokens: (B, N_tokens, d_model) embedded vector tokens.
            visual_features: (B, N_visual, d_model) visual feature sequence.
            spatial_mask: (B, N_tokens, N_visual) bool mask from
                SpatialAttentionWindow, True = masked out.
            token_padding_mask: (B, N_tokens) bool, True = padding (ignore).
                Used as key_padding_mask when vectors serve as keys.

        Returns:
            Fused token embeddings (B, N_tokens, d_model) and last-layer
            vision-to-vector per-head attention weights
            (B, n_heads, N_tokens, N_visual) for interpretability.
        """
        x = vector_tokens
        v = visual_features
        last_weights: torch.Tensor | None = None

        # Expand spatial masks for multi-head attention
        v2v_mask = None
        vec2vis_mask = None
        if spatial_mask is not None:
            v2v_mask = _expand_mask_for_heads(spatial_mask, self.n_heads)
            vec2vis_mask = _expand_mask_for_heads(spatial_mask.transpose(1, 2), self.n_heads)

        for i in range(self.n_layers):
            # Bidirectional cross-attention
            enriched_vec, last_weights = self.v2v_attn[i](x, v, attn_mask=v2v_mask)
            enriched_vis, _ = self.vec2vis_attn[i](
                v, x, attn_mask=vec2vis_mask, key_padding_mask=token_padding_mask
            )

            # Gated fusion: sigmoid gate modulates projected enriched vectors
            gate = torch.sigmoid(self.gate_sigmoid[i](enriched_vec))
            x = enriched_vec + gate * self.gate_proj[i](enriched_vec)

            # Propagate enriched vision features to next layer
            v = enriched_vis

            # FFN with residual + LayerNorm
            x = self.ffn_norm[i](x + self.ffn[i](x))

        return x, last_weights


# ---------------------------------------------------------------------------
# T-008: Vector-Only Fallback
# ---------------------------------------------------------------------------


class VectorOnlyFallback(nn.Module):
    """Self-attention fallback when raster features are unavailable.

    Replaces cross-modal TEF fusion with self-attention on vector tokens,
    ensuring tokens still receive contextual enrichment from neighboring
    tokens even without visual features.

    Reference: ARCHITECTURE.md §Stage 2 Key Design Decisions.
    """

    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = _DEFAULT_TEF_LAYERS,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.self_attn = nn.ModuleList()
        self.attn_norm = nn.ModuleList()
        self.ffn = nn.ModuleList()
        self.ffn_norm = nn.ModuleList()

        for _ in range(n_layers):
            self.self_attn.append(
                nn.MultiheadAttention(
                    embed_dim=d_model,
                    num_heads=n_heads,
                    dropout=dropout,
                    batch_first=True,
                )
            )
            self.attn_norm.append(nn.LayerNorm(d_model))
            self.ffn.append(
                nn.Sequential(
                    nn.Linear(d_model, d_model * 4),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(d_model * 4, d_model),
                    nn.Dropout(dropout),
                )
            )
            self.ffn_norm.append(nn.LayerNorm(d_model))

    def forward(
        self,
        tokens: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run self-attention on vector tokens.

        Args:
            tokens: (B, N, d_model) vector token embeddings.
            attention_mask: (B, N) bool, True = valid token.

        Returns:
            Contextually enriched tokens (B, N, d_model).
        """
        # Convert True=valid → True=ignore for key_padding_mask
        key_padding_mask = ~attention_mask if attention_mask is not None else None

        x = tokens
        for i in range(len(self.self_attn)):
            out, _ = self.self_attn[i](x, x, x, key_padding_mask=key_padding_mask)
            x = self.attn_norm[i](x + out)
            x = self.ffn_norm[i](x + self.ffn[i](x))
        return x


# ---------------------------------------------------------------------------
# Top-Level Tokenizer
# ---------------------------------------------------------------------------


def _token_midpoints(coordinates: torch.Tensor) -> torch.Tensor:
    """Compute (x, y) midpoints from [x1, y1, x2, y2] coordinates.

    Args:
        coordinates: (B, N, 4) normalized coordinates.

    Returns:
        Midpoints (B, N, 2).
    """
    return torch.stack(
        [
            (coordinates[..., 0] + coordinates[..., 2]) / 2.0,
            (coordinates[..., 1] + coordinates[..., 3]) / 2.0,
        ],
        dim=-1,
    )


class Tokenizer(nn.Module):
    """Full Stage 2 pipeline: RawGraph + optional raster -> EnrichedTokenSequence.

    Wires together:
    1. VectorTokenizer (embedding + positional encoding)
    2. VisionBackbone (raster feature extraction)
    3. SpatialAttentionWindow (bounded cross-attention radius)
    4. TokenizedEarlyFusion (bidirectional cross-modal fusion)
    5. VectorOnlyFallback (self-attention when no raster available)

    Reference: ARCHITECTURE.md §Stage 2.
    """

    def __init__(
        self,
        config: TokenizerConfig,
        n_tef_layers: int = _DEFAULT_TEF_LAYERS,
    ) -> None:
        """Initialize the full tokenization pipeline.

        Args:
            config: Tokenizer configuration with d_model, n_heads,
                attention_radius_fraction, dropout, etc.
            n_tef_layers: Number of stacked TEF / fallback layers.
        """
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.n_heads = config.n_heads

        self.vector_tokenizer = VectorTokenizer(config)
        self.vision_backbone = VisionBackbone(config)
        self.spatial_window = SpatialAttentionWindow(
            attention_radius_fraction=config.attention_radius_fraction,
        )
        self.tef = TokenizedEarlyFusion(
            d_model=config.d_model,
            n_heads=config.n_heads,
            n_layers=n_tef_layers,
            dropout=config.dropout,
        )
        self.vector_fallback = VectorOnlyFallback(
            d_model=config.d_model,
            n_heads=config.n_heads,
            n_layers=n_tef_layers,
            dropout=config.dropout,
        )

    def forward(
        self,
        raw_features: dict[str, torch.Tensor],
        images: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> EnrichedTokenSequence:
        """Run full tokenization pipeline.

        Args:
            raw_features: Dict from graph_to_token_features() / collate_graphs()
                with keys: operator_type, coordinates, stroke_width, dash_hash,
                color_rgb, confidence_wall, attention_mask, raw_coordinates.
            images: (B, 3, H, W) preprocessed raster images, or None for
                vector-only mode.
            attention_mask: (B, N) bool, True for valid tokens. If None,
                uses raw_features["attention_mask"].

        Returns:
            EnrichedTokenSequence ready for the Diffusion Engine.
        """
        if attention_mask is None:
            attention_mask = raw_features.get("attention_mask")

        # 1. Embed vector tokens with positional encoding
        vector_tokens = self.vector_tokenizer(raw_features)
        pos_encodings = self.vector_tokenizer.positional_encoding(raw_features["coordinates"])

        # 2. Fusion path selection
        is_vector_only = images is None
        raster_out: torch.Tensor | None = None
        v2v_weights: torch.Tensor | None = None

        if not is_vector_only:
            # Vision backbone -> multi-scale features
            vision = self.vision_backbone(images)

            # Spatial attention window mask
            token_pos = _token_midpoints(raw_features["coordinates"])
            spatial_mask = self.spatial_window.compute_mask(token_pos, vision.spatial_positions)

            # Padding mask: True=valid -> True=ignore for MHA convention
            token_pad = ~attention_mask if attention_mask is not None else None

            # TEF bidirectional cross-modal fusion
            fused, v2v_weights = self.tef(
                vector_tokens,
                vision.flat_features,
                spatial_mask=spatial_mask,
                token_padding_mask=token_pad,
            )
            raster_out = vision.feature_maps[0]
        else:
            # Vector-only fallback: self-attention
            if not self.config.vector_only_fallback:
                logger.warning("No images provided and vector_only_fallback is disabled")
            fused = self.vector_fallback(vector_tokens, attention_mask)

        # 3. Assemble stroke features for downstream constraint agent
        stroke_features = torch.cat(
            [
                raw_features["stroke_width"].unsqueeze(-1),
                raw_features["color_rgb"],
            ],
            dim=-1,
        )  # (B, N, 4)

        # 4. Build output
        mask = (
            attention_mask
            if attention_mask is not None
            else torch.ones(fused.shape[:2], dtype=torch.bool, device=fused.device)
        )

        return EnrichedTokenSequence(
            token_embeddings=fused,
            attention_mask=mask,
            position_encodings=pos_encodings,
            raw_coordinates=raw_features["raw_coordinates"],
            stroke_features=stroke_features,
            edge_indices=torch.empty(2, 0, dtype=torch.long, device=fused.device),
            confidence_wall=raw_features["confidence_wall"],
            raster_features=raster_out,
            vision_to_vector_weights=v2v_weights,
            d_model=self.d_model,
            n_heads=self.n_heads,
            is_vector_only=is_vector_only,
        )

    @torch.no_grad()
    def tokenize_graph(
        self,
        graph: RawGraph,
        image: np.ndarray | None = None,
        config: TokenizerConfig | None = None,
    ) -> EnrichedTokenSequence:
        """Convenience method: RawGraph -> EnrichedTokenSequence in one call.

        Handles feature conversion, batching, device placement, and optional
        raster preprocessing in a single inference-mode call.

        Args:
            graph: RawGraph from the parser.
            image: Optional raster image as (H, W, 3) uint8 numpy array.
            config: Reserved for future per-call config overrides.

        Returns:
            EnrichedTokenSequence with all fields populated including
            edge_indices and page_dimensions.
        """
        was_training = self.training
        self.eval()
        try:
            device = next(self.parameters()).device

            # Convert graph to batched feature tensors (batch size 1)
            features = graph_to_token_features(graph)
            raw_features = {k: v.unsqueeze(0).to(device) for k, v in features.items()}

            # Preprocess raster image if provided
            images = None
            if image is not None:
                images = preprocess_image(image).to(device)

            result = self.forward(raw_features, images=images)

            # Populate graph-specific fields
            if len(graph.edges) > 0:
                result.edge_indices = torch.from_numpy(graph.edges.T.astype(np.int64)).to(device)

            result.page_dimensions = torch.tensor(
                [[graph.page_width, graph.page_height]],
                dtype=torch.float64,
                device=device,
            )

            return result
        finally:
            if was_training:
                self.train()
