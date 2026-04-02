"""Reverse diffusion process, transformer backbone, DDIM sampling, and VLB loss.

Implements the neural denoising network epsilon_theta and the full reverse
diffusion procedure for recovering clean structural graphs from noise.

Components:
    1. TimestepEmbedding  -- sinusoidal + MLP time conditioning
    2. GraphTransformerBlock -- HDSE-biased attention + cross-attention
    3. GraphTransformerBackbone -- 12-layer denoising network
    4. ReverseDiffusion -- DDPM single-step, DDIM sampling, VLB loss
    5. DiffusionLoss -- loss container
    6. GraphDiffusionModel -- top-level module wiring forward+reverse+HDSE

Reference:
    MODEL_SPEC.md  -- EQ-06, Generative Graph Denoising Diffusion Engine
    ARCHITECTURE.md -- Stage 3: Graph Diffusion Engine
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as nnf
from torch import nn

from src.diffusion.forward import ForwardDiffusion
from src.diffusion.hdse import HDSE
from src.diffusion.scheduler import DiffusionScheduler

if TYPE_CHECKING:
    from docs.interfaces.diffusion_output import RefinedStructuralGraph
    from src.diffusion.hdse import HDSEOutput
    from src.pipeline.config import DiffusionConfig


# ---------------------------------------------------------------------------
# Timestep Embedding
# ---------------------------------------------------------------------------


class TimestepEmbedding(nn.Module):
    """Sinusoidal timestep embedding.

    Maps scalar timestep t to a d_model vector using sinusoidal positional
    encoding followed by a 2-layer MLP projection.

    Args:
        d_model: Output embedding dimension.
    """

    def __init__(self, d_model: int = 512) -> None:
        super().__init__()
        self.d_model = d_model
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Embed timesteps.

        Args:
            t: (B,) int64 timestep indices.

        Returns:
            (B, d_model) float32 embeddings.
        """
        half_dim = self.d_model // 2
        freqs = torch.exp(
            -math.log(10000.0)
            * torch.arange(half_dim, device=t.device, dtype=torch.float32)
            / half_dim
        )
        args = t.float().unsqueeze(-1) * freqs.unsqueeze(0)  # (B, half_dim)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # (B, d_model)
        if self.d_model % 2 == 1:
            emb = nnf.pad(emb, (0, 1))
        return self.mlp(emb)


# ---------------------------------------------------------------------------
# Graph Transformer Block (D-004)
# ---------------------------------------------------------------------------


class GraphTransformerBlock(nn.Module):
    """Single transformer block with HDSE-biased self-attention.

    Architecture (pre-norm):
        1. Self-attention with HDSE bias added to scores before softmax
        2. Cross-attention to context embeddings c from the tokenizer
        3. Position-wise FFN (d_model -> 4*d_model -> d_model, GELU)
        4. Residual connections around each sub-layer

    Args:
        d_model: Hidden dimension.
        n_heads: Number of attention heads.
        d_context: Dimension of cross-modal context (tokenizer d_model=256).
        dropout: Dropout probability.
    """

    def __init__(
        self,
        d_model: int = 512,
        n_heads: int = 8,
        d_context: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        # Self-attention
        self.norm_sa = nn.LayerNorm(d_model)
        self.qkv_sa = nn.Linear(d_model, 3 * d_model)
        self.out_sa = nn.Linear(d_model, d_model)
        self.drop_sa = nn.Dropout(dropout)

        # Cross-attention
        self.norm_ca = nn.LayerNorm(d_model)
        self.q_ca = nn.Linear(d_model, d_model)
        self.kv_ca = nn.Linear(d_context, 2 * d_model)
        self.out_ca = nn.Linear(d_model, d_model)
        self.drop_ca = nn.Dropout(dropout)

        # FFN
        self.norm_ff = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        hdse_bias: torch.Tensor | None,
        context: torch.Tensor | None = None,
        node_mask: torch.Tensor | None = None,
        context_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass through one transformer block.

        Args:
            x: (B, N, d_model) node features.
            hdse_bias: (B, n_heads, N, N) HDSE attention bias.
            context: (B, N_ctx, d_ctx) cross-modal context from tokenizer.
            node_mask: (B, N) bool mask for valid nodes.
            context_mask: (B, N_ctx) bool mask for valid context tokens.

        Returns:
            (B, N, d_model) updated node features.
        """
        bsz, seq, dim = x.shape
        nh, dk = self.n_heads, self.d_k

        # --- Self-attention with HDSE bias ---
        h = self.norm_sa(x)
        qkv = self.qkv_sa(h).view(bsz, seq, 3, nh, dk).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # each (bsz, nh, seq, dk)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(dk)
        if hdse_bias is not None:
            scores = scores + hdse_bias
        if node_mask is not None:
            scores = scores.masked_fill(~node_mask[:, None, None, :], -1e9)
        attn = self.drop_sa(nnf.softmax(scores, dim=-1))
        sa_out = attn @ v  # (bsz, nh, seq, dk)
        sa_out = sa_out.transpose(1, 2).contiguous().view(bsz, seq, dim)
        x = x + self.out_sa(sa_out)

        # --- Cross-attention (skip when no context) ---
        if context is not None:
            h = self.norm_ca(x)
            n_ctx = context.size(1)
            q = self.q_ca(h).view(bsz, seq, nh, dk).transpose(1, 2)
            kv = self.kv_ca(context).view(bsz, n_ctx, 2, nh, dk).permute(2, 0, 3, 1, 4)
            k_c, v_c = kv.unbind(0)

            scores = (q @ k_c.transpose(-2, -1)) / math.sqrt(dk)
            if context_mask is not None:
                scores = scores.masked_fill(~context_mask[:, None, None, :], -1e9)
            attn = self.drop_ca(nnf.softmax(scores, dim=-1))
            ca_out = (attn @ v_c).transpose(1, 2).contiguous().view(bsz, seq, dim)
            x = x + self.out_ca(ca_out)

        # --- FFN ---
        x = x + self.ffn(self.norm_ff(x))
        return x


# ---------------------------------------------------------------------------
# Graph Transformer Backbone (D-004)
# ---------------------------------------------------------------------------


class GraphTransformerBackbone(nn.Module):
    """Stack of transformer blocks forming the denoising network epsilon_theta.

    12 layers, d_model=512, n_heads=8 per ARCHITECTURE.md Stage 3.

    The backbone predicts coordinate noise epsilon and edge existence logits
    from the noised graph G_t, timestep t, HDSE encodings, and cross-modal
    context c.

    Args:
        config: Diffusion configuration.
    """

    def __init__(self, config: DiffusionConfig) -> None:
        super().__init__()
        d = config.d_model
        self.d_model = d

        # Input projections
        self.coord_proj = nn.Linear(2, d)
        self.degree_proj = nn.Linear(1, d)
        self.time_embed = TimestepEmbedding(d)

        # Transformer blocks (d_context=256 per tokenizer spec)
        self.blocks = nn.ModuleList(
            [
                GraphTransformerBlock(d, config.n_heads, d_context=256, dropout=config.dropout)
                for _ in range(config.n_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(d)

        # Output heads
        self.coord_head = nn.Linear(d, 2)
        edge_dim = d // 4
        self.edge_src = nn.Linear(d, edge_dim)
        self.edge_dst = nn.Linear(d, edge_dim)
        self.edge_bias = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        x_t: torch.Tensor,
        a_t: torch.Tensor,
        t: torch.Tensor,
        hdse_output: HDSEOutput | None,
        context: torch.Tensor | None,
        node_mask: torch.Tensor | None,
        context_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict coordinate noise and edge logits.

        Args:
            x_t: (B, N, 2) noised coordinates.
            a_t: (B, N, N) noised adjacency.
            t: (B,) timesteps.
            hdse_output: HDSE attention bias and node encodings.
            context: (B, N_ctx, d_ctx) from tokenizer.
            node_mask: (B, N) valid node mask.
            context_mask: (B, N_ctx) valid context mask.

        Returns:
            epsilon_pred: (B, N, 2) predicted coordinate noise.
            edge_logits: (B, N, N) predicted edge existence logits.
        """
        n = x_t.size(1)

        # Build node representations
        h = self.coord_proj(x_t)
        h = h + self.degree_proj(a_t.sum(dim=-1, keepdim=True))
        h = h + self.time_embed(t).unsqueeze(1)

        hdse_bias = None
        if hdse_output is not None:
            h = h + hdse_output.node_encodings
            hdse_bias = hdse_output.attention_bias

        for block in self.blocks:
            h = block(h, hdse_bias, context, node_mask, context_mask)
        h = self.final_norm(h)

        # Coordinate noise head
        eps = self.coord_head(h)

        # Edge logits via symmetric bilinear
        hs = self.edge_src(h)
        hd = self.edge_dst(h)
        edge_logits = (hs @ hd.transpose(-2, -1)) + self.edge_bias
        edge_logits = (edge_logits + edge_logits.transpose(-2, -1)) / 2.0

        # Zero diagonal (no self-loops)
        diag = torch.eye(n, device=edge_logits.device, dtype=torch.bool).unsqueeze(0)
        edge_logits = edge_logits.masked_fill(diag, 0.0)

        # Mask padded positions
        if node_mask is not None:
            cm = node_mask.unsqueeze(-1).float()
            eps = eps * cm
            pm = node_mask.unsqueeze(-1) & node_mask.unsqueeze(-2)
            edge_logits = edge_logits.masked_fill(~pm, 0.0)

        return eps, edge_logits


# ---------------------------------------------------------------------------
# Loss container (D-007)
# ---------------------------------------------------------------------------


@dataclass
class DiffusionLoss:
    """Variational lower bound training loss components."""

    total: torch.Tensor
    """Combined loss (scalar)."""

    coordinate_loss: torch.Tensor
    """MSE on predicted vs actual coordinate noise."""

    adjacency_loss: torch.Tensor
    """BCE on edge logits vs clean adjacency."""

    t: torch.Tensor
    """(B,) timesteps sampled for this batch."""


# ---------------------------------------------------------------------------
# Reverse Diffusion (D-005 / D-006 / D-007)
# ---------------------------------------------------------------------------


class ReverseDiffusion(nn.Module):
    """Reverse denoising process: recover clean graph from noise.

    Provides three entry points:
        - ``denoise_step``: single DDPM step G_t -> G_{t-1}
        - ``ddim_sample``:  fast DDIM inference in *num_steps* steps
        - ``compute_vlb_loss``: training loss (coordinate MSE + adjacency BCE)

    Posterior formula for coordinates:
        x_{t-1} = (1/sqrt(alpha_t))(x_t - (beta_t/sqrt(1-alpha_bar_t)) eps_theta) + sigma_t z

    Args:
        backbone: Denoising transformer network.
        scheduler: Precomputed noise schedule.
    """

    def __init__(self, backbone: GraphTransformerBackbone, scheduler: DiffusionScheduler) -> None:
        super().__init__()
        self.backbone = backbone
        self.scheduler = scheduler

    # ------------------------------------------------------------------
    # Single DDPM step (D-005)
    # ------------------------------------------------------------------

    def denoise_step(
        self,
        x_t: torch.Tensor,
        a_t: torch.Tensor,
        t: torch.Tensor,
        hdse_output: HDSEOutput | None,
        context: torch.Tensor | None,
        node_mask: torch.Tensor | None,
        context_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Single reverse diffusion step: G_t -> G_{t-1}.

        Args:
            x_t: (B, N, 2) noised coordinates.
            a_t: (B, N, N) noised adjacency.
            t: (B,) current timesteps.
            hdse_output: HDSE encodings.
            context: (B, N_ctx, d_ctx) cross-modal context.
            node_mask: (B, N) valid node mask.
            context_mask: (B, N_ctx) valid context mask.

        Returns:
            (x_{t-1}, a_{t-1}) denoised by one step.
        """
        eps_pred, edge_logits = self.backbone(
            x_t,
            a_t,
            t,
            hdse_output,
            context,
            node_mask,
            context_mask,
        )
        sch = self.scheduler

        # --- Coordinate posterior ---
        sqrt_ab = sch._gather(sch.sqrt_alphas_cumprod, t)
        sqrt_1mab = sch._gather(sch.sqrt_one_minus_alphas_cumprod, t)
        x0_pred = (x_t - sqrt_1mab * eps_pred) / sqrt_ab.clamp(min=1e-8)

        coef1 = sch._gather(sch.posterior_mean_coef1, t)
        coef2 = sch._gather(sch.posterior_mean_coef2, t)
        mean = coef1 * x0_pred + coef2 * x_t

        log_var = sch._gather(sch.posterior_log_variance_clipped, t)
        noise = torch.randn_like(x_t)
        nonzero = (t > 0).float().view(-1, 1, 1)
        x_prev = mean + nonzero * (0.5 * log_var).exp() * noise

        # --- Adjacency: predict clean, re-noise to t-1 ---
        a0_pred = torch.sigmoid(edge_logits)
        t_prev = (t - 1).clamp(min=0)
        ab_prev = sch.get_alpha_bar(t_prev)  # (B, 1, 1)

        keep = torch.rand_like(a0_pred) < ab_prev
        rand_adj = (torch.rand_like(a0_pred) > 0.5).float()
        a_prev = torch.where(keep, (a0_pred > 0.5).float(), rand_adj)

        # At t=0 just threshold
        at_zero = (t == 0).float().view(-1, 1, 1)
        a_prev = at_zero * (a0_pred > 0.5).float() + (1.0 - at_zero) * a_prev

        # Symmetrize and remove self-loops
        a_prev = _symmetrize(a_prev)

        # Mask padded nodes
        if node_mask is not None:
            x_prev = x_prev * node_mask.unsqueeze(-1).float()
            pm = (node_mask.unsqueeze(-1) & node_mask.unsqueeze(-2)).float()
            a_prev = a_prev * pm

        return x_prev, a_prev

    # ------------------------------------------------------------------
    # DDIM sampling (D-006)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def ddim_sample(
        self,
        num_nodes: int,
        batch_size: int = 1,
        num_steps: int = 50,
        context: torch.Tensor | None = None,
        node_mask: torch.Tensor | None = None,
        context_mask: torch.Tensor | None = None,
        eta: float = 0.0,
        hdse: HDSE | None = None,
        device: torch.device | None = None,
    ) -> RefinedStructuralGraph:
        """DDIM sampling for fast inference.

        Generates a structural graph from pure noise in ``num_steps`` steps
        (default 50 vs 1000 for full DDPM) using a stride of T/num_steps.

        Args:
            num_nodes: Number of nodes N per graph.
            batch_size: Batch dimension B.
            num_steps: Number of DDIM sampling steps.
            context: (B, N_ctx, d_ctx) cross-modal context from tokenizer.
            node_mask: (B, N) valid node mask.
            context_mask: (B, N_ctx) valid context mask.
            eta: DDIM stochasticity (0 = deterministic, 1 = DDPM).
            hdse: Optional HDSE module (recomputed at each step).
            device: Target device.

        Returns:
            :class:`RefinedStructuralGraph` with denoised positions/adjacency.
        """
        # Lazy import to avoid circular / missing-package issues at module load.
        from docs.interfaces.diffusion_output import (
            JunctionType,
            RefinedStructuralGraph,
        )

        if device is None:
            device = next(self.backbone.parameters()).device
        total_t = self.scheduler.num_timesteps

        # Build timestep schedule with uniform stride
        stride = max(total_t // num_steps, 1)
        timesteps = list(range(total_t - 1, -1, -stride))
        if timesteps[-1] != 0:
            timesteps.append(0)

        # Start from pure noise
        x_t = torch.randn(batch_size, num_nodes, 2, device=device)
        a_t = _symmetrize(
            (torch.rand(batch_size, num_nodes, num_nodes, device=device) > 0.5).float()
        )

        if node_mask is None:
            node_mask = torch.ones(batch_size, num_nodes, dtype=torch.bool, device=device)
        cm = node_mask.unsqueeze(-1).float()
        pm = (node_mask.unsqueeze(-1) & node_mask.unsqueeze(-2)).float()
        x_t = x_t * cm
        a_t = a_t * pm

        edge_logits_last: torch.Tensor | None = None

        for i, t_val in enumerate(timesteps):
            t = torch.full((batch_size,), t_val, device=device, dtype=torch.long)

            # HDSE from current adjacency
            hdse_out = hdse(a_t, x_t, node_mask) if hdse is not None else None

            eps_pred, edge_logits = self.backbone(
                x_t,
                a_t,
                t,
                hdse_out,
                context,
                node_mask,
                context_mask,
            )
            edge_logits_last = edge_logits

            # --- DDIM coordinate update ---
            ab_t = self.scheduler._gather(self.scheduler.alphas_cumprod, t)
            x0_pred = (x_t - (1.0 - ab_t).sqrt() * eps_pred) / ab_t.sqrt().clamp(min=1e-8)

            is_last = i == len(timesteps) - 1
            if not is_last:
                t_prev = torch.full(
                    (batch_size,),
                    timesteps[i + 1],
                    device=device,
                    dtype=torch.long,
                )
                ab_prev = self.scheduler._gather(self.scheduler.alphas_cumprod, t_prev)
            else:
                ab_prev = torch.ones_like(ab_t)

            # Sigma and direction
            sigma = (
                eta
                * (
                    (1.0 - ab_prev)
                    / (1.0 - ab_t).clamp(min=1e-8)
                    * (1.0 - ab_t / ab_prev.clamp(min=1e-8))
                )
                .clamp(min=0.0)
                .sqrt()
            )
            dir_xt = (1.0 - ab_prev - sigma**2).clamp(min=0.0).sqrt() * eps_pred
            x_t = ab_prev.sqrt() * x0_pred + dir_xt
            if eta > 0 and not is_last:
                x_t = x_t + sigma * torch.randn_like(x_t)

            # --- Adjacency update ---
            a0_pred = torch.sigmoid(edge_logits)
            if not is_last:
                keep = torch.rand_like(a0_pred) < ab_prev
                rand_a = (torch.rand_like(a0_pred) > 0.5).float()
                a_t = torch.where(keep, (a0_pred > 0.5).float(), rand_a)
            else:
                a_t = (a0_pred > 0.5).float()
            a_t = _symmetrize(a_t)

            x_t = x_t * cm
            a_t = a_t * pm

        # --- Build RefinedStructuralGraph ---
        assert edge_logits_last is not None
        ei_parts: list[torch.Tensor] = []
        el_parts: list[torch.Tensor] = []
        bi_parts: list[torch.Tensor] = []

        for b in range(batch_size):
            nv = int(node_mask[b].sum().item())
            src, dst = torch.where(a_t[b, :nv, :nv] > 0.5)
            offset = b * num_nodes
            ei_parts.append(torch.stack([src + offset, dst + offset]))
            el_parts.append(edge_logits_last[b, src, dst])
            bi_parts.append(torch.full((nv,), b, device=device, dtype=torch.long))

        edge_index = (
            torch.cat(ei_parts, dim=1)
            if ei_parts
            else torch.empty(2, 0, device=device, dtype=torch.long)
        )
        edge_lgt = torch.cat(el_parts) if el_parts else torch.empty(0, device=device)
        batch_idx = (
            torch.cat(bi_parts) if bi_parts else torch.empty(0, device=device, dtype=torch.long)
        )

        junction_types = [
            [JunctionType.UNCLASSIFIED] * int(node_mask[b].sum().item()) for b in range(batch_size)
        ]

        return RefinedStructuralGraph(
            node_positions=x_t,
            adjacency_logits=edge_logits_last,
            node_mask=node_mask,
            edge_index=edge_index,
            edge_logits=edge_lgt,
            junction_types=junction_types,
            node_features=torch.zeros(
                batch_size,
                num_nodes,
                self.backbone.d_model,
                device=device,
            ),
            denoising_step=0,
            total_steps=self.scheduler.num_timesteps,
            noise_level=0.0,
            context_embeddings=context,
            batch_indices=batch_idx,
        )

    # ------------------------------------------------------------------
    # VLB loss (D-007)
    # ------------------------------------------------------------------

    def compute_vlb_loss(
        self,
        x_0: torch.Tensor,
        a_0: torch.Tensor,
        forward_diffusion: ForwardDiffusion,
        hdse: HDSE | None,
        context: torch.Tensor | None,
        node_mask: torch.Tensor | None,
        context_mask: torch.Tensor | None = None,
    ) -> DiffusionLoss:
        """Compute the variational lower bound training loss.

        L = E_t E_eps [ ||eps - eps_theta(G_t, t, c)||^2 ] + BCE(edge_logits, a_0)

        Samples a random timestep, noises via the forward process, predicts
        noise via the backbone, and returns coordinate MSE + adjacency BCE.

        Args:
            x_0: (B, N, 2) clean coordinates.
            a_0: (B, N, N) clean adjacency.
            forward_diffusion: Forward noising process.
            hdse: Optional HDSE module.
            context: (B, N_ctx, d_ctx) cross-modal context.
            node_mask: (B, N) valid node mask.
            context_mask: (B, N_ctx) valid context mask.

        Returns:
            :class:`DiffusionLoss` with total, coordinate, and adjacency losses.
        """
        bsz, n, _ = x_0.shape
        device = x_0.device

        t = self.scheduler.sample_timesteps(bsz, device)
        fwd = forward_diffusion(x_0, a_0, t, node_mask)

        # HDSE from noised graph
        hdse_out = hdse(fwd.a_t, fwd.x_t, node_mask) if hdse is not None else None

        eps_pred, edge_logits = self.backbone(
            fwd.x_t,
            fwd.a_t,
            t,
            hdse_out,
            context,
            node_mask,
            context_mask,
        )

        # --- Coordinate loss: MSE on noise ---
        if node_mask is not None:
            cm = node_mask.unsqueeze(-1).float()
            coord_loss = nnf.mse_loss(eps_pred * cm, fwd.epsilon * cm)
        else:
            coord_loss = nnf.mse_loss(eps_pred, fwd.epsilon)

        # --- Adjacency loss: BCE (upper-triangle only to avoid double count) ---
        triu = torch.triu(torch.ones(n, n, device=device), diagonal=1).unsqueeze(0)
        if node_mask is not None:
            pm = (node_mask.unsqueeze(-1) & node_mask.unsqueeze(-2)).float()
            weight = pm * triu
        else:
            weight = triu.expand(bsz, -1, -1)
        denom = weight.sum().clamp(min=1.0)
        adj_loss = (
            nnf.binary_cross_entropy_with_logits(
                edge_logits,
                a_0,
                weight=weight,
                reduction="sum",
            )
            / denom
        )

        return DiffusionLoss(
            total=coord_loss + adj_loss,
            coordinate_loss=coord_loss,
            adjacency_loss=adj_loss,
            t=t,
        )


# ---------------------------------------------------------------------------
# Top-Level Model
# ---------------------------------------------------------------------------


class GraphDiffusionModel(nn.Module):
    """Complete graph diffusion model: forward + reverse + HDSE.

    Wires the scheduler, forward process, HDSE, transformer backbone, and
    reverse diffusion into a single ``nn.Module``.  Training uses
    ``forward()`` (VLB loss); inference uses ``sample()`` (DDIM).

    Args:
        config: Diffusion configuration.
    """

    def __init__(self, config: DiffusionConfig) -> None:
        super().__init__()
        self.config = config

        self.scheduler = DiffusionScheduler(
            num_timesteps=config.timesteps_train,
            schedule_type=config.noise_schedule.value,
        )
        self.forward_diffusion = ForwardDiffusion(self.scheduler)
        self.hdse: HDSE | None = HDSE(config) if config.use_hdse else None
        self.backbone = GraphTransformerBackbone(config)
        self.reverse = ReverseDiffusion(self.backbone, self.scheduler)

    def forward(
        self,
        x_0: torch.Tensor,
        a_0: torch.Tensor,
        context: torch.Tensor | None = None,
        node_mask: torch.Tensor | None = None,
        context_mask: torch.Tensor | None = None,
    ) -> DiffusionLoss:
        """Training forward pass: compute VLB loss.

        Args:
            x_0: (B, N, 2) clean node coordinates.
            a_0: (B, N, N) clean adjacency matrix.
            context: (B, N_ctx, d_ctx) cross-modal context.
            node_mask: (B, N) valid node mask.
            context_mask: (B, N_ctx) valid context mask.

        Returns:
            :class:`DiffusionLoss` with all loss components.
        """
        return self.reverse.compute_vlb_loss(
            x_0,
            a_0,
            self.forward_diffusion,
            self.hdse,
            context,
            node_mask,
            context_mask,
        )

    @torch.no_grad()
    def sample(
        self,
        num_nodes: int,
        batch_size: int = 1,
        num_steps: int | None = None,
        context: torch.Tensor | None = None,
        node_mask: torch.Tensor | None = None,
        context_mask: torch.Tensor | None = None,
        device: torch.device | None = None,
    ) -> RefinedStructuralGraph:
        """Generate a structural graph via DDIM sampling.

        Args:
            num_nodes: Number of nodes per graph.
            batch_size: Batch size.
            num_steps: DDIM steps (default: ``config.timesteps_inference``).
            context: (B, N_ctx, d_ctx) cross-modal context.
            node_mask: (B, N) valid node mask.
            context_mask: (B, N_ctx) valid context mask.
            device: Target device.

        Returns:
            :class:`RefinedStructuralGraph` with predicted positions and adjacency.
        """
        if num_steps is None:
            num_steps = self.config.timesteps_inference
        return self.reverse.ddim_sample(
            num_nodes=num_nodes,
            batch_size=batch_size,
            num_steps=num_steps,
            context=context,
            node_mask=node_mask,
            context_mask=context_mask,
            eta=0.0,
            hdse=self.hdse,
            device=device,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _symmetrize(a: torch.Tensor) -> torch.Tensor:
    """Symmetrize adjacency and zero diagonal (no self-loops)."""
    upper = torch.triu(a, diagonal=1)
    sym = upper + upper.transpose(-2, -1)
    diag = torch.eye(a.size(-1), device=a.device, dtype=a.dtype).unsqueeze(0)
    return sym * (1.0 - diag)
