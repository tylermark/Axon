"""Shared helpers for diffusion module tests.

Provides a small DiffusionConfig and synthetic graph factory
optimized for fast CPU-only testing.
"""

from __future__ import annotations

import torch

from src.pipeline.config import DiffusionConfig, NoiseSchedule


def create_small_diffusion_config(
    use_hdse: bool = True,
    noise_schedule: NoiseSchedule = NoiseSchedule.COSINE,
) -> DiffusionConfig:
    """Config for fast testing — small model, short schedule."""
    return DiffusionConfig(
        d_model=64,
        n_heads=4,
        n_layers=2,
        timesteps_train=100,
        timesteps_inference=10,
        noise_schedule=noise_schedule,
        max_nodes=16,
        use_hdse=use_hdse,
        hdse_max_distance=5,
        dropout=0.0,
    )


def create_synthetic_graph(
    batch_size: int = 2,
    num_nodes: int = 8,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create clean graph data for testing.

    Builds a simple cycle graph where nodes form a regular polygon
    with edges connecting consecutive nodes.

    Returns:
        (x_0, a_0, node_mask):
            x_0: (B, N, 2) node coordinates in [0, 1]
            a_0: (B, N, N) symmetric adjacency (float, {0,1})
            node_mask: (B, N) bool — all True
    """
    gen = torch.Generator().manual_seed(seed)

    # Regular polygon coordinates in [0.2, 0.8]
    angles = torch.linspace(0, 2 * 3.14159265, num_nodes + 1)[:num_nodes]
    x_single = torch.stack(
        [0.5 + 0.3 * torch.cos(angles), 0.5 + 0.3 * torch.sin(angles)],
        dim=-1,
    )  # (N, 2)
    # Add small per-batch perturbation
    x_0 = x_single.unsqueeze(0).expand(batch_size, -1, -1).clone()
    x_0 += torch.randn(batch_size, num_nodes, 2, generator=gen) * 0.02

    # Cycle adjacency: each node connected to its neighbors
    a_single = torch.zeros(num_nodes, num_nodes)
    for i in range(num_nodes):
        j = (i + 1) % num_nodes
        a_single[i, j] = 1.0
        a_single[j, i] = 1.0
    a_0 = a_single.unsqueeze(0).expand(batch_size, -1, -1).clone()

    node_mask = torch.ones(batch_size, num_nodes, dtype=torch.bool)

    return x_0, a_0, node_mask
