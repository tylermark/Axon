"""Stage 3 — Graph Diffusion Engine.

Implements the DDPM-based forward and reverse diffusion process for
joint continuous (node coordinates) and discrete (adjacency) structural
graphs.

Reference: ARCHITECTURE.md §Stage 3, MODEL_SPEC.md §Generative Graph
Denoising Diffusion Engine.
"""

from src.diffusion.forward import ForwardDiffusion, ForwardDiffusionOutput
from src.diffusion.hdse import (
    HDSE,
    HDSEOutput,
    HierarchicalLevelEncoding,
    RandomWalkEncoding,
    ShortestPathEncoding,
)
from src.diffusion.reverse import (
    DiffusionLoss,
    GraphDiffusionModel,
    GraphTransformerBackbone,
    GraphTransformerBlock,
    ReverseDiffusion,
    TimestepEmbedding,
)
from src.diffusion.scheduler import DiffusionScheduler

__all__ = [
    "HDSE",
    "DiffusionLoss",
    "DiffusionScheduler",
    "ForwardDiffusion",
    "ForwardDiffusionOutput",
    "GraphDiffusionModel",
    "GraphTransformerBackbone",
    "GraphTransformerBlock",
    "HDSEOutput",
    "HierarchicalLevelEncoding",
    "RandomWalkEncoding",
    "ReverseDiffusion",
    "ShortestPathEncoding",
    "TimestepEmbedding",
]
