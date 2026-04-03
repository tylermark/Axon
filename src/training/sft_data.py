"""SFT dataset adapter — converts entity-based datasets to SFT batch format.

Bridges the gap between ArchCAD400K/FloorPlanCAD entity format and the
x_0/a_0/raw_features format expected by SFTTrainer.

Entity format (from data_engine):
    entities:       (N, 7) [x1, y1, x2, y2, type, 0, 0]
    attention_mask: (N,) bool
    entity_types:   (N,) int64

SFT batch format (consumed by SFTTrainer._train_step):
    x_0:          (N, 2) node midpoint coordinates
    a_0:          (N, N) adjacency matrix from endpoint proximity
    node_mask:    (N,) bool
    raw_features: dict matching tokenizer's VectorTokenEmbedding input

Reference: TASKS.md TR-003.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

# Endpoint proximity threshold for building adjacency (normalized coords)
_ADJACENCY_THRESHOLD: float = 0.02


class SFTDatasetAdapter(Dataset):
    """Wraps an entity-based dataset for SFT training.

    Converts each sample's entity primitives (line segments) into a
    structural graph (nodes at midpoints, adjacency from shared endpoints)
    plus the raw_features dict for the tokenizer.

    Args:
        base_dataset: An ArchCAD400K or FloorPlanCAD dataset instance.
        max_nodes: Maximum number of nodes (entities) per sample.
        adjacency_threshold: Distance threshold for connecting entities
            via shared endpoints (in normalized coordinate space).
    """

    def __init__(
        self,
        base_dataset: Dataset,
        max_nodes: int = 512,
        adjacency_threshold: float = _ADJACENCY_THRESHOLD,
    ) -> None:
        self.base = base_dataset
        self.max_nodes = max_nodes
        self.threshold = adjacency_threshold

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = self.base[idx]
        entities = sample["entities"]  # (N, 7)
        mask = sample["attention_mask"]  # (N,)
        entity_types = sample.get("entity_types", torch.zeros(entities.shape[0], dtype=torch.long))

        # Truncate to max_nodes
        N = min(entities.shape[0], self.max_nodes)
        entities = entities[:N]
        mask = mask[:N]
        entity_types = entity_types[:N]

        n_valid = int(mask.sum().item())

        # ── x_0: midpoint coordinates (N, 2) ──
        coords = entities[:, :4]  # (N, 4) [x1, y1, x2, y2]
        midpoints = torch.stack([
            (coords[:, 0] + coords[:, 2]) / 2.0,  # (x1 + x2) / 2
            (coords[:, 1] + coords[:, 3]) / 2.0,  # (y1 + y2) / 2
        ], dim=-1)  # (N, 2)

        # Normalize to [0, 1] range
        if n_valid > 0:
            valid_mid = midpoints[:n_valid]
            vmin = valid_mid.min(dim=0).values
            vmax = valid_mid.max(dim=0).values
            span = (vmax - vmin).clamp(min=1e-6)
            midpoints[:n_valid] = (valid_mid - vmin) / span

        x_0 = midpoints

        # ── a_0: adjacency from endpoint proximity (N, N) ──
        a_0 = self._build_adjacency(coords, mask, n_valid)

        # ── raw_features: tokenizer-compatible dict ──
        raw_features = self._build_raw_features(entities, entity_types, mask, n_valid)

        return {
            "x_0": x_0,
            "a_0": a_0,
            "node_mask": mask,
            "raw_features": raw_features,
        }

    def _build_adjacency(
        self,
        coords: torch.Tensor,
        mask: torch.Tensor,
        n_valid: int,
    ) -> torch.Tensor:
        """Build adjacency matrix from endpoint proximity.

        Two entities are connected if any pair of their endpoints
        (start-start, start-end, end-start, end-end) are within threshold.
        """
        N = coords.shape[0]
        a_0 = torch.zeros(N, N, dtype=torch.float32)

        if n_valid <= 1:
            return a_0

        # Extract start/end points for valid entities
        starts = coords[:n_valid, :2]  # (n, 2) [x1, y1]
        ends = coords[:n_valid, 2:4]   # (n, 2) [x2, y2]

        # Normalize endpoints to [0, 1]
        all_pts = torch.cat([starts, ends], dim=0)
        vmin = all_pts.min(dim=0).values
        vmax = all_pts.max(dim=0).values
        span = (vmax - vmin).clamp(min=1e-6)
        starts = (starts - vmin) / span
        ends = (ends - vmin) / span

        # Check all four endpoint combinations for proximity
        for pts_a, pts_b in [
            (starts, starts),
            (starts, ends),
            (ends, starts),
            (ends, ends),
        ]:
            # Pairwise L2 distance
            diff = pts_a.unsqueeze(1) - pts_b.unsqueeze(0)  # (n, n, 2)
            dist = diff.norm(dim=-1)  # (n, n)
            connected = dist < self.threshold
            # Remove self-connections
            connected.fill_diagonal_(False)
            a_0[:n_valid, :n_valid] = torch.maximum(
                a_0[:n_valid, :n_valid],
                connected.float(),
            )

        return a_0

    def _build_raw_features(
        self,
        entities: torch.Tensor,
        entity_types: torch.Tensor,
        mask: torch.Tensor,
        n_valid: int,
    ) -> dict[str, torch.Tensor]:
        """Build the raw_features dict for the tokenizer."""
        N = entities.shape[0]
        coords = entities[:, :4].clone()

        # Normalize coordinates to [0, 1]
        if n_valid > 0:
            valid_coords = coords[:n_valid]
            vmin = valid_coords.min()
            vmax = valid_coords.max()
            span = max(vmax - vmin, 1e-6)
            coords[:n_valid] = (valid_coords - vmin) / span

        # Clamp entity types to valid operator range (0-3)
        op_types = entity_types.clamp(0, 3)

        return {
            "operator_type": op_types,
            "coordinates": coords,
            "stroke_width": torch.ones(N, dtype=torch.float32) * mask.float(),
            "dash_hash": torch.zeros(N, dtype=torch.long),
            "color_rgb": torch.zeros(N, 3, dtype=torch.float32),
            "confidence_wall": mask.float(),
            "attention_mask": mask,
            "raw_coordinates": coords.double(),
        }


class ResPlanSFTAdapter(Dataset):
    """Wraps ResPlanDataset for SFT training.

    ResPlan already provides node positions and edges — just needs
    conversion to dense adjacency and a raw_features placeholder.

    Args:
        base_dataset: A ResPlanDataset instance.
    """

    def __init__(self, base_dataset: Dataset) -> None:
        self.base = base_dataset

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = self.base[idx]
        nodes = sample["nodes"]      # (max_nodes, 2)
        edges = sample["edges"]      # (E, 2)
        node_mask = sample["node_mask"]  # (max_nodes,)

        N = nodes.shape[0]
        n_valid = int(node_mask.sum().item())

        # x_0 is directly the node positions (already normalized)
        x_0 = nodes

        # Build dense adjacency from edge list
        a_0 = torch.zeros(N, N, dtype=torch.float32)
        for i in range(edges.shape[0]):
            src, dst = int(edges[i, 0].item()), int(edges[i, 1].item())
            if 0 <= src < n_valid and 0 <= dst < n_valid:
                a_0[src, dst] = 1.0
                a_0[dst, src] = 1.0

        # Build placeholder raw_features (ResPlan has no drawing primitives)
        # Use node positions as line segment endpoints (degenerate segments)
        coords = torch.zeros(N, 4, dtype=torch.float32)
        coords[:, :2] = nodes
        coords[:, 2:] = nodes

        raw_features = {
            "operator_type": torch.zeros(N, dtype=torch.long),
            "coordinates": coords,
            "stroke_width": node_mask.float(),
            "dash_hash": torch.zeros(N, dtype=torch.long),
            "color_rgb": torch.zeros(N, 3, dtype=torch.float32),
            "confidence_wall": node_mask.float(),
            "attention_mask": node_mask,
            "raw_coordinates": coords.double(),
        }

        return {
            "x_0": x_0,
            "a_0": a_0,
            "node_mask": node_mask,
            "raw_features": raw_features,
        }
