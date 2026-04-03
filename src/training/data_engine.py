"""TR-002: Dataset loaders for Axon pre-training.

Provides PyTorch Dataset implementations for four floor-plan datasets:
    1. ArchCAD400K  -- ~41K vector floor plans (primary, JSON zip + pre-processed shards)
    2. FloorPlanCAD -- 15K+ CAD drawings (tar.xz archives)
    3. MLSTRUCT-FP  -- 935 raster floor plan images
    4. ResPlan      -- 17K structured plans with room polygons (pickle)

Plus a CombinedFloorPlanDataset that weighted-samples across datasets.

All datasets accept a ``data_root`` parameter for portability between local
dev and Google Colab (``/content/drive/MyDrive/axon_data/``).

Reference: ARCHITECTURE.md, MODEL_SPEC.md §Pre-Training (MPM).
"""

from __future__ import annotations

import json
import logging
import pickle
import tarfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Maximum number of primitives (edges/tokens) per sample.  Samples with more
# are truncated; shorter ones are padded.
DEFAULT_MAX_PRIMITIVES = 2048

# Entity type string -> integer index.  Matches tokenizer convention.
ENTITY_TYPE_TO_IDX: dict[str, int] = {
    "LINE": 0,
    "ARC": 1,
    "CIRCLE": 2,
    "ELLIPSE": 3,
    "POLYLINE": 4,
    "SPLINE": 5,
}
NUM_ENTITY_TYPES = len(ENTITY_TYPE_TO_IDX)

# Default repo-relative dataset root.
_DEFAULT_DATA_ROOT = Path("datasets")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_data_root(data_root: str | Path | None) -> Path:
    """Resolve the dataset root directory.

    Args:
        data_root: Explicit path, or *None* for the repo-relative default.

    Returns:
        Resolved ``Path`` object.
    """
    if data_root is None:
        return _DEFAULT_DATA_ROOT
    return Path(data_root)


def _normalize_coords(coords: np.ndarray) -> np.ndarray:
    """Normalize coordinates to [0, 1] using per-sample min/max.

    Args:
        coords: (N, D) coordinate array.

    Returns:
        Normalized array with same shape.
    """
    if len(coords) == 0:
        return coords
    lo = coords.min(axis=0)
    hi = coords.max(axis=0)
    span = hi - lo
    span[span < 1e-8] = 1.0  # avoid division by zero
    return (coords - lo) / span


def _entity_to_feature(entity: dict[str, Any]) -> np.ndarray | None:
    """Convert a single archcad400k JSON entity to a feature vector.

    LINE  -> [x_start, y_start, x_end, y_end, entity_type_idx, 0, 0]
    ARC   -> [cx, cy, radius, start_angle, end_angle, entity_type_idx, 0]

    Returns *None* for unsupported entity types.
    """
    etype = entity.get("type", "").upper()
    idx = ENTITY_TYPE_TO_IDX.get(etype)
    if idx is None:
        return None

    if etype == "LINE":
        start = entity.get("start", [0.0, 0.0])
        end = entity.get("end", [0.0, 0.0])
        return np.array(
            [start[0], start[1], end[0], end[1], float(idx), 0.0, 0.0],
            dtype=np.float32,
        )

    if etype == "ARC":
        center = entity.get("center", [0.0, 0.0])
        radius = entity.get("radius", 0.0)
        start_angle = entity.get("start_angle", 0.0)
        end_angle = entity.get("end_angle", 0.0)
        return np.array(
            [center[0], center[1], radius, start_angle, end_angle, float(idx), 0.0],
            dtype=np.float32,
        )

    if etype == "CIRCLE":
        center = entity.get("center", [0.0, 0.0])
        radius = entity.get("radius", 0.0)
        return np.array(
            [center[0], center[1], radius, 0.0, 2.0 * np.pi, float(idx), 0.0],
            dtype=np.float32,
        )

    # Fallback for other supported types -- store as generic point pair.
    start = entity.get("start", entity.get("center", [0.0, 0.0]))
    end = entity.get("end", start)
    return np.array(
        [start[0], start[1], end[0], end[1], float(idx), 0.0, 0.0],
        dtype=np.float32,
    )


def _parse_archcad_json(raw: bytes, max_primitives: int) -> dict[str, torch.Tensor]:
    """Parse a single archcad400k JSON file into padded feature tensors.

    Args:
        raw: Raw bytes of the JSON file.
        max_primitives: Pad/truncate to this length.

    Returns:
        Dict with ``entities`` (max_primitives, 7) float32 tensor,
        ``attention_mask`` (max_primitives,) bool tensor,
        ``entity_types`` (max_primitives,) int64 tensor.
    """
    data = json.loads(raw)
    entity_list = data.get("entities", [])

    features: list[np.ndarray] = []
    for ent in entity_list:
        feat = _entity_to_feature(ent)
        if feat is not None:
            features.append(feat)

    n_valid = min(len(features), max_primitives)
    entities = np.zeros((max_primitives, 7), dtype=np.float32)
    mask = np.zeros(max_primitives, dtype=bool)
    types = np.zeros(max_primitives, dtype=np.int64)

    if n_valid > 0:
        stacked = np.stack(features[:n_valid])
        entities[:n_valid] = stacked
        mask[:n_valid] = True
        types[:n_valid] = stacked[:, 4].astype(np.int64)

        # Normalize coordinate columns to [0, 1]
        coords = entities[:n_valid, :4]  # first 4 cols are coordinates
        entities[:n_valid, :4] = _normalize_coords(coords)

    return {
        "entities": torch.from_numpy(entities),
        "attention_mask": torch.from_numpy(mask),
        "entity_types": torch.from_numpy(types),
    }


# ---------------------------------------------------------------------------
# 1. ArchCAD400K
# ---------------------------------------------------------------------------


class ArchCAD400KDataset(Dataset):
    """Load archcad400k vector entities from pre-processed shards or JSON zip.

    **Primary mode** (fast, for training): reads ``.pt`` shard files from
    ``<data_root>/archcad400k/processed/shard_XXX.pt``.

    **Fallback mode** (on-the-fly): reads individual JSON files from
    ``<data_root>/archcad400k/json.zip`` when shards are unavailable.

    Args:
        data_root: Root directory containing ``archcad400k/``.
        use_shards: Prefer pre-processed shards (default ``True``).
        max_primitives: Pad/truncate entity sequences to this length.
    """

    def __init__(
        self,
        data_root: str | Path | None = None,
        use_shards: bool = True,
        max_primitives: int = DEFAULT_MAX_PRIMITIVES,
    ) -> None:
        super().__init__()
        self.data_root = _resolve_data_root(data_root)
        self.max_primitives = max_primitives
        self._shard_mode = False

        base = self.data_root / "archcad400k"
        shard_dir = base / "processed"

        if use_shards and shard_dir.exists():
            shard_files = sorted(shard_dir.glob("shard_*.pt"))
            if shard_files:
                self._init_shard_mode(shard_files)
                return

        # Fallback to JSON zip.
        zip_path = base / "json.zip"
        if zip_path.exists():
            self._init_zip_mode(zip_path)
        else:
            logger.warning(
                "ArchCAD400K: neither shards in %s nor json.zip at %s found. "
                "Dataset will be empty.",
                shard_dir,
                zip_path,
            )
            self._file_names: list[str] = []
            self._zip_path: Path | None = None

    # -- Shard mode ---------------------------------------------------------

    def _init_shard_mode(self, shard_files: list[Path]) -> None:
        """Load pre-processed shard metadata.

        Shards are packed per-drawing: ``features`` is a flat tensor of all
        primitives across drawings, and ``drawing_offsets`` marks where each
        drawing starts.  The number of *drawings* (not primitives) is what
        we count as samples.
        """
        self._shard_mode = True
        self._shard_files = shard_files

        self._shard_lengths: list[int] = []
        self._shard_cache: dict[int, dict] = {}
        total = 0
        for sf in shard_files:
            peek = torch.load(sf, map_location="cpu", weights_only=True)
            if "drawing_offsets" in peek:
                # Number of drawings = len(offsets) - 1
                n = peek["drawing_offsets"].size(0) - 1
            elif "drawing_names" in peek and isinstance(peek["drawing_names"], list):
                n = len(peek["drawing_names"])
            else:
                # Fallback: assume first tensor's dim-0 is sample count
                n = next(iter(peek.values())).size(0) if peek else 0
            self._shard_lengths.append(n)
            total += n
            del peek

        self._total_len = total
        logger.info(
            "ArchCAD400K shard mode: %d shards, %d drawings total",
            len(shard_files),
            total,
        )

    def _get_shard_sample(self, idx: int) -> dict[str, torch.Tensor]:
        """Retrieve a single drawing's primitives from shard storage.

        Each shard packs multiple drawings into flat ``features`` and
        ``labels`` tensors, indexed by ``drawing_offsets``.  This method
        slices out the primitives for one drawing and pads/truncates to
        ``max_primitives``.
        """
        offset = 0
        for shard_idx, length in enumerate(self._shard_lengths):
            if idx < offset + length:
                local_idx = idx - offset
                if shard_idx not in self._shard_cache:
                    self._shard_cache[shard_idx] = torch.load(
                        self._shard_files[shard_idx],
                        map_location="cpu",
                        weights_only=True,
                    )
                shard = self._shard_cache[shard_idx]
                return self._extract_drawing(shard, local_idx)
            offset += length
        msg = f"Index {idx} out of range for {self._total_len} shard samples"
        raise IndexError(msg)

    def _extract_drawing(
        self, shard: dict, drawing_idx: int
    ) -> dict[str, torch.Tensor]:
        """Extract and pad a single drawing from a packed shard."""
        offsets = shard["drawing_offsets"]
        start = int(offsets[drawing_idx])
        end = int(offsets[drawing_idx + 1])

        raw_features = shard["features"][start:end]  # (n_prims, 12)
        raw_labels = shard["labels"][start:end]  # (n_prims,)
        n_prims = raw_features.size(0)

        # Map 12-dim shard features to 7-dim entity format:
        # Take first 4 cols as coordinates (x1, y1, x2, y2), col 4 as type,
        # and zero-pad remaining 2 dims.
        n_feat = raw_features.size(1)
        if n_feat >= 7:
            feat7 = raw_features[:, :7]
        else:
            feat7 = torch.zeros(n_prims, 7, dtype=torch.float32)
            feat7[:, :min(n_feat, 4)] = raw_features[:, :min(n_feat, 4)]
            if n_feat > 4:
                feat7[:, 4] = raw_features[:, 4]

        # Normalize coordinates to [0, 1]
        coords = feat7[:, :4]
        if coords.numel() > 0:
            cmin = coords.min()
            cmax = coords.max()
            if cmax - cmin > 1e-6:
                feat7[:, :4] = (coords - cmin) / (cmax - cmin)

        # Pad or truncate to max_primitives
        mp = self.max_primitives
        entities = torch.zeros(mp, 7, dtype=torch.float32)
        mask = torch.zeros(mp, dtype=torch.bool)
        types = torch.zeros(mp, dtype=torch.int64)

        n = min(n_prims, mp)
        entities[:n] = feat7[:n]
        mask[:n] = True
        types[:n] = raw_labels[:n]

        return {
            "entities": entities,
            "attention_mask": mask,
            "entity_types": types,
        }

    # -- Zip mode -----------------------------------------------------------

    def _init_zip_mode(self, zip_path: Path) -> None:
        """Index JSON file names inside the zip archive."""
        self._zip_path = zip_path
        self._zip_file: zipfile.ZipFile | None = None
        with zipfile.ZipFile(zip_path, "r") as zf:
            self._file_names = sorted(
                n for n in zf.namelist() if n.endswith(".json") and not n.startswith("__MACOSX")
            )
        logger.info("ArchCAD400K zip mode: %d JSON files in %s", len(self._file_names), zip_path)

    def _get_zip_sample(self, idx: int) -> dict[str, torch.Tensor]:
        """Parse a single JSON from the zip."""
        # Lazy-open: ZipFile is not fork-safe, so each worker opens its own.
        if self._zip_file is None:
            assert self._zip_path is not None
            self._zip_file = zipfile.ZipFile(self._zip_path, "r")
        raw = self._zip_file.read(self._file_names[idx])
        return _parse_archcad_json(raw, self.max_primitives)

    # -- Dataset interface --------------------------------------------------

    def __len__(self) -> int:
        if self._shard_mode:
            return self._total_len
        return len(self._file_names)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Return a single floor-plan sample.

        Returns:
            Dict with keys:
                ``entities``: (max_primitives, 7) float32
                ``attention_mask``: (max_primitives,) bool
                ``entity_types``: (max_primitives,) int64
                ``metadata``: dict with ``dataset`` name and ``index``
        """
        if self._shard_mode:
            sample = self._get_shard_sample(idx)
        else:
            sample = self._get_zip_sample(idx)

        sample["metadata"] = {"dataset": "archcad400k", "index": idx}
        return sample


# ---------------------------------------------------------------------------
# 2. FloorPlanCAD
# ---------------------------------------------------------------------------


class FloorPlanCADDataset(Dataset):
    """Load FloorPlanCAD from tar.xz archives.

    Lazily streams files from ``train-00.tar.xz``, ``train-01.tar.xz``, and
    ``test-00.tar.xz``. Archives are indexed at init time so individual
    samples can be seeked by offset.

    Args:
        data_root: Root directory containing ``FLOORPLANCAD/``.
        split: ``"train"`` or ``"test"``.
        max_primitives: Pad/truncate entity sequences to this length.
    """

    def __init__(
        self,
        data_root: str | Path | None = None,
        split: str = "train",
        max_primitives: int = DEFAULT_MAX_PRIMITIVES,
    ) -> None:
        super().__init__()
        self.data_root = _resolve_data_root(data_root)
        self.split = split
        self.max_primitives = max_primitives

        base = self.data_root / "FLOORPLANCAD"
        tar_files = sorted(base.glob(f"{split}-*.tar.xz"))

        self._members: list[tuple[Path, str]] = []  # (archive_path, member_name)
        for tf in tar_files:
            try:
                with tarfile.open(tf, "r:xz") as tar:
                    for member in tar.getmembers():
                        if member.isfile() and (
                            member.name.endswith(".json")
                            or member.name.endswith(".svg")
                            or member.name.endswith(".png")
                        ):
                            self._members.append((tf, member.name))
            except Exception:
                logger.warning("Failed to index archive %s", tf, exc_info=True)

        logger.info(
            "FloorPlanCAD %s: %d files from %d archives",
            split,
            len(self._members),
            len(tar_files),
        )

        # Cache of opened tar file handles per worker.
        self._tar_handles: dict[Path, tarfile.TarFile] = {}

    def _open_tar(self, archive_path: Path) -> tarfile.TarFile:
        """Open or reuse a tar file handle.

        Handles are kept open for the lifetime of the dataset instance to
        avoid repeated decompression.  Callers should not close these
        handles directly.
        """
        if archive_path not in self._tar_handles:
            handle = tarfile.open(archive_path, "r:xz")  # noqa: SIM115
            self._tar_handles[archive_path] = handle
        return self._tar_handles[archive_path]

    def __len__(self) -> int:
        return len(self._members)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Return a floor-plan sample from a tar.xz archive.

        Returns:
            Dict with ``entities``, ``attention_mask``, ``entity_types``
            tensors, plus ``metadata``.
        """
        archive_path, member_name = self._members[idx]
        tar = self._open_tar(archive_path)

        try:
            member = tar.getmember(member_name)
            fobj = tar.extractfile(member)
            if fobj is None:
                return self._empty_sample(idx)
            raw = fobj.read()
        except Exception:
            logger.warning("Failed to read %s from %s", member_name, archive_path)
            return self._empty_sample(idx)

        if member_name.endswith(".json"):
            sample = _parse_archcad_json(raw, self.max_primitives)
        elif member_name.endswith(".png"):
            sample = self._parse_image(raw)
        else:
            sample = self._empty_sample(idx)

        sample["metadata"] = {
            "dataset": "floorplancad",
            "split": self.split,
            "index": idx,
            "member": member_name,
        }
        return sample

    def _parse_image(self, raw: bytes) -> dict[str, torch.Tensor]:
        """Minimal raster parsing -- store raw bytes as metadata placeholder."""
        # For pre-training, raster images are optional cross-modal context.
        # Return empty entities with raster flag.
        return {
            "entities": torch.zeros(self.max_primitives, 7, dtype=torch.float32),
            "attention_mask": torch.zeros(self.max_primitives, dtype=torch.bool),
            "entity_types": torch.zeros(self.max_primitives, dtype=torch.int64),
        }

    def _empty_sample(self, idx: int) -> dict[str, torch.Tensor]:
        """Return an empty sample as a graceful fallback."""
        return {
            "entities": torch.zeros(self.max_primitives, 7, dtype=torch.float32),
            "attention_mask": torch.zeros(self.max_primitives, dtype=torch.bool),
            "entity_types": torch.zeros(self.max_primitives, dtype=torch.int64),
            "metadata": {"dataset": "floorplancad", "index": idx, "empty": True},
        }


# ---------------------------------------------------------------------------
# 3. MLSTRUCT-FP
# ---------------------------------------------------------------------------


class MLStructDataset(Dataset):
    """Load MLSTRUCT-FP raster floor plan images.

    Reads 935 PNG images from ``<data_root>/MLSTRUCT-FP_v1/``.
    Returns images as (C, H, W) float32 tensors normalized to [0, 1].

    For MPM pre-training, raster images are optional cross-modal context
    rather than primary training targets.

    Args:
        data_root: Root directory containing ``MLSTRUCT-FP_v1/``.
        image_size: Resize images to ``(image_size, image_size)`` if not None.
    """

    def __init__(
        self,
        data_root: str | Path | None = None,
        image_size: int | None = 512,
    ) -> None:
        super().__init__()
        self.data_root = _resolve_data_root(data_root)
        self.image_size = image_size

        base = self.data_root / "MLSTRUCT-FP_v1"
        self._image_paths = sorted(base.glob("*.png")) if base.exists() else []
        logger.info("MLSTRUCT-FP: %d images in %s", len(self._image_paths), base)

    def __len__(self) -> int:
        return len(self._image_paths)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Return a raster floor plan image.

        Returns:
            Dict with ``image`` (C, H, W) float32 tensor and ``metadata``.
        """
        img_path = self._image_paths[idx]

        try:
            # Use PIL for image loading -- widely available, no extra deps.
            from PIL import Image

            img = Image.open(img_path).convert("RGB")
            if self.image_size is not None:
                img = img.resize((self.image_size, self.image_size), Image.BILINEAR)

            arr = np.array(img, dtype=np.float32) / 255.0  # (H, W, C)
            tensor = torch.from_numpy(arr).permute(2, 0, 1)  # (C, H, W)
        except Exception:
            logger.warning("Failed to load image %s", img_path, exc_info=True)
            c = 3
            h = w = self.image_size or 512
            tensor = torch.zeros(c, h, w, dtype=torch.float32)

        return {
            "image": tensor,
            "metadata": {
                "dataset": "mlstruct",
                "index": idx,
                "path": str(img_path),
            },
        }


# ---------------------------------------------------------------------------
# 4. ResPlan
# ---------------------------------------------------------------------------

# Room keys expected in each ResPlan dict.
_RESPLAN_ROOM_KEYS = [
    "wall",
    "bathroom",
    "bedroom",
    "kitchen",
    "living",
    "door",
    "window",
]

# Room label string -> integer index (for classification targets).
ROOM_LABEL_TO_IDX: dict[str, int] = {k: i for i, k in enumerate(_RESPLAN_ROOM_KEYS)}
NUM_ROOM_LABELS = len(ROOM_LABEL_TO_IDX)


def _multipolygon_to_coords(geom: Any) -> np.ndarray:
    """Extract exterior coordinates from a Shapely MultiPolygon/Polygon.

    Args:
        geom: A Shapely geometry object.

    Returns:
        (N, 2) float64 array of boundary coordinates, or empty (0, 2) array.
    """
    if geom is None or geom.is_empty:
        return np.empty((0, 2), dtype=np.float64)

    coords_list: list[np.ndarray] = []
    # Handle both Polygon and MultiPolygon.
    geoms = list(geom.geoms) if hasattr(geom, "geoms") else [geom]
    for poly in geoms:
        if hasattr(poly, "exterior"):
            ring = np.array(poly.exterior.coords, dtype=np.float64)
            coords_list.append(ring)

    if not coords_list:
        return np.empty((0, 2), dtype=np.float64)
    return np.concatenate(coords_list, axis=0)


class ResPlanDataset(Dataset):
    """Load ResPlan structured floor plans from pickle.

    Each sample contains Shapely MultiPolygon geometries for walls, rooms,
    doors, and windows, plus a networkx graph.  This dataset converts them
    into node coordinate arrays, edge lists, and room label arrays
    compatible with Axon's graph format.

    Args:
        data_root: Root directory containing ``ResPlan/``.
        max_nodes: Pad/truncate node arrays to this length.
    """

    def __init__(
        self,
        data_root: str | Path | None = None,
        max_nodes: int = 512,
    ) -> None:
        super().__init__()
        self.data_root = _resolve_data_root(data_root)
        self.max_nodes = max_nodes

        pkl_path = self.data_root / "ResPlan" / "ResPlan.pkl"
        if pkl_path.exists():
            with open(pkl_path, "rb") as f:
                self._records: list[dict[str, Any]] = pickle.load(f)  # trusted data
            logger.info("ResPlan: %d records from %s", len(self._records), pkl_path)
        else:
            logger.warning("ResPlan pickle not found at %s. Dataset will be empty.", pkl_path)
            self._records = []

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Return a structured floor plan sample.

        Returns:
            Dict with:
                ``nodes``: (max_nodes, 2) float32 normalized coordinates
                ``edges``: (max_edges, 2) int64 edge index pairs
                ``node_mask``: (max_nodes,) bool
                ``room_labels``: (max_nodes,) int64 per-node room labels
                ``wall_depth``: scalar float32 tensor
                ``metadata``: dict
        """
        record = self._records[idx]

        # --- Extract nodes from the networkx graph ---
        graph = record.get("graph")
        if graph is not None:
            try:
                node_list = sorted(graph.nodes())
                node_id_to_idx = {nid: i for i, nid in enumerate(node_list)}

                # Extract node positions.
                positions: list[list[float]] = []
                for nid in node_list:
                    ndata = graph.nodes[nid]
                    pos = ndata.get("pos", ndata.get("position", [0.0, 0.0]))
                    if isinstance(pos, (tuple, list)):
                        positions.append([float(pos[0]), float(pos[1])])
                    else:
                        positions.append([0.0, 0.0])

                raw_nodes = np.array(positions, dtype=np.float32)

                # Extract edges.
                edge_list = []
                for u, v in graph.edges():
                    if u in node_id_to_idx and v in node_id_to_idx:
                        edge_list.append([node_id_to_idx[u], node_id_to_idx[v]])
                raw_edges = (
                    np.array(edge_list, dtype=np.int64)
                    if edge_list
                    else np.empty((0, 2), dtype=np.int64)
                )
            except Exception:
                logger.warning("Failed to parse networkx graph for record %d", idx)
                raw_nodes = np.empty((0, 2), dtype=np.float32)
                raw_edges = np.empty((0, 2), dtype=np.int64)
        else:
            # Fallback: extract nodes from wall polygon coordinates.
            wall_geom = record.get("wall")
            wall_coords = _multipolygon_to_coords(wall_geom).astype(np.float32)
            raw_nodes = wall_coords if len(wall_coords) > 0 else np.empty((0, 2), dtype=np.float32)
            # Build sequential edges along polygon boundaries.
            n = len(raw_nodes)
            if n > 1:
                raw_edges = np.stack([np.arange(n - 1), np.arange(1, n)], axis=1).astype(np.int64)
            else:
                raw_edges = np.empty((0, 2), dtype=np.int64)

        # --- Pad/truncate nodes ---
        n_valid = min(len(raw_nodes), self.max_nodes)
        nodes = np.zeros((self.max_nodes, 2), dtype=np.float32)
        node_mask = np.zeros(self.max_nodes, dtype=bool)
        if n_valid > 0:
            nodes[:n_valid] = _normalize_coords(raw_nodes[:n_valid])
            node_mask[:n_valid] = True

        # --- Filter and pad edges ---
        if len(raw_edges) > 0:
            valid_edge_mask = (raw_edges[:, 0] < n_valid) & (raw_edges[:, 1] < n_valid)
            raw_edges = raw_edges[valid_edge_mask]

        max_edges = self.max_nodes * 4  # heuristic upper bound
        n_edges = min(len(raw_edges), max_edges)
        edges = np.zeros((max_edges, 2), dtype=np.int64)
        if n_edges > 0:
            edges[:n_edges] = raw_edges[:n_edges]

        # --- Room labels (per-node, based on polygon containment) ---
        room_labels = np.zeros(self.max_nodes, dtype=np.int64)
        # Default all valid nodes to "wall" (label 0).
        # More precise per-node labeling would require point-in-polygon tests
        # against each room geometry, which we defer to shard preprocessing.

        # --- Wall depth ---
        wall_depth = float(record.get("wall_depth", 0.0))

        return {
            "nodes": torch.from_numpy(nodes),
            "edges": torch.from_numpy(edges),
            "node_mask": torch.from_numpy(node_mask),
            "room_labels": torch.from_numpy(room_labels),
            "wall_depth": torch.tensor(wall_depth, dtype=torch.float32),
            "metadata": {
                "dataset": "resplan",
                "index": idx,
                "unit_type": record.get("unitType", ""),
                "area": float(record.get("area", 0.0)),
                "n_valid_nodes": n_valid,
                "n_valid_edges": n_edges,
            },
        }


# ---------------------------------------------------------------------------
# 5. Combined Dataset
# ---------------------------------------------------------------------------


@dataclass
class DatasetSpec:
    """Specification for a single dataset in the combined sampler.

    Args:
        dataset: A PyTorch Dataset instance.
        weight: Relative sampling weight (higher = sampled more often).
        name: Human-readable name for logging.
    """

    dataset: Dataset
    weight: float = 1.0
    name: str = ""


class CombinedFloorPlanDataset(Dataset):
    """Combines multiple floor-plan datasets with configurable sampling weights.

    Uses a deterministic index mapping: each global index maps to a
    specific (dataset, local_index) pair based on the weight-proportional
    allocation of the combined virtual length.

    Args:
        specs: List of DatasetSpec entries.
        total_virtual_length: Total virtual dataset length. If None,
            defaults to the sum of all dataset lengths.
    """

    def __init__(
        self,
        specs: list[DatasetSpec],
        total_virtual_length: int | None = None,
    ) -> None:
        super().__init__()
        if not specs:
            msg = "CombinedFloorPlanDataset requires at least one dataset"
            raise ValueError(msg)

        self.specs = specs

        # Compute weight-proportional allocation.
        total_weight = sum(s.weight for s in specs)
        if total_weight <= 0:
            msg = "Total sampling weight must be positive"
            raise ValueError(msg)

        real_total = sum(len(s.dataset) for s in specs)
        virtual_total = total_virtual_length if total_virtual_length is not None else real_total

        # Allocate virtual indices per dataset proportionally.
        self._allocations: list[int] = []
        allocated = 0
        for i, s in enumerate(specs):
            if i == len(specs) - 1:
                # Last dataset gets the remainder to avoid rounding issues.
                n = virtual_total - allocated
            else:
                n = max(1, int(virtual_total * s.weight / total_weight))
            self._allocations.append(n)
            allocated += n

        self._total_len = sum(self._allocations)

        # Build cumulative boundaries for O(log n) lookup.
        self._boundaries: list[int] = []
        cumsum = 0
        for n in self._allocations:
            cumsum += n
            self._boundaries.append(cumsum)

        logger.info(
            "CombinedFloorPlanDataset: %d total virtual samples from %d datasets: %s",
            self._total_len,
            len(specs),
            [
                (s.name or f"ds{i}", alloc)
                for i, (s, alloc) in enumerate(zip(specs, self._allocations, strict=False))
            ],
        )

    def __len__(self) -> int:
        return self._total_len

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Map a global index to the appropriate dataset sample.

        Uses modular indexing when the virtual allocation exceeds the
        underlying dataset length (upsampling).
        """
        # Binary search for which dataset this index belongs to.
        ds_idx = 0
        for i, boundary in enumerate(self._boundaries):
            if idx < boundary:
                ds_idx = i
                break

        offset = self._boundaries[ds_idx - 1] if ds_idx > 0 else 0
        local_idx = idx - offset

        ds = self.specs[ds_idx].dataset
        # Modular wrap for upsampling.
        real_len = len(ds)
        if real_len == 0:
            return {"metadata": {"dataset": "empty", "index": idx}}
        local_idx = local_idx % real_len

        sample = ds[local_idx]
        if isinstance(sample, dict):
            sample.setdefault("metadata", {})
            if isinstance(sample["metadata"], dict):
                sample["metadata"]["combined_index"] = idx
                sample["metadata"]["source_dataset"] = self.specs[ds_idx].name or f"ds{ds_idx}"
        return sample


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------


def build_combined_dataset(
    data_root: str | Path | None = None,
    archcad_weight: float = 4.0,
    floorplancad_weight: float = 2.0,
    resplan_weight: float = 1.0,
    max_primitives: int = DEFAULT_MAX_PRIMITIVES,
    max_nodes: int = 512,
) -> CombinedFloorPlanDataset:
    """Build a combined dataset with sensible default weights.

    ArchCAD400K is weighted highest as the primary vector-native dataset.

    Args:
        data_root: Root directory for all datasets.
        archcad_weight: Sampling weight for ArchCAD400K.
        floorplancad_weight: Sampling weight for FloorPlanCAD.
        resplan_weight: Sampling weight for ResPlan.
        max_primitives: Max entity sequence length.
        max_nodes: Max graph nodes for ResPlan.

    Returns:
        A configured CombinedFloorPlanDataset instance.
    """
    specs = [
        DatasetSpec(
            dataset=ArchCAD400KDataset(data_root=data_root, max_primitives=max_primitives),
            weight=archcad_weight,
            name="archcad400k",
        ),
        DatasetSpec(
            dataset=FloorPlanCADDataset(data_root=data_root, max_primitives=max_primitives),
            weight=floorplancad_weight,
            name="floorplancad",
        ),
        DatasetSpec(
            dataset=ResPlanDataset(data_root=data_root, max_nodes=max_nodes),
            weight=resplan_weight,
            name="resplan",
        ),
    ]

    # Filter out empty datasets.
    specs = [s for s in specs if len(s.dataset) > 0]
    if not specs:
        logger.warning("No datasets found at %s — returning ArchCAD400K (empty)", data_root)
        specs = [
            DatasetSpec(
                dataset=ArchCAD400KDataset(data_root=data_root, max_primitives=max_primitives),
                weight=1.0,
                name="archcad400k",
            )
        ]

    return CombinedFloorPlanDataset(specs)
