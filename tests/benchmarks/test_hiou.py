"""Benchmark: Hierarchical Intersection over Union (HIoU) for wall extraction.

Q-008: HIoU measures how well predicted wall segments overlap with ground
truth wall rectangles.  Uses Hungarian matching to find the best
assignment and reports mean IoU of matched pairs.

Reference: ARCHITECTURE.md (Evaluation Metrics), MODEL_SPEC.md.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.optimize import linear_sum_assignment

# ---------------------------------------------------------------------------
# HIoU implementation
# ---------------------------------------------------------------------------


def wall_segment_to_rect(
    start: np.ndarray,
    end: np.ndarray,
    thickness: float,
) -> np.ndarray:
    """Convert a wall segment to 4 corner points of its rectangle.

    Args:
        start: Start coordinate, shape (2,).
        end: End coordinate, shape (2,).
        thickness: Wall thickness.

    Returns:
        Array of shape (4, 2) — rectangle corners in CCW order.
    """
    direction = end - start
    length = np.linalg.norm(direction)
    if length < 1e-10:
        return np.zeros((4, 2))
    normal = np.array([-direction[1], direction[0]]) / length
    half_t = thickness / 2.0
    return np.array([
        start + normal * half_t,
        start - normal * half_t,
        end - normal * half_t,
        end + normal * half_t,
    ])


def polygon_iou(poly1: np.ndarray, poly2: np.ndarray) -> float:
    """Compute IoU between two convex polygons using Shapely.

    Args:
        poly1: Vertices of polygon 1, shape (M, 2).
        poly2: Vertices of polygon 2, shape (K, 2).

    Returns:
        IoU in [0, 1].
    """
    shapely_polygon_cls = pytest.importorskip("shapely.geometry").Polygon

    p1 = shapely_polygon_cls(poly1)
    p2 = shapely_polygon_cls(poly2)
    if not p1.is_valid or not p2.is_valid:
        return 0.0
    inter = p1.intersection(p2).area
    union = p1.union(p2).area
    return inter / union if union > 0 else 0.0


def compute_hiou(
    pred_walls: list[tuple[np.ndarray, np.ndarray, float]],
    gt_walls: list[tuple[np.ndarray, np.ndarray, float]],
) -> float:
    """Compute HIoU between predicted and ground truth wall sets.

    Each wall is a tuple of (start_coord, end_coord, thickness).

    Uses Hungarian matching on the IoU cost matrix to find the optimal
    one-to-one assignment, then returns the mean IoU of matched pairs.

    Args:
        pred_walls: Predicted wall segments.
        gt_walls: Ground truth wall segments.

    Returns:
        Mean IoU of Hungarian-matched pairs.  0.0 when either set is empty.
    """
    if len(pred_walls) == 0 or len(gt_walls) == 0:
        return 0.0

    pred_rects = [wall_segment_to_rect(s, e, t) for s, e, t in pred_walls]
    gt_rects = [wall_segment_to_rect(s, e, t) for s, e, t in gt_walls]

    n_pred = len(pred_rects)
    n_gt = len(gt_rects)
    iou_matrix = np.zeros((n_pred, n_gt))
    for i in range(n_pred):
        for j in range(n_gt):
            iou_matrix[i, j] = polygon_iou(pred_rects[i], gt_rects[j])

    # Hungarian matching maximizes IoU → minimize (1 - IoU).
    cost_matrix = 1.0 - iou_matrix
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    matched_iou = iou_matrix[row_ind, col_ind]
    return float(matched_iou.mean())


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

_T = 2.0  # Default wall thickness for tests.


def _wall(x0: float, y0: float, x1: float, y1: float, t: float = _T):
    """Convenience: create a wall tuple."""
    return (np.array([x0, y0]), np.array([x1, y1]), t)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.benchmark
class TestHIoU:
    """Unit tests for the HIoU metric."""

    def test_hiou_perfect_match(self):
        """Identical predicted and GT walls yield HIoU = 1.0."""
        walls = [
            _wall(0, 0, 100, 0),
            _wall(100, 0, 100, 100),
            _wall(100, 100, 0, 100),
            _wall(0, 100, 0, 0),
        ]
        hiou = compute_hiou(walls, walls)
        assert hiou == pytest.approx(1.0, abs=1e-6)

    def test_hiou_no_overlap(self):
        """Completely separate walls yield HIoU ~ 0.0."""
        pred = [_wall(0, 0, 100, 0), _wall(0, 0, 0, 100)]
        gt = [_wall(500, 500, 600, 500), _wall(500, 500, 500, 600)]
        hiou = compute_hiou(pred, gt)
        assert hiou == pytest.approx(0.0, abs=1e-6)

    def test_hiou_partial_overlap(self):
        """Shifted walls produce 0 < HIoU < 1."""
        gt = [_wall(0, 0, 100, 0, 4.0)]
        # Shift horizontally by 25% of the wall length.
        pred = [_wall(25, 0, 125, 0, 4.0)]
        hiou = compute_hiou(pred, gt)
        assert 0.0 < hiou < 1.0

    def test_hiou_empty_pred(self):
        """Empty prediction set returns 0.0."""
        gt = [_wall(0, 0, 100, 0)]
        assert compute_hiou([], gt) == 0.0

    def test_hiou_empty_gt(self):
        """Empty ground truth set returns 0.0."""
        pred = [_wall(0, 0, 100, 0)]
        assert compute_hiou(pred, []) == 0.0


@pytest.mark.slow
@pytest.mark.benchmark
class TestHIoUPipeline:
    """End-to-end: run Layer1Pipeline and compute HIoU on the output."""

    @pytest.fixture(scope="class")
    def pipeline(self):
        from src.pipeline.config import (
            AxonConfig,
            DiffusionConfig,
            NoiseSchedule,
            TokenizerConfig,
        )
        from src.pipeline.config import VisionBackbone as VisionBackboneType
        from src.pipeline.layer1 import Layer1Pipeline

        cfg = AxonConfig(
            tokenizer=TokenizerConfig(
                d_model=256,
                n_heads=8,
                vision_backbone=VisionBackboneType.HRNET_W32,
                dropout=0.0,
            ),
            diffusion=DiffusionConfig(
                d_model=64,
                n_heads=4,
                n_layers=2,
                timesteps_train=100,
                timesteps_inference=10,
                noise_schedule=NoiseSchedule.COSINE,
                max_nodes=16,
                use_hdse=True,
                hdse_max_distance=5,
                dropout=0.0,
            ),
            device="cpu",
        )
        return Layer1Pipeline(config=cfg, device="cpu")

    def test_hiou_on_pipeline_output(self, pipeline, tmp_path):
        """HIoU metric runs end-to-end on pipeline output (untrained)."""
        from tests.fixtures.pdf_factory import create_simple_rect_pdf

        pdf = create_simple_rect_pdf(
            tmp_path / "rect.pdf", x=100, y=100, w=200, h=150, stroke_width=1.5,
        )
        result = pipeline.extract(str(pdf), page_index=0, use_raster=False)

        # Build predicted walls from FinalizedGraph.
        pred_walls = [
            (ws.start_coord, ws.end_coord, ws.thickness)
            for ws in result.wall_segments
        ]

        # Approximate ground truth: the four sides of the rectangle.
        gt_walls = [
            _wall(100, 100, 300, 100, 1.5),
            _wall(300, 100, 300, 250, 1.5),
            _wall(300, 250, 100, 250, 1.5),
            _wall(100, 250, 100, 100, 1.5),
        ]

        # With untrained models quality will be poor.  Just verify the
        # metric runs without error and returns a valid float in [0, 1].
        if len(pred_walls) > 0:
            hiou = compute_hiou(pred_walls, gt_walls)
            assert 0.0 <= hiou <= 1.0
