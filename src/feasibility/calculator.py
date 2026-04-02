"""FS-001: Prefab coverage calculation.

Computes prefab coverage metrics from a PanelizationResult along three
orthogonal axes: wall length, wall area, and cost fraction.

All dollar amounts are excluded — only ratios and percentages are
reported. Dollar-value estimation is the BOM Agent's domain.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from docs.interfaces.feasibility_report import CoverageMetrics
from src.knowledge_graph.query import get_valid_panels
from src.knowledge_graph.schema import PanelType

if TYPE_CHECKING:
    from docs.interfaces.drl_output import PanelizationResult
    from src.knowledge_graph.loader import KnowledgeGraphStore

logger = logging.getLogger(__name__)

# Default wall height when KG panel height is unavailable.
_DEFAULT_WALL_HEIGHT_INCHES: float = 96.0  # 8 ft

# Mapping from classifier WallType values to KG PanelType.
# WallType.EXTERIOR maps to PanelType.ENVELOPE; WallType.CURTAIN has no
# direct KG equivalent and is treated as ENVELOPE for cost estimation.
_WALL_TYPE_TO_PANEL_TYPE: dict[str, PanelType] = {
    "load_bearing": PanelType.LOAD_BEARING,
    "partition": PanelType.PARTITION,
    "shear": PanelType.SHEAR,
    "exterior": PanelType.ENVELOPE,
    "curtain": PanelType.ENVELOPE,
    "fire_rated": PanelType.FIRE_RATED,
    "unknown": PanelType.PARTITION,  # conservative default
}


def _fire_rating_to_hours(rating_value: str) -> float:
    """Convert a FireRating enum value to numeric hours.

    Args:
        rating_value: The ``FireRating.value`` string, e.g. ``"1_hour"``.

    Returns:
        Fire rating in hours, or 0.0 for unrated / unknown.
    """
    mapping: dict[str, float] = {
        "none": 0.0,
        "1_hour": 1.0,
        "2_hour": 2.0,
        "3_hour": 3.0,
        "4_hour": 4.0,
        "unknown": 0.0,
    }
    return mapping.get(rating_value, 0.0)


def _resolve_wall_height(
    wall_panelization: object,
    store: KnowledgeGraphStore | None,
) -> float:
    """Determine wall height in inches for area computation.

    Tries, in order:
    1. Panel ``height_inches`` from the first assigned panel in the KG.
    2. Falls back to ``_DEFAULT_WALL_HEIGHT_INCHES`` (96 in / 8 ft).

    Args:
        wall_panelization: A ``WallPanelization`` instance.
        store: KG store for panel lookup (may be None for length-only calc).

    Returns:
        Wall height in inches.
    """
    if store is not None and hasattr(wall_panelization, "panels") and wall_panelization.panels:
        first_sku = wall_panelization.panels[0].panel_sku
        panel = store.panels.get(first_sku)
        if panel is not None:
            return panel.height_inches
    return _DEFAULT_WALL_HEIGHT_INCHES


def _estimate_cost_per_inch(
    wall_type_value: str,
    fire_rating_hours: float,
    store: KnowledgeGraphStore,
) -> float:
    """Estimate per-inch cost for a wall type from KG panel pricing.

    Finds the cheapest compatible panel and returns its
    ``unit_cost_per_foot / 12`` to get cost per inch.

    Args:
        wall_type_value: The ``WallType.value`` string.
        fire_rating_hours: Required fire rating in hours.
        store: KG store.

    Returns:
        Estimated cost per linear inch, or 0.0 if no panels found.
    """
    panel_type = _WALL_TYPE_TO_PANEL_TYPE.get(wall_type_value)
    if panel_type is None:
        return 0.0

    # Get all panels of this type meeting fire rating — use a large
    # length to avoid filtering by size.
    panels = get_valid_panels(
        store,
        wall_length_inches=48.0,  # reasonable mid-range
        wall_type=panel_type,
        fire_rating_hours=fire_rating_hours if fire_rating_hours > 0 else None,
    )
    if not panels:
        # Widen search: drop fire rating filter.
        panels = get_valid_panels(
            store,
            wall_length_inches=48.0,
            wall_type=panel_type,
        )
    if not panels:
        return 0.0

    # Use cheapest panel's cost per foot, convert to per inch.
    cheapest = min(panels, key=lambda p: p.unit_cost_per_foot)
    return cheapest.unit_cost_per_foot / 12.0


def calculate_coverage(
    result: PanelizationResult,
    store: KnowledgeGraphStore | None = None,
) -> CoverageMetrics:
    """Compute prefab coverage metrics from a panelization result.

    Calculates three coverage percentages:

    - **by_wall_length_pct**: ratio of panelized wall length to total
      wall length.
    - **by_area_pct**: ratio of panelized wall area (length x height)
      to total wall area.
    - **by_cost_pct**: ratio of panelized wall cost fraction to total
      estimated wall cost, using KG unit pricing. Only computed when
      ``store`` is provided.

    Args:
        result: The ``PanelizationResult`` from the DRL Agent.
        store: KG store for height and cost lookups. When ``None``,
            area defaults to 96-inch height and cost is set to 0.

    Returns:
        A ``CoverageMetrics`` instance with all fields populated.
    """
    total_length = 0.0
    panelized_length = 0.0
    total_area = 0.0
    panelized_area = 0.0
    total_cost_units = 0.0
    panelized_cost_units = 0.0

    # Build a quick lookup from edge_id -> classification.
    classifications_by_edge: dict[int, object] = {}
    for cls in result.source_graph.classifications:
        classifications_by_edge[cls.edge_id] = cls

    for wp in result.panel_map.walls:
        wall_len = wp.wall_length_inches
        total_length += wall_len

        # Determine wall height.
        height = _resolve_wall_height(wp, store)

        # Wall area in square inches, convert to sqft at the end.
        wall_area = wall_len * height
        total_area += wall_area

        # Cost estimation — requires KG store.
        if store is not None:
            cls = classifications_by_edge.get(wp.edge_id)
            wall_type_val = cls.wall_type.value if cls is not None else "unknown"
            fire_hrs = _fire_rating_to_hours(cls.fire_rating.value) if cls is not None else 0.0
            cpi = _estimate_cost_per_inch(wall_type_val, fire_hrs, store)
            wall_cost = wall_len * cpi
            total_cost_units += wall_cost
        else:
            wall_cost = 0.0

        if wp.is_panelizable:
            panelized_length += wall_len
            panelized_area += wall_area
            panelized_cost_units += wall_cost

    # Compute percentages.
    length_pct = (panelized_length / total_length * 100.0) if total_length > 0 else 0.0
    area_pct = (panelized_area / total_area * 100.0) if total_area > 0 else 0.0
    cost_pct = (panelized_cost_units / total_cost_units * 100.0) if total_cost_units > 0 else 0.0

    # Convert area from sq-inches to sq-feet (1 sqft = 144 sqin).
    total_area_sqft = total_area / 144.0
    panelized_area_sqft = panelized_area / 144.0

    return CoverageMetrics(
        by_wall_length_pct=round(length_pct, 2),
        by_area_pct=round(area_pct, 2),
        by_cost_pct=round(cost_pct, 2),
        total_wall_length_inches=round(total_length, 2),
        panelized_wall_length_inches=round(panelized_length, 2),
        total_wall_area_sqft=round(total_area_sqft, 2),
        panelized_wall_area_sqft=round(panelized_area_sqft, 2),
    )
