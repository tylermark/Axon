"""Axon BOM Agent — Bill of Materials generation.

Produces itemized quantity takeoffs, cost estimates, and labor projections
from the DRL agent's PanelizationResult and Knowledge Graph pricing data.

Public API:
    generate_bom — orchestrate full BOM from PanelizationResult + KG
    export_bom   — render BOM to CSV, Excel, and/or PDF
"""

from __future__ import annotations

from src.bom.cfs_takeoff import compute_cfs_takeoff
from src.bom.costing import (
    compute_labor_estimates,
    compute_material_summary,
    compute_project_cost,
)
from src.bom.export import export_bom
from src.bom.generator import generate_bom
from src.bom.pod_takeoff import compute_pod_takeoff

__all__ = [
    "compute_cfs_takeoff",
    "compute_labor_estimates",
    "compute_material_summary",
    "compute_pod_takeoff",
    "compute_project_cost",
    "export_bom",
    "generate_bom",
]
