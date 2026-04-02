"""Axon Feasibility Agent — prefab scoring, blockers, and suggestions.

Analyzes a PanelizationResult to determine how much of a floor plan
can be prefabricated by Capsule Manufacturing, identifies what blocks
the rest, and recommends design changes to increase coverage.
"""

from __future__ import annotations

from src.feasibility.blockers import identify_blockers
from src.feasibility.calculator import calculate_coverage
from src.feasibility.report import generate_feasibility_report
from src.feasibility.suggestions import generate_suggestions

__all__ = [
    "calculate_coverage",
    "generate_feasibility_report",
    "generate_suggestions",
    "identify_blockers",
]
