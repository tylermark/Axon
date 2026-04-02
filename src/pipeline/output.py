"""Pipeline output formatting and file writing.

Implements I-011: serializes a PipelineResult to disk and renders a
human-readable summary to stdout.

Output files written to the output directory:
    feasibility_report.json — serialized FeasibilityReport
    bom.csv / bom.xlsx / bom.txt — Bill of Materials exports (from BOM agent)
    model.ifc (or model.json) — IFC export from BIM Transplant
    summary.txt — human-readable project summary
    pipeline_result.json — machine-readable full result

Reference: AGENTS.md §Integration Agent, CLAUDE.md §What Axon Does.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from src.pipeline.full_pipeline import PipelineResult

logger = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════════════════
# JSON serialization helpers
# ══════════════════════════════════════════════════════════════════════════════


class _PipelineEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy arrays, dataclasses, enums, and Paths."""

    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, Path):
            return str(obj)
        if hasattr(obj, "value"):  # Enum
            return obj.value
        if is_dataclass(obj):
            return asdict(obj)
        return super().default(obj)


def _safe_asdict(obj: Any) -> Any:
    """Recursively convert dataclass / numpy / enum to JSON-safe primitives."""
    if obj is None:
        return None
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, Path):
        return str(obj)
    if hasattr(obj, "value") and not callable(obj.value):  # Enum
        return obj.value
    if is_dataclass(obj):
        return {k: _safe_asdict(v) for k, v in asdict(obj).items()}
    if isinstance(obj, dict):
        return {k: _safe_asdict(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_safe_asdict(v) for v in obj]
    if isinstance(obj, tuple):
        return [_safe_asdict(v) for v in obj]
    return obj


# ══════════════════════════════════════════════════════════════════════════════
# Feasibility report serialization
# ══════════════════════════════════════════════════════════════════════════════


def _serialize_feasibility(result: PipelineResult) -> dict[str, Any]:
    """Build a JSON-safe dict from the FeasibilityReport.

    Args:
        result: Complete pipeline result.

    Returns:
        Dictionary suitable for json.dumps.
    """
    feas = result.feasibility
    if feas is None:
        return {"error": "feasibility stage did not produce output"}

    return {
        "project_score": feas.project_score,
        "coverage": {
            "by_wall_length_pct": feas.coverage.by_wall_length_pct,
            "by_area_pct": feas.coverage.by_area_pct,
            "by_cost_pct": feas.coverage.by_cost_pct,
            "total_wall_length_inches": feas.coverage.total_wall_length_inches,
            "panelized_wall_length_inches": feas.coverage.panelized_wall_length_inches,
        },
        "summary": _safe_asdict(feas.summary),
        "floor_scores": [_safe_asdict(fs) for fs in feas.floor_scores],
        "blockers": [_safe_asdict(b) for b in feas.blockers],
        "suggestions": [_safe_asdict(s) for s in feas.suggestions],
        "wall_feasibility": [_safe_asdict(w) for w in feas.wall_feasibility],
        "room_feasibility": [_safe_asdict(r) for r in feas.room_feasibility],
        "metadata": feas.metadata,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Pipeline result serialization
# ══════════════════════════════════════════════════════════════════════════════


def _serialize_pipeline_result(result: PipelineResult) -> dict[str, Any]:
    """Build a JSON-safe summary dict from the full PipelineResult.

    Includes only lightweight scalar / string data — not the full graph
    node arrays, which are very large.

    Args:
        result: The complete PipelineResult.

    Returns:
        JSON-safe dictionary for pipeline_result.json.
    """
    raw = result.raw_graph
    classified = result.classified_graph
    panelization = result.panelization
    bom = result.bom
    feas = result.feasibility

    return {
        "metadata": _safe_asdict(result.metadata),
        "processing_time_seconds": result.processing_time_seconds,
        "stage_errors": result.stage_errors,
        "outputs": {
            "ifc_path": str(result.ifc_path) if result.ifc_path else None,
            "json_fallback_path": str(result.json_fallback_path)
            if result.json_fallback_path
            else None,
            "bom_export_paths": [str(p) for p in result.bom_export_paths],
        },
        "layer1": {
            "nodes": raw.nodes.shape[0] if raw is not None else None,
            "edges": raw.edges.shape[0] if raw is not None else None,
            "walls": len(raw.wall_segments) if raw is not None else None,
            "rooms": len(raw.rooms) if raw is not None else None,
            "betti_0": raw.betti_0 if raw is not None else None,
            "betti_1": raw.betti_1 if raw is not None else None,
        },
        "classification": classified.classification_summary if classified is not None else None,
        "drl": {
            "spur_score": panelization.spur_score if panelization else None,
            "coverage_percentage": panelization.coverage_percentage if panelization else None,
            "waste_percentage": panelization.waste_percentage if panelization else None,
            "pod_placement_rate": panelization.pod_placement_rate if panelization else None,
            "total_panel_count": panelization.total_panel_count if panelization else None,
            "total_splice_count": panelization.total_splice_count if panelization else None,
        },
        "feasibility": {
            "project_score": feas.project_score if feas else None,
            "hard_blockers": feas.summary.hard_blocker_count if feas else None,
            "soft_blockers": feas.summary.soft_blocker_count if feas else None,
            "suggestions": feas.summary.suggestion_count if feas else None,
        },
        "bom": {
            "line_items": len(bom.line_items) if bom else None,
            "material_total_usd": bom.material_summary.material_total_usd if bom else None,
            "labor_total_usd": bom.total_labor_cost_usd if bom else None,
            "project_total_usd": bom.project_cost.total_project_cost_usd if bom else None,
        },
    }


# ══════════════════════════════════════════════════════════════════════════════
# Summary text
# ══════════════════════════════════════════════════════════════════════════════


def _build_summary_lines(result: PipelineResult) -> list[str]:
    """Render a human-readable summary of the pipeline result.

    Args:
        result: Complete pipeline result.

    Returns:
        List of text lines (no trailing newlines).
    """
    lines: list[str] = []
    sep = "=" * 72
    thin = "-" * 72

    meta = result.metadata
    pdf_path = meta.get("pdf_path", "unknown")
    page_idx = meta.get("page_index", 0)

    lines.append(sep)
    lines.append("AXON — FULL PIPELINE REPORT")
    lines.append(sep)
    lines.append(f"  Source:  {pdf_path}  (page {page_idx})")
    lines.append(f"  Runtime: {result.processing_time_seconds:.2f}s")

    if result.stage_errors:
        lines.append(f"  Errors:  {len(result.stage_errors)} stage(s) failed")
        for stage, msg in result.stage_errors.items():
            lines.append(f"    [{stage}] {msg[:120]}")
    else:
        lines.append("  Status:  All stages completed successfully")

    # ── Layer 1 ─────────────────────────────────────────────────────────────
    lines.append("")
    lines.append(thin)
    lines.append("LAYER 1 — EXTRACTION")
    lines.append(thin)
    raw = result.raw_graph
    if raw is not None:
        lines.append(f"  Nodes:            {raw.nodes.shape[0]}")
        lines.append(f"  Edges:            {raw.edges.shape[0]}")
        lines.append(f"  Wall segments:    {len(raw.wall_segments)}")
        lines.append(f"  Openings:         {len(raw.openings)}")
        lines.append(f"  Rooms:            {len(raw.rooms)}")
        lines.append(f"  Betti-0/1:        {raw.betti_0} / {raw.betti_1}")
    else:
        lines.append("  [not available]")

    # ── Classification ───────────────────────────────────────────────────────
    lines.append("")
    lines.append(thin)
    lines.append("WALL CLASSIFICATION")
    lines.append(thin)
    cl = result.classified_graph
    if cl is not None:
        for wtype, count in sorted(cl.classification_summary.items()):
            lines.append(f"  {wtype:<22} {count}")
        lines.append(f"  Flagged for review: {len(cl.walls_flagged_for_review)}")
    else:
        lines.append("  [not available]")

    # ── DRL ─────────────────────────────────────────────────────────────────
    lines.append("")
    lines.append(thin)
    lines.append("DRL PANELIZATION + PLACEMENT")
    lines.append(thin)
    pan = result.panelization
    if pan is not None:
        lines.append(f"  SPUR score:       {pan.spur_score:.4f}")
        lines.append(f"  Wall coverage:    {pan.coverage_percentage:.1f}%")
        lines.append(f"  Material waste:   {pan.waste_percentage:.1f}%")
        lines.append(f"  Pod placement:    {pan.pod_placement_rate:.1f}%")
        lines.append(f"  Total panels:     {pan.total_panel_count}")
        lines.append(f"  Total splices:    {pan.total_splice_count}")
        lines.append(f"  Unique panel SKUs:{len(pan.panel_map.unique_panel_skus)}")
        lines.append(f"  Unique pod SKUs:  {len(pan.placement_map.unique_pod_skus)}")
    else:
        lines.append("  [not available]")

    # ── Feasibility ──────────────────────────────────────────────────────────
    lines.append("")
    lines.append(thin)
    lines.append("FEASIBILITY ASSESSMENT")
    lines.append(thin)
    feas = result.feasibility
    if feas is not None:
        lines.append(f"  Project score:    {feas.project_score:.4f} / 1.0000")
        lines.append(f"  Coverage (length):{feas.coverage.by_wall_length_pct:.1f}%")
        lines.append(f"  Coverage (area):  {feas.coverage.by_area_pct:.1f}%")
        lines.append(
            f"  Walls panelized:  {feas.summary.panelized_wall_count}"
            f" / {feas.summary.total_wall_count}"
        )
        lines.append(
            f"  Pods placed:      {feas.summary.placed_room_count}"
            f" / {feas.summary.eligible_room_count} eligible"
        )
        lines.append(f"  Hard blockers:    {feas.summary.hard_blocker_count}")
        lines.append(f"  Soft blockers:    {feas.summary.soft_blocker_count}")
        lines.append(
            f"  Suggestions:      {feas.summary.suggestion_count}"
            f" (max gain {feas.summary.max_coverage_gain_pct:.1f}%)"
        )

        top_suggestions = sorted(
            feas.suggestions,
            key=lambda s: s.estimated_coverage_gain_pct,
            reverse=True,
        )[:3]
        if top_suggestions:
            lines.append("")
            lines.append("  Top design suggestions:")
            for i, sug in enumerate(top_suggestions, 1):
                gain = f"+{sug.estimated_coverage_gain_pct:.1f}%"
                lines.append(
                    f"    {i}. [{sug.effort_level or '?'} effort] {gain} — {sug.description[:80]}"
                )
    else:
        lines.append("  [not available]")

    # ── BOM ─────────────────────────────────────────────────────────────────
    lines.append("")
    lines.append(thin)
    lines.append("BILL OF MATERIALS")
    lines.append(thin)
    bom = result.bom
    if bom is not None:
        pc = bom.project_cost
        ms = bom.material_summary
        lines.append(f"  Line items:         {len(bom.line_items)}")
        lines.append(f"  Material total:     ${ms.material_total_usd:>12,.2f}")
        lines.append(f"  Labor total:        ${bom.total_labor_cost_usd:>12,.2f}")
        lines.append(f"  Fabrication sub:    ${pc.fabrication_subtotal_usd:>12,.2f}")
        lines.append(f"  Pod cost:           ${pc.pod_cost_usd:>12,.2f}")
        lines.append(f"  Shipping:           ${pc.shipping_usd:>12,.2f}")
        lines.append(f"  Installation sub:   ${pc.installation_subtotal_usd:>12,.2f}")
        lines.append(f"  TOTAL PROJECT COST: ${pc.total_project_cost_usd:>12,.2f}")
        contingency_amt = pc.total_project_cost_usd * pc.contingency_pct / 100.0
        total_with = pc.total_project_cost_usd + contingency_amt
        lines.append(f"  Total w/ {pc.contingency_pct:.0f}% ctg.:  ${total_with:>12,.2f}")
        lines.append(f"  Total labor hours:  {bom.total_labor_hours:.1f} hrs")
    else:
        lines.append("  [not available]")

    # ── Outputs ──────────────────────────────────────────────────────────────
    lines.append("")
    lines.append(thin)
    lines.append("OUTPUT FILES")
    lines.append(thin)
    if result.ifc_path:
        lines.append(f"  IFC model:     {result.ifc_path}")
    if result.json_fallback_path:
        lines.append(f"  JSON model:    {result.json_fallback_path}")
    for p in result.bom_export_paths:
        lines.append(f"  BOM export:    {p}")

    lines.append("")
    lines.append(sep)
    return lines


# ══════════════════════════════════════════════════════════════════════════════
# Public API
# ══════════════════════════════════════════════════════════════════════════════


def write_pipeline_outputs(result: PipelineResult, output_dir: Path) -> dict[str, Path]:
    """Write all pipeline outputs to the output directory.

    Creates:
    - ``feasibility_report.json`` — serialized FeasibilityReport
    - ``summary.txt`` — human-readable summary
    - ``pipeline_result.json`` — full serialized result for programmatic access

    BOM CSV/Excel/text and IFC/JSON model files are written by earlier
    pipeline stages (recorded in ``result.bom_export_paths``,
    ``result.ifc_path``, ``result.json_fallback_path``).  This function
    records all paths in the returned manifest.

    Args:
        result: Complete pipeline result from :func:`run_full_pipeline`.
        output_dir: Directory to write new output files into. Created if
            it does not exist.

    Returns:
        Dictionary mapping output type key to file path for every file
        written or recorded by this function.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    written: dict[str, Path] = {}

    # ── feasibility_report.json ──────────────────────────────────────────────
    feas_path = output_dir / "feasibility_report.json"
    try:
        feas_data = _serialize_feasibility(result)
        feas_path.write_text(
            json.dumps(feas_data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        written["feasibility_report"] = feas_path
        logger.info("Wrote %s", feas_path)
    except Exception as exc:
        logger.warning("Could not write feasibility_report.json: %s", exc)

    # ── summary.txt ─────────────────────────────────────────────────────────
    summary_path = output_dir / "summary.txt"
    try:
        lines = _build_summary_lines(result)
        summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        written["summary"] = summary_path
        logger.info("Wrote %s", summary_path)
    except Exception as exc:
        logger.warning("Could not write summary.txt: %s", exc)

    # ── pipeline_result.json ─────────────────────────────────────────────────
    result_path = output_dir / "pipeline_result.json"
    try:
        result_data = _serialize_pipeline_result(result)
        result_path.write_text(
            json.dumps(result_data, indent=2, ensure_ascii=False, cls=_PipelineEncoder),
            encoding="utf-8",
        )
        written["pipeline_result"] = result_path
        logger.info("Wrote %s", result_path)
    except Exception as exc:
        logger.warning("Could not write pipeline_result.json: %s", exc)

    # ── Record BOM and IFC paths from prior stages ────────────────────────────
    for i, p in enumerate(result.bom_export_paths):
        written[f"bom_export_{i}"] = p

    if result.ifc_path:
        written["ifc_model"] = result.ifc_path

    if result.json_fallback_path:
        written["json_model"] = result.json_fallback_path

    return written


def print_summary(result: PipelineResult) -> None:
    """Print a human-readable summary of the pipeline result to stdout.

    Covers: project feasibility score, wall/area coverage, panel count,
    splice count, pod placement rate, blocker counts, top 3 suggestions,
    total estimated cost, and IFC file path.

    Args:
        result: Complete pipeline result from :func:`run_full_pipeline`.
    """
    lines = _build_summary_lines(result)
    print("\n".join(lines))
