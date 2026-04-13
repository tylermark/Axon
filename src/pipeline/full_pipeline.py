"""Full Axon pipeline: PDF -> Parser -> Tokenizer -> Diffusion -> Constraints
-> Classifier -> DRL -> Feasibility + BOM + IFC.

Implements I-009: wires Layer 1 (extraction) to Layer 2 (prefab intelligence)
into a single end-to-end function, with graceful degradation when any stage
fails.

Reference: CLAUDE.md §What Axon Does, AGENTS.md §Integration Agent.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from docs.interfaces.drl_output import PanelizationResult
from src.bom.export import export_bom
from src.bom.generator import generate_bom
from src.classifier.classifier import classify_wall_graph
from src.feasibility.report import generate_feasibility_report
from src.optimization.solver import OptimizationConfig, optimize_panelization
from src.pipeline.config import AxonConfig
from src.pipeline.layer1 import Layer1Pipeline
from src.transplant.assembler import assemble_walls
from src.transplant.ifc_export import export_ifc
from src.transplant.matcher import match_bim_families
from src.transplant.openings import attach_openings

if TYPE_CHECKING:
    from docs.interfaces.bill_of_materials import BillOfMaterials
    from docs.interfaces.classified_wall_graph import ClassifiedWallGraph
    from docs.interfaces.feasibility_report import FeasibilityReport
    from docs.interfaces.graph_to_serializer import FinalizedGraph
    from src.knowledge_graph.loader import KnowledgeGraphStore

logger = logging.getLogger(__name__)

# PDF user units to inches (72 pt/inch).
_PDF_UNITS_TO_INCHES: float = 1.0 / 72.0


@dataclass
class PipelineResult:
    """Complete output of the Axon pipeline.

    Each field is populated when its stage succeeds.  When a stage fails,
    the corresponding field is ``None`` (or an empty default) and the
    error is recorded in ``stage_errors``.

    Attributes:
        raw_graph: FinalizedGraph from the Layer 1 extraction pipeline.
        classified_graph: ClassifiedWallGraph from the wall classifier.
        panelization: PanelizationResult from the DRL agent.
        feasibility: FeasibilityReport from the feasibility agent.
        bom: BillOfMaterials from the BOM agent.
        ifc_path: Path to the exported IFC (or JSON fallback) model.
        json_fallback_path: Path to the JSON fallback model when IFC
            export uses the JSON backend.
        bom_export_paths: Paths to BOM CSV/Excel/text files.
        processing_time_seconds: Total wall-clock time for the run.
        stage_errors: Mapping of stage name to error message for any
            stage that raised an exception.
        metadata: Arbitrary pipeline metadata (PDF path, page index, etc.).
    """

    raw_graph: FinalizedGraph | None = None
    classified_graph: ClassifiedWallGraph | None = None
    panelization: PanelizationResult | None = None
    feasibility: FeasibilityReport | None = None
    bom: BillOfMaterials | None = None
    ifc_path: Path | None = None
    json_fallback_path: Path | None = None
    bom_export_paths: list[Path] = field(default_factory=list)
    processing_time_seconds: float = 0.0
    stage_errors: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, object] = field(default_factory=dict)


# ══════════════════════════════════════════════════════════════════════════════
# Public API
# ══════════════════════════════════════════════════════════════════════════════


def run_full_pipeline(
    pdf_path: str | Path,
    kg_store: KnowledgeGraphStore,
    output_dir: str | Path,
    config: AxonConfig | None = None,
    page_index: int = 0,
    device: str = "cpu",
    use_raster: bool = True,
) -> PipelineResult:
    """Run the complete Axon pipeline on a single PDF.

    Orchestrates all pipeline stages in order:
        1. Layer 1 extraction (parse -> tokenize -> diffuse -> constrain)
        2. Wall classification
        3. DRL panelization + pod placement (greedy policy)
        4. Feasibility analysis
        5. BOM generation
        6. BOM export (CSV / Excel / text)
        7. IFC / JSON model export

    If any stage raises an exception, the error is logged and stored in
    ``PipelineResult.stage_errors`` and processing continues with whatever
    partial results are available.

    Args:
        pdf_path: File-system path to the source PDF.
        kg_store: Pre-loaded KnowledgeGraphStore.
        output_dir: Directory where all output files are written.
        config: Axon pipeline config.  Uses defaults when ``None``.
        page_index: Zero-based page number to extract (default 0).
        device: Torch device string (``"cpu"`` or ``"cuda"``).
        use_raster: Whether to include raster features in the tokenizer.

    Returns:
        A :class:`PipelineResult` with all available outputs and any
        stage errors recorded in ``stage_errors``.
    """
    pdf_path = Path(pdf_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if config is None:
        config = AxonConfig()

    result = PipelineResult(
        metadata={
            "pdf_path": str(pdf_path),
            "page_index": page_index,
            "device": device,
        }
    )
    t_start = time.perf_counter()

    # ── Stage 1 & 2 & 3 & 4: Layer 1 extraction ──────────────────────────
    logger.info(
        "[pipeline] Stage 1-4: Layer 1 extraction — %s (page %d)", pdf_path.name, page_index
    )
    try:
        layer1 = Layer1Pipeline(config=config, device=device)
        raw_graph: FinalizedGraph = layer1.extract(
            pdf_path=str(pdf_path),
            page_index=page_index,
            use_raster=use_raster,
        )
        result.raw_graph = raw_graph
        logger.info(
            "[pipeline] Layer 1 done: %d nodes, %d edges, %d walls",
            raw_graph.nodes.shape[0],
            raw_graph.edges.shape[0],
            len(raw_graph.wall_segments),
        )
    except Exception as exc:
        logger.exception("[pipeline] Layer 1 FAILED: %s", exc)
        result.stage_errors["layer1"] = str(exc)
        result.processing_time_seconds = time.perf_counter() - t_start
        return result

    # ── Stage 5: Wall classification ──────────────────────────────────────
    logger.info("[pipeline] Stage 5: Wall classification")
    try:
        classified_graph: ClassifiedWallGraph = classify_wall_graph(raw_graph)
        result.classified_graph = classified_graph
        logger.info(
            "[pipeline] Classification done: %s",
            classified_graph.classification_summary,
        )
    except Exception as exc:
        logger.exception("[pipeline] Classifier FAILED: %s", exc)
        result.stage_errors["classifier"] = str(exc)
        result.processing_time_seconds = time.perf_counter() - t_start
        return result

    # ── Stage 6: Panelization + placement (OR solver or DRL fallback) ────
    opt_config = OptimizationConfig.from_pydantic(config.optimization)
    backend = opt_config.solver_backend
    logger.info("[pipeline] Stage 6: Panelization (backend=%s)", backend)
    panelization: PanelizationResult | None = None
    try:
        panelization = optimize_panelization(
            classified_graph, kg_store, opt_config,
        )
        result.panelization = panelization
        logger.info(
            "[pipeline] Panelization done: SPUR=%.4f, coverage=%.1f%%, waste=%.1f%%",
            panelization.spur_score,
            panelization.coverage_percentage,
            panelization.waste_percentage,
        )
    except Exception as exc:
        logger.exception("[pipeline] Panelization FAILED: %s", exc)
        result.stage_errors["panelization"] = str(exc)

    # ── Stage 7: Feasibility report ───────────────────────────────────────
    if panelization is not None:
        logger.info("[pipeline] Stage 7: Feasibility analysis")
        try:
            feasibility = generate_feasibility_report(panelization, kg_store)
            result.feasibility = feasibility
            logger.info(
                "[pipeline] Feasibility done: project_score=%.4f, %d blockers",
                feasibility.project_score,
                len(feasibility.blockers),
            )
        except Exception as exc:
            logger.exception("[pipeline] Feasibility FAILED: %s", exc)
            result.stage_errors["feasibility"] = str(exc)

    # ── Stage 8: BOM generation ────────────────────────────────────────────
    if panelization is not None:
        logger.info("[pipeline] Stage 8: BOM generation")
        try:
            bom = generate_bom(panelization, kg_store)
            result.bom = bom
            logger.info(
                "[pipeline] BOM done: %d line items, $%.2f total",
                len(bom.line_items),
                bom.project_cost.total_project_cost_usd,
            )
        except Exception as exc:
            logger.exception("[pipeline] BOM FAILED: %s", exc)
            result.stage_errors["bom"] = str(exc)

    # ── Stage 9: BOM export ────────────────────────────────────────────────
    if result.bom is not None:
        logger.info("[pipeline] Stage 9: BOM export")
        try:
            bom_dir = output_dir / "bom"
            bom_paths = export_bom(result.bom, bom_dir)
            result.bom_export_paths = bom_paths
            logger.info("[pipeline] BOM export: %d files", len(bom_paths))
        except Exception as exc:
            logger.exception("[pipeline] BOM export FAILED: %s", exc)
            result.stage_errors["bom_export"] = str(exc)

    # ── Stage 10: IFC / JSON export ───────────────────────────────────────
    if panelization is not None:
        logger.info("[pipeline] Stage 10: IFC export")
        try:
            bim_matches = match_bim_families(panelization, kg_store)
            assemblies = assemble_walls(panelization, bim_matches, kg_store)
            opening_attachments = attach_openings(panelization, assemblies, kg_store)

            ifc_path = output_dir / "model.ifc"
            actual_path = export_ifc(assemblies, opening_attachments, panelization, ifc_path)

            if actual_path.suffix.lower() == ".json":
                result.json_fallback_path = actual_path
            else:
                result.ifc_path = actual_path

            logger.info("[pipeline] IFC export: %s", actual_path)
        except Exception as exc:
            logger.exception("[pipeline] IFC export FAILED: %s", exc)
            result.stage_errors["ifc_export"] = str(exc)

    result.processing_time_seconds = round(time.perf_counter() - t_start, 3)
    logger.info(
        "[pipeline] Complete in %.2fs — %d stage error(s)",
        result.processing_time_seconds,
        len(result.stage_errors),
    )
    return result
