"""Axon CLI — command-line interface for floor plan extraction and reporting.

Exposes the full Axon pipeline via a Click CLI.

Usage::

    # Layer 1 only (extraction)
    axon extract floor_plan.pdf
    axon extract floor_plan.pdf -o output.json -f summary --no-raster

    # Full pipeline: extract -> classify -> DRL -> feasibility + BOM + IFC
    axon report floor_plan.pdf --output-dir results/ --kg-data src/knowledge_graph/data/

    # Batch: process a directory of PDFs
    axon batch /path/to/pdfs/ --output-dir results/

Reference: CLAUDE.md §What Axon Does, ARCHITECTURE.md §Layer 1 + Layer 2.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import click
import numpy as np
from rich.console import Console
from rich.table import Table

from src.pipeline.config import AxonConfig
from src.pipeline.layer1 import Layer1Pipeline

if TYPE_CHECKING:
    from docs.interfaces.graph_to_serializer import FinalizedGraph, Opening, Room, WallSegment

console = Console()
logger = logging.getLogger("axon")


# ---------------------------------------------------------------------------
# Serialization helper
# ---------------------------------------------------------------------------


def _wall_segment_to_dict(ws: WallSegment) -> dict[str, Any]:
    """Convert a WallSegment dataclass to a JSON-serializable dict.

    Args:
        ws: A WallSegment instance.

    Returns:
        Dictionary with all WallSegment fields, numpy arrays converted
        to plain lists.
    """
    return {
        "edge_id": ws.edge_id,
        "start_node": ws.start_node,
        "end_node": ws.end_node,
        "start_coord": ws.start_coord.tolist()
        if isinstance(ws.start_coord, np.ndarray)
        else list(ws.start_coord),
        "end_coord": ws.end_coord.tolist()
        if isinstance(ws.end_coord, np.ndarray)
        else list(ws.end_coord),
        "thickness": ws.thickness,
        "height": ws.height,
        "wall_type": ws.wall_type.value,
        "angle": ws.angle,
        "length": ws.length,
        "confidence": ws.confidence,
    }


def _opening_to_dict(opening: Opening) -> dict[str, Any]:
    """Convert an Opening dataclass to a JSON-serializable dict.

    Args:
        opening: An Opening instance.

    Returns:
        Dictionary with all Opening fields.
    """
    return {
        "opening_type": opening.opening_type.value,
        "wall_edge_id": opening.wall_edge_id,
        "position_along_wall": opening.position_along_wall,
        "width": opening.width,
        "height": opening.height,
        "sill_height": opening.sill_height,
        "confidence": opening.confidence,
    }


def _room_to_dict(room: Room) -> dict[str, Any]:
    """Convert a Room dataclass to a JSON-serializable dict.

    Args:
        room: A Room instance.

    Returns:
        Dictionary with all Room fields.
    """
    return {
        "room_id": room.room_id,
        "boundary_edges": room.boundary_edges,
        "boundary_nodes": room.boundary_nodes,
        "area": room.area,
        "label": room.label,
        "is_exterior": room.is_exterior,
    }


def _finalized_graph_to_dict(graph: FinalizedGraph) -> dict[str, Any]:
    """Convert a FinalizedGraph to a JSON-serializable dictionary.

    Args:
        graph: A FinalizedGraph instance from the Layer 1 pipeline.

    Returns:
        Dictionary suitable for ``json.dumps``, with numpy arrays
        converted to nested Python lists.
    """
    return {
        "nodes": graph.nodes.tolist(),
        "edges": graph.edges.tolist(),
        "wall_segments": [_wall_segment_to_dict(ws) for ws in graph.wall_segments],
        "openings": [_opening_to_dict(o) for o in graph.openings],
        "rooms": [_room_to_dict(r) for r in graph.rooms],
        "page_width": graph.page_width,
        "page_height": graph.page_height,
        "page_index": graph.page_index,
        "source_path": graph.source_path,
        "assumed_wall_height": graph.assumed_wall_height,
        "structural_viability": graph.structural_viability,
        "betti_0": graph.betti_0,
        "betti_1": graph.betti_1,
        "coordinate_system": graph.coordinate_system,
        "scale_factor": graph.scale_factor,
        "metadata": graph.metadata,
    }


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------


def _print_summary(graph: FinalizedGraph, elapsed: float) -> None:
    """Print a rich console summary of the extraction result.

    Args:
        graph: The extracted FinalizedGraph.
        elapsed: Wall-clock extraction time in seconds.
    """
    console.rule("[bold blue]Axon Extraction Summary[/bold blue]")

    # Overview table
    table = Table(title="Graph Statistics", show_header=True, header_style="bold cyan")
    table.add_column("Metric", style="dim")
    table.add_column("Value", justify="right")

    n_nodes = graph.nodes.shape[0] if graph.nodes.size > 0 else 0
    n_edges = graph.edges.shape[0] if graph.edges.size > 0 else 0
    n_walls = len(graph.wall_segments)
    n_openings = len(graph.openings)
    n_rooms = len(graph.rooms)

    table.add_row("Nodes", str(n_nodes))
    table.add_row("Edges", str(n_edges))
    table.add_row("Wall segments", str(n_walls))
    table.add_row("Openings", str(n_openings))
    table.add_row("Rooms", str(n_rooms))
    table.add_row("Betti-0 (components)", str(graph.betti_0))
    table.add_row("Betti-1 (loops)", str(graph.betti_1))
    table.add_row("Structural viability", graph.structural_viability)
    table.add_row("Page size", f"{graph.page_width:.1f} x {graph.page_height:.1f} pts")
    table.add_row("Extraction time", f"{elapsed:.2f}s")

    console.print(table)

    # Source info
    console.print(f"\n[dim]Source:[/dim] {graph.source_path}  (page {graph.page_index})")
    console.print(f"[dim]Wall height:[/dim] {graph.assumed_wall_height:.0f} mm")
    console.print(f"[dim]Coordinate system:[/dim] {graph.coordinate_system}")


# ---------------------------------------------------------------------------
# CLI definition
# ---------------------------------------------------------------------------


@click.group()
def main() -> None:
    """Axon -- Floor plan extraction and prefab intelligence."""


@main.command()
@click.argument("pdf_path", type=click.Path(exists=True))
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    default=None,
    help="Output file path. Default: <input_stem>_extracted.json",
)
@click.option(
    "--format",
    "-f",
    "output_format",
    type=click.Choice(["json", "summary"], case_sensitive=False),
    default="json",
    help="Output format: json (file) or summary (console).",
)
@click.option(
    "--page",
    "-p",
    type=int,
    default=0,
    show_default=True,
    help="Zero-based page index to extract.",
)
@click.option(
    "--no-raster",
    is_flag=True,
    default=False,
    help="Skip raster rendering; use vector-only fallback.",
)
@click.option(
    "--device",
    type=click.Choice(["cpu", "cuda"], case_sensitive=False),
    default="cpu",
    show_default=True,
    help="Compute device.",
)
@click.option(
    "--config",
    "config_path",
    type=click.Path(exists=True),
    default=None,
    help="Optional JSON config file to override AxonConfig defaults.",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    default=False,
    help="Enable verbose output.",
)
def extract(
    pdf_path: str,
    output: str | None,
    output_format: str,
    page: int,
    no_raster: bool,
    device: str,
    config_path: str | None,
    verbose: bool,
) -> None:
    """Extract structural graph from a PDF floor plan.

    Runs the full Layer 1 pipeline (parse, tokenize, diffuse, constrain)
    on a single page and outputs either a JSON file or a console summary.

    Example::

        axon extract floor_plan.pdf
        axon extract floor_plan.pdf -f summary -p 2 --no-raster
        axon extract floor_plan.pdf -o result.json --device cuda
    """
    # --- Logging setup ---
    log_level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # --- Load config ---
    if config_path is not None:
        config_text = Path(config_path).read_text(encoding="utf-8")
        config = AxonConfig.model_validate_json(config_text)
        if verbose:
            console.print(f"[dim]Loaded config from {config_path}[/dim]")
    else:
        config = AxonConfig()

    # --- Build pipeline ---
    if verbose:
        console.print(f"[dim]Initializing Layer 1 pipeline on {device}...[/dim]")

    t_init_start = time.perf_counter()
    pipeline = Layer1Pipeline(config=config, device=device)
    t_init_end = time.perf_counter()

    if verbose:
        console.print(f"[dim]Pipeline initialized in {t_init_end - t_init_start:.2f}s[/dim]")

    # --- Extract ---
    use_raster = not no_raster
    if verbose:
        raster_label = "vector+raster" if use_raster else "vector-only"
        console.print(f"[dim]Extracting page {page} ({raster_label})...[/dim]")

    t_extract_start = time.perf_counter()
    try:
        result = pipeline.extract(
            pdf_path=pdf_path,
            page_index=page,
            use_raster=use_raster,
        )
    except Exception as exc:
        console.print(f"[bold red]Extraction failed:[/bold red] {exc}")
        raise SystemExit(1) from exc
    t_extract_end = time.perf_counter()

    elapsed = t_extract_end - t_extract_start

    # --- Output ---
    if output_format == "summary":
        _print_summary(result, elapsed)
    else:
        # JSON output
        output_path = output
        if output_path is None:
            stem = Path(pdf_path).stem
            output_path = f"{stem}_extracted.json"

        data = _finalized_graph_to_dict(result)
        Path(output_path).write_text(
            json.dumps(data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        console.print(f"[green]Wrote {output_path}[/green]")

    # --- Timing ---
    total_time = t_extract_end - t_init_start
    console.print(
        f"\n[dim]Timing: init={t_init_end - t_init_start:.2f}s  "
        f"extract={elapsed:.2f}s  total={total_time:.2f}s[/dim]"
    )


# ---------------------------------------------------------------------------
# report command — full pipeline (Layer 1 + Layer 2)
# ---------------------------------------------------------------------------


@main.command()
@click.argument("pdf_path", type=click.Path(exists=True))
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(),
    default="axon_output",
    show_default=True,
    help="Directory where all output files are written.",
)
@click.option(
    "--kg-data",
    type=click.Path(exists=True),
    default=None,
    help=(
        "Path to knowledge graph data directory containing panels.json, "
        "pods.json, machines.json, connections.json. "
        "Defaults to src/knowledge_graph/data/."
    ),
)
@click.option(
    "--page",
    "-p",
    type=int,
    default=0,
    show_default=True,
    help="Zero-based page index to extract.",
)
@click.option(
    "--no-raster",
    is_flag=True,
    default=False,
    help="Skip raster rendering; use vector-only fallback for Layer 1.",
)
@click.option(
    "--device",
    type=click.Choice(["cpu", "cuda"], case_sensitive=False),
    default="cpu",
    show_default=True,
    help="Compute device for Layer 1 ML stages.",
)
@click.option(
    "--config",
    "config_path",
    type=click.Path(exists=True),
    default=None,
    help="Optional JSON config file to override AxonConfig defaults.",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    default=False,
    help="Enable verbose logging.",
)
def report(
    pdf_path: str,
    output_dir: str,
    kg_data: str | None,
    page: int,
    no_raster: bool,
    device: str,
    config_path: str | None,
    verbose: bool,
) -> None:
    """Run the full Axon pipeline and produce a prefab report.

    Extracts the floor plan from a PDF (Layer 1), classifies walls,
    runs DRL panelization, generates a feasibility report and BOM,
    and exports an IFC model (or JSON fallback).

    Example::

        axon report floor_plan.pdf --output-dir results/
        axon report floor_plan.pdf --kg-data src/knowledge_graph/data/ -v
    """
    import traceback

    from src.knowledge_graph.loader import load_knowledge_graph
    from src.pipeline.full_pipeline import run_full_pipeline
    from src.pipeline.output import print_summary, write_pipeline_outputs

    # --- Logging ---
    log_level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # --- Config ---
    if config_path is not None:
        config_text = Path(config_path).read_text(encoding="utf-8")
        config = AxonConfig.model_validate_json(config_text)
        if verbose:
            console.print(f"[dim]Loaded config from {config_path}[/dim]")
    else:
        config = AxonConfig()

    # --- Load KG ---
    console.print("[dim]Loading Knowledge Graph...[/dim]")
    try:
        kg_store = load_knowledge_graph(kg_data)
        console.print("[dim]KG loaded.[/dim]")
    except Exception as exc:
        console.print(f"[bold red]KG load failed:[/bold red] {exc}")
        if verbose:
            traceback.print_exc()
        raise SystemExit(1) from exc

    # --- Run full pipeline ---
    output_path = Path(output_dir)
    console.print(
        f"[dim]Running full pipeline on {pdf_path} (page {page}, device={device})...[/dim]"
    )
    pipeline_result = run_full_pipeline(
        pdf_path=pdf_path,
        kg_store=kg_store,
        output_dir=output_path,
        config=config,
        page_index=page,
        device=device,
        use_raster=not no_raster,
    )

    # --- Write outputs ---
    try:
        written = write_pipeline_outputs(pipeline_result, output_path)
        for key, path in written.items():
            if "bom_export" in key or key in ("feasibility_report", "summary", "pipeline_result"):
                console.print(f"[green]Wrote {path}[/green]")
    except Exception as exc:
        console.print(f"[yellow]Warning: output writing partial: {exc}[/yellow]")

    # --- Print summary ---
    print_summary(pipeline_result)

    # --- Exit status ---
    if pipeline_result.stage_errors:
        console.print(
            f"\n[yellow]{len(pipeline_result.stage_errors)} stage(s) had errors — "
            "check logs for details.[/yellow]"
        )
        raise SystemExit(2)


# ---------------------------------------------------------------------------
# batch command — process a directory of PDFs
# ---------------------------------------------------------------------------


@main.command()
@click.argument("pdf_dir", type=click.Path(exists=True, file_okay=False))
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(),
    default="axon_batch_output",
    show_default=True,
    help="Root directory for all per-PDF output subdirectories.",
)
@click.option(
    "--kg-data",
    type=click.Path(exists=True),
    default=None,
    help=("Path to knowledge graph data directory. Defaults to src/knowledge_graph/data/."),
)
@click.option(
    "--page",
    "-p",
    type=int,
    default=0,
    show_default=True,
    help="Zero-based page index to extract from every PDF.",
)
@click.option(
    "--no-raster",
    is_flag=True,
    default=False,
    help="Skip raster rendering for all PDFs.",
)
@click.option(
    "--device",
    type=click.Choice(["cpu", "cuda"], case_sensitive=False),
    default="cpu",
    show_default=True,
    help="Compute device.",
)
@click.option(
    "--config",
    "config_path",
    type=click.Path(exists=True),
    default=None,
    help="Optional JSON config file.",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    default=False,
    help="Enable verbose logging.",
)
def batch(
    pdf_dir: str,
    output_dir: str,
    kg_data: str | None,
    page: int,
    no_raster: bool,
    device: str,
    config_path: str | None,
    verbose: bool,
) -> None:
    """Process all PDF files in a directory through the full Axon pipeline.

    Each PDF gets its own subdirectory inside OUTPUT_DIR named after the
    PDF stem.  Failures on individual files are logged and skipped so the
    batch continues.

    Example::

        axon batch /path/to/pdfs/ --output-dir results/
        axon batch plans/ --kg-data src/knowledge_graph/data/ --device cuda
    """
    import traceback

    from src.knowledge_graph.loader import load_knowledge_graph
    from src.pipeline.full_pipeline import run_full_pipeline
    from src.pipeline.output import write_pipeline_outputs

    # --- Logging ---
    log_level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # --- Config ---
    if config_path is not None:
        config_text = Path(config_path).read_text(encoding="utf-8")
        config = AxonConfig.model_validate_json(config_text)
    else:
        config = AxonConfig()

    # --- Load KG (once, shared across all PDFs) ---
    console.print("[dim]Loading Knowledge Graph...[/dim]")
    try:
        kg_store = load_knowledge_graph(kg_data)
        console.print("[dim]KG loaded.[/dim]")
    except Exception as exc:
        console.print(f"[bold red]KG load failed:[/bold red] {exc}")
        if verbose:
            traceback.print_exc()
        raise SystemExit(1) from exc

    # --- Collect PDFs ---
    pdf_files = sorted(Path(pdf_dir).glob("*.pdf"))
    if not pdf_files:
        console.print(f"[yellow]No PDF files found in {pdf_dir}[/yellow]")
        raise SystemExit(0)

    console.print(f"[dim]Found {len(pdf_files)} PDF(s) in {pdf_dir}[/dim]")

    root_output = Path(output_dir)
    root_output.mkdir(parents=True, exist_ok=True)

    results_summary: list[dict] = []
    failed: list[str] = []

    for i, pdf_path in enumerate(pdf_files, 1):
        pdf_output = root_output / pdf_path.stem
        console.print(
            f"\n[bold cyan][{i}/{len(pdf_files)}][/bold cyan] "
            f"Processing [bold]{pdf_path.name}[/bold]..."
        )

        try:
            pipeline_result = run_full_pipeline(
                pdf_path=pdf_path,
                kg_store=kg_store,
                output_dir=pdf_output,
                config=config,
                page_index=page,
                device=device,
                use_raster=not no_raster,
            )
            write_pipeline_outputs(pipeline_result, pdf_output)

            score = (
                pipeline_result.feasibility.project_score if pipeline_result.feasibility else None
            )
            total_cost = (
                pipeline_result.bom.project_cost.total_project_cost_usd
                if pipeline_result.bom
                else None
            )
            results_summary.append(
                {
                    "file": pdf_path.name,
                    "score": f"{score:.4f}" if score is not None else "n/a",
                    "cost": f"${total_cost:,.2f}" if total_cost is not None else "n/a",
                    "errors": len(pipeline_result.stage_errors),
                    "time": f"{pipeline_result.processing_time_seconds:.1f}s",
                }
            )

            status = (
                "[green]OK[/green]"
                if not pipeline_result.stage_errors
                else "[yellow]PARTIAL[/yellow]"
            )
            console.print(
                f"  {status}  score={score or 'n/a'}  "
                f"cost={total_cost or 'n/a'}  "
                f"time={pipeline_result.processing_time_seconds:.1f}s"
            )

        except Exception as exc:
            console.print(f"  [bold red]FAILED:[/bold red] {exc}")
            if verbose:
                traceback.print_exc()
            failed.append(pdf_path.name)
            results_summary.append(
                {
                    "file": pdf_path.name,
                    "score": "ERROR",
                    "cost": "ERROR",
                    "errors": -1,
                    "time": "n/a",
                }
            )

    # --- Batch summary ---
    console.print("\n")
    console.rule("[bold blue]Batch Summary[/bold blue]")

    from rich.table import Table

    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("File")
    table.add_column("Score", justify="right")
    table.add_column("Total Cost", justify="right")
    table.add_column("Errors", justify="right")
    table.add_column("Time", justify="right")

    for row in results_summary:
        style = "red" if row["errors"] == -1 else ("yellow" if row["errors"] > 0 else "")
        table.add_row(
            row["file"],
            row["score"],
            row["cost"],
            str(row["errors"]) if row["errors"] >= 0 else "FAIL",
            row["time"],
            style=style,
        )

    console.print(table)
    console.print(
        f"\n[dim]Processed {len(pdf_files)} PDF(s). "
        f"Failed: {len(failed)}. "
        f"Output: {root_output}[/dim]"
    )

    if failed:
        raise SystemExit(2)
