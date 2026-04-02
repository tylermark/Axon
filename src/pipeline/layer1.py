"""Layer 1 extraction pipeline: PDF -> FinalizedGraph.

Chains the four Layer 1 stages end-to-end for inference:
    1. Parser    — PDF vector extraction -> RawGraph
    2. Tokenizer — cross-modal fusion -> EnrichedTokenSequence
    3. Diffusion — DDPM sampling -> RefinedStructuralGraph
    4. Constraints — NeSy SAT enforcement -> ConstraintGradients
    5. Finalize  — assemble FinalizedGraph for serialization

Reference: ARCHITECTURE.md §Stages 1-4, CLAUDE.md §Layer 1 — Extraction.
"""

from __future__ import annotations

import logging
from math import pi
from typing import TYPE_CHECKING

import numpy as np
import torch
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from docs.interfaces.diffusion_output import RefinedStructuralGraph
from docs.interfaces.graph_to_serializer import (
    FinalizedGraph,
    WallSegment,
    WallType,
)
from src.constraints import ConstraintSolver
from src.diffusion import GraphDiffusionModel
from src.parser import apply_filters, build_raw_graph, extract_paths_from_pdf
from src.pipeline.config import AxonConfig
from src.tokenizer import Tokenizer, collate_graphs, preprocess_image, render_pdf_page

if TYPE_CHECKING:
    from docs.interfaces.constraint_signals import ConstraintGradients
    from docs.interfaces.parser_to_tokenizer import RawGraph
    from docs.interfaces.tokenizer_to_diffusion import EnrichedTokenSequence

logger = logging.getLogger(__name__)

# Default wall thickness when no parallel-pair estimate is available (PDF units).
_DEFAULT_WALL_THICKNESS = 6.0

# Maximum edges (tokens) the tokenizer can handle without OOM.
# Attention is O(N²), so 2048 tokens ≈ 16M attention entries — safe on CPU/GPU.
_MAX_TOKENIZER_EDGES = 2048


def _get_page_dimensions(pdf_path: str, page_index: int) -> tuple[float, float]:
    """Read page width and height from a PDF via PyMuPDF.

    Args:
        pdf_path: Path to the PDF file.
        page_index: Zero-based page index.

    Returns:
        Tuple of (page_width, page_height) in PDF user units.
    """
    import fitz

    doc = fitz.open(pdf_path)
    try:
        page = doc[page_index]
        rect = page.rect
        return float(rect.width), float(rect.height)
    finally:
        doc.close()


def _empty_finalized_graph(
    page_width: float,
    page_height: float,
    page_index: int,
    source_path: str,
    wall_height: float,
) -> FinalizedGraph:
    """Return an empty FinalizedGraph for degenerate inputs (0 nodes/edges)."""
    return FinalizedGraph(
        nodes=np.empty((0, 2), dtype=np.float64),
        edges=np.empty((0, 2), dtype=np.int64),
        wall_segments=[],
        openings=[],
        rooms=[],
        page_width=page_width,
        page_height=page_height,
        page_index=page_index,
        source_path=source_path,
        assumed_wall_height=wall_height,
        structural_viability="unknown",
        betti_0=0,
        betti_1=0,
    )


class Layer1Pipeline:
    """End-to-end Layer 1 extraction: PDF -> FinalizedGraph.

    Chains parser, tokenizer, diffusion, and constraint stages into a
    single inference pipeline.  All torch operations run under
    ``torch.no_grad()`` on the specified device.

    Example::

        pipeline = Layer1Pipeline(device="cuda")
        result = pipeline.extract("floor_plan.pdf", page_index=0)
    """

    def __init__(
        self,
        config: AxonConfig | None = None,
        device: str = "cpu",
    ) -> None:
        """Initialize Layer 1 pipeline models.

        Args:
            config: Top-level Axon config.  Uses defaults when ``None``.
            device: Torch device string (``"cpu"``, ``"cuda"``, ``"cuda:0"``).
        """
        self.config = config if config is not None else AxonConfig()
        self.device = device

        # --- Tokenizer (Stage 2) ---
        self.tokenizer = Tokenizer(
            config=self.config.tokenizer,
            n_tef_layers=2,
        )
        self.tokenizer.to(self.device)
        self.tokenizer.eval()

        # --- Diffusion model (Stage 3) ---
        self.diffusion_model = GraphDiffusionModel(config=self.config.diffusion)
        self.diffusion_model.to(self.device)
        self.diffusion_model.eval()

        # --- Constraint solver (Stage 4) ---
        self.constraint_solver = ConstraintSolver(config=self.config.constraints)
        self.constraint_solver.to(self.device)
        self.constraint_solver.eval()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(
        self,
        pdf_path: str,
        page_index: int = 0,
        use_raster: bool = True,
    ) -> FinalizedGraph:
        """Run full Layer 1 extraction on a single PDF page.

        Args:
            pdf_path: File-system path to the source PDF.
            page_index: Zero-based page number to process.
            use_raster: When ``True``, render the page and feed raster
                features to the tokenizer.  Set ``False`` for vector-only
                fallback.

        Returns:
            A :class:`FinalizedGraph` ready for downstream serialization
            (Layer 2 / IFC export).
        """
        raw_graph = self._parse(pdf_path, page_index)

        # Handle degenerate (empty) parser output early.
        if raw_graph.nodes.shape[0] < 2 or raw_graph.edges.shape[0] == 0:
            logger.warning(
                "Parser returned %d nodes / %d edges — returning empty graph",
                raw_graph.nodes.shape[0],
                raw_graph.edges.shape[0],
            )
            return _empty_finalized_graph(
                page_width=raw_graph.page_width,
                page_height=raw_graph.page_height,
                page_index=page_index,
                source_path=pdf_path,
                wall_height=self.config.serializer.default_wall_height_mm,
            )

        # Trim large graphs to avoid tokenizer OOM (attention is O(N²)).
        raw_graph = self._trim_graph(raw_graph)

        with torch.no_grad():
            enriched = self._tokenize(raw_graph, pdf_path, page_index, use_raster)
            refined = self._diffuse(enriched, raw_graph)
            constrained = self._constrain(refined)

        return self._finalize(constrained, refined, raw_graph)

    # ------------------------------------------------------------------
    # Stage implementations
    # ------------------------------------------------------------------

    def _parse(self, pdf_path: str, page_index: int) -> RawGraph:
        """Stage 1 — Parse PDF into a RawGraph.

        Args:
            pdf_path: Path to the PDF file.
            page_index: Page to extract.

        Returns:
            Filtered :class:`RawGraph`.
        """
        pages = extract_paths_from_pdf(
            pdf_path,
            page_indices=[page_index],
            config=self.config.parser,
        )

        paths = pages.get(page_index, [])
        page_width, page_height = _get_page_dimensions(pdf_path, page_index)

        graph = build_raw_graph(
            paths,
            config=self.config.parser,
            page_width=page_width,
            page_height=page_height,
            page_index=page_index,
            source_path=pdf_path,
        )

        graph = apply_filters(graph, config=self.config.parser)
        return graph

    def _trim_graph(self, raw_graph: RawGraph) -> RawGraph:
        """Trim graph to fit tokenizer memory budget.

        Keeps the top-K edges by wall confidence when the graph exceeds
        ``_MAX_TOKENIZER_EDGES``.  Rebuilds node/edge arrays so indices
        remain contiguous.

        Args:
            raw_graph: Filtered raw graph from the parser.

        Returns:
            Trimmed :class:`RawGraph` (or the original if already small enough).
        """
        n_edges = raw_graph.edges.shape[0]
        if n_edges <= _MAX_TOKENIZER_EDGES:
            return raw_graph

        logger.info(
            "Trimming graph from %d to %d edges (top by wall confidence)",
            n_edges,
            _MAX_TOKENIZER_EDGES,
        )

        # Select top-K edges by wall confidence.
        keep_idx = np.argsort(raw_graph.confidence_wall)[::-1][:_MAX_TOKENIZER_EDGES]
        keep_idx = np.sort(keep_idx)  # preserve original order

        kept_edges = raw_graph.edges[keep_idx]

        # Remap node indices to a compact range.
        unique_nodes = np.unique(kept_edges)
        old_to_new = {int(old): new for new, old in enumerate(unique_nodes)}
        new_edges = np.vectorize(old_to_new.get)(kept_edges).astype(np.int64)
        new_nodes = raw_graph.nodes[unique_nodes]

        from docs.interfaces.parser_to_tokenizer import RawGraph as RawGraphCls

        return RawGraphCls(
            nodes=new_nodes,
            edges=new_edges,
            operator_types=[raw_graph.operator_types[i] for i in keep_idx],
            stroke_widths=raw_graph.stroke_widths[keep_idx],
            stroke_colors=raw_graph.stroke_colors[keep_idx],
            fill_colors=raw_graph.fill_colors[keep_idx] if raw_graph.fill_colors is not None else None,
            dash_patterns=[raw_graph.dash_patterns[i] for i in keep_idx],
            path_metadata=raw_graph.path_metadata,
            edge_to_path=raw_graph.edge_to_path[keep_idx],
            bezier_controls=[raw_graph.bezier_controls[i] for i in keep_idx],
            confidence_wall=raw_graph.confidence_wall[keep_idx],
            page_width=raw_graph.page_width,
            page_height=raw_graph.page_height,
            page_index=raw_graph.page_index,
            source_path=raw_graph.source_path,
            vertex_merge_tolerance=raw_graph.vertex_merge_tolerance,
            bezier_sample_resolution=raw_graph.bezier_sample_resolution,
            num_original_paths=raw_graph.num_original_paths,
        )

    def _tokenize(
        self,
        raw_graph: RawGraph,
        pdf_path: str,
        page_index: int,
        use_raster: bool,
    ) -> EnrichedTokenSequence:
        """Stage 2 — Cross-modal tokenization.

        Args:
            raw_graph: Filtered raw graph from the parser.
            pdf_path: Source PDF (needed for raster rendering).
            page_index: Page number.
            use_raster: Whether to include raster features.

        Returns:
            :class:`EnrichedTokenSequence` on ``self.device``.
        """
        features = collate_graphs([raw_graph])
        features = {k: v.to(self.device) for k, v in features.items()}

        images: torch.Tensor | None = None
        if use_raster:
            raster = render_pdf_page(
                pdf_path,
                page_index=page_index,
                dpi=self.config.tokenizer.raster_dpi,
            )
            images = preprocess_image(raster).to(self.device)

        return self.tokenizer(features, images=images)

    def _diffuse(
        self,
        enriched: EnrichedTokenSequence,
        raw_graph: RawGraph,
    ) -> RefinedStructuralGraph:
        """Stage 3 — Graph diffusion sampling.

        Args:
            enriched: Enriched token sequence from the tokenizer.
            raw_graph: Original raw graph (used for node count).

        Returns:
            :class:`RefinedStructuralGraph` from reverse diffusion.
        """
        num_nodes = min(
            raw_graph.nodes.shape[0],
            self.config.diffusion.max_nodes,
        )

        # Edge case: cannot run diffusion with < 2 nodes.
        if num_nodes < 2:
            logger.warning("num_nodes=%d < 2; returning minimal diffusion graph", num_nodes)
            n = max(num_nodes, 2)
            return RefinedStructuralGraph(
                node_positions=torch.zeros(1, n, 2, device=self.device),
                adjacency_logits=torch.zeros(1, n, n, device=self.device),
                node_mask=torch.ones(1, n, dtype=torch.bool, device=self.device),
                edge_index=torch.empty(2, 0, dtype=torch.long, device=self.device),
                edge_logits=torch.empty(0, device=self.device),
                junction_types=[["unclassified"] * n],
                node_features=torch.zeros(1, n, self.config.diffusion.d_model, device=self.device),
                denoising_step=0,
                total_steps=self.config.diffusion.timesteps_train,
            )

        # Context = tokenizer embeddings, shape (B, N_ctx, 256)
        context = enriched.token_embeddings.to(self.device)
        context_mask = enriched.attention_mask.to(self.device)

        return self.diffusion_model.sample(
            num_nodes=num_nodes,
            batch_size=1,
            context=context,
            context_mask=context_mask,
            device=self.device,
        )

    def _constrain(self, refined: RefinedStructuralGraph) -> ConstraintGradients:
        """Stage 4 — NeSy SAT constraint enforcement (inference mode).

        Args:
            refined: Output of the diffusion sampling step.

        Returns:
            :class:`ConstraintGradients` with projected geometry.
        """
        return self.constraint_solver(
            node_positions=refined.node_positions,
            adjacency=refined.adjacency_logits,
            edge_index=refined.edge_index,
            node_mask=refined.node_mask,
            denoising_step=0,
            total_steps=refined.total_steps,
            is_inference=True,
        )

    # ------------------------------------------------------------------
    # Finalization
    # ------------------------------------------------------------------

    def _finalize(
        self,
        constraints: ConstraintGradients,
        refined: RefinedStructuralGraph,
        raw_graph: RawGraph,
    ) -> FinalizedGraph:
        """Convert diffusion + constraint outputs into a FinalizedGraph.

        Steps:
            1. Select projected positions/adjacency when available.
            2. Denormalize from [0,1] to PDF user units.
            3. Extract edges from thresholded adjacency.
            4. Build WallSegment per edge.
            5. Compute Betti numbers.

        Args:
            constraints: Constraint solver output.
            refined: Diffusion model output.
            raw_graph: Original raw graph (page dimensions, source path).

        Returns:
            :class:`FinalizedGraph` ready for serialization.
        """
        page_width = raw_graph.page_width
        page_height = raw_graph.page_height
        wall_height = self.config.serializer.default_wall_height_mm

        # 1. Use projected geometry when the constraint solver provided it.
        if constraints.projected_positions is not None:
            positions = constraints.projected_positions[0].detach().cpu()
        else:
            positions = refined.node_positions[0].detach().cpu()

        if constraints.projected_adjacency is not None:
            adj = constraints.projected_adjacency[0].detach().cpu()
        else:
            adj = torch.sigmoid(refined.adjacency_logits[0].detach().cpu())

        node_mask = refined.node_mask[0].detach().cpu()

        # Mask out padding nodes.
        valid_count = int(node_mask.sum().item())
        positions = positions[:valid_count]
        adj = adj[:valid_count, :valid_count]

        # 2. Denormalize: diffusion operates in [0,1] page space.
        scale = torch.tensor([page_width, page_height], dtype=positions.dtype)
        positions_denorm = positions * scale

        nodes_np = positions_denorm.numpy().astype(np.float64)

        # 3. Extract edges from adjacency (threshold > 0.5).
        adj_binary = (adj > 0.5).float()
        # Ensure symmetric and no self-loops.
        adj_binary = adj_binary * (1.0 - torch.eye(valid_count))
        adj_symmetric = torch.triu(adj_binary, diagonal=1)
        edge_src, edge_dst = torch.where(adj_symmetric > 0.5)
        edges_np = np.stack([edge_src.numpy(), edge_dst.numpy()], axis=1).astype(np.int64)

        if edges_np.shape[0] == 0:
            return _empty_finalized_graph(
                page_width=page_width,
                page_height=page_height,
                page_index=raw_graph.page_index,
                source_path=raw_graph.source_path,
                wall_height=wall_height,
            )

        # 4. Build WallSegment for each edge.
        wall_thickness_map: dict[int, float] = {}
        if constraints.wall_thickness_estimates is not None:
            wt = constraints.wall_thickness_estimates.detach().cpu().numpy()
            for idx, thickness in enumerate(wt):
                wall_thickness_map[idx] = float(thickness)

        wall_segments: list[WallSegment] = []
        for eidx in range(edges_np.shape[0]):
            src, dst = int(edges_np[eidx, 0]), int(edges_np[eidx, 1])
            start = nodes_np[src]
            end = nodes_np[dst]
            delta = end - start
            length = float(np.linalg.norm(delta))
            angle = float(np.arctan2(delta[1], delta[0]) % pi)
            thickness = wall_thickness_map.get(eidx, _DEFAULT_WALL_THICKNESS)

            wall_segments.append(
                WallSegment(
                    edge_id=eidx,
                    start_node=src,
                    end_node=dst,
                    start_coord=start.copy(),
                    end_coord=end.copy(),
                    thickness=thickness,
                    height=wall_height,
                    wall_type=WallType.UNKNOWN,
                    angle=angle,
                    length=length,
                    confidence=1.0,
                )
            )

        # 5. Betti numbers via scipy connected components.
        n_nodes = nodes_np.shape[0]
        n_edges = edges_np.shape[0]

        if n_nodes > 0 and n_edges > 0:
            # Build symmetric sparse adjacency for connected_components.
            rows = np.concatenate([edges_np[:, 0], edges_np[:, 1]])
            cols = np.concatenate([edges_np[:, 1], edges_np[:, 0]])
            data = np.ones(len(rows), dtype=np.float64)
            sparse_adj = csr_matrix((data, (rows, cols)), shape=(n_nodes, n_nodes))
            betti_0 = int(connected_components(sparse_adj, directed=False, return_labels=False))
        else:
            betti_0 = n_nodes  # Each isolated node is its own component.

        # Betti-1 via Euler formula for planar graphs: E - N + betti_0.
        betti_1 = max(0, n_edges - n_nodes + betti_0)

        return FinalizedGraph(
            nodes=nodes_np,
            edges=edges_np,
            wall_segments=wall_segments,
            openings=[],
            rooms=[],
            page_width=page_width,
            page_height=page_height,
            page_index=raw_graph.page_index,
            source_path=raw_graph.source_path,
            assumed_wall_height=wall_height,
            structural_viability="unknown",
            betti_0=betti_0,
            betti_1=betti_1,
        )
