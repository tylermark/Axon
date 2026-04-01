# ARCHITECTURE.md — Axon Technical Architecture

## System Overview

Axon is a **seven-stage inference pipeline** with an offline **three-phase training system**. Every stage operates natively on vector/graph representations. Rasterization exists only as a secondary input channel for semantic context.

---

## Stage 1: PDF Vector Parsing

**Agent:** `parse`  
**Module:** `src/parser/`

### Process
1. Open PDF with PyMuPDF, iterate page content streams
2. Parse PostScript operators into typed primitives:
   - `m` (moveto) → path start coordinates
   - `l` (lineto) → line segment endpoints
   - `c/v/y` (curveto) → cubic Bézier control points
   - `h` (closepath) → loop closure edge
3. Track graphics state stack (`q/Q`) to resolve:
   - Cumulative CTM (Current Transformation Matrix) per path
   - Stroke width, dash pattern, color (stroke and fill)
   - Clipping paths (`W/W*`)
4. Construct raw graph G₀ = (V₀, E₀):
   - V₀: all moveto/lineto/curveto endpoints as (x, y) float64
   - E₀: drawn segments connecting consecutive points within a path

### Key Design Decisions
- **Bézier curves are sampled into polylines** at configurable resolution (default: 8 segments per curve) for graph compatibility. Original control points are preserved as edge metadata.
- **Duplicate vertices are merged** within a configurable tolerance (default: 0.5 PDF units) using a KD-tree spatial index.
- **Decorative elements are flagged, not removed.** A `confidence_wall` score is assigned based on stroke width, color, and geometric regularity. The Tokenizer makes the final semantic decision.

### Performance Target
- Parse a 50K-path ARCH E sheet in <2 seconds

---

## Stage 2: Cross-Modal Tokenization

**Agent:** `token`  
**Module:** `src/tokenizer/`

### Process
1. **Vector tokenization:** Convert each path segment into a token embedding:
   - Token = [operator_type, x₁, y₁, x₂, y₂, stroke_width, dash_hash, color_rgb, ctm_flat]
   - Positional encoding: learned 2D positional embedding based on normalized page coordinates
2. **Raster feature extraction:** Render PDF page to image (300 DPI), pass through HRNet/Swin backbone to produce multi-scale feature maps at 1/4, 1/8, 1/16, 1/32 resolution
3. **Cross-attention fusion (TEF):**
   - Vision→Vector: `Attn(Q=vector_tokens, K=visual_features, V=visual_features)`
   - Vector→Vision: `Attn(Q=visual_features, K=vector_tokens, V=vector_tokens)`
   - Both use multi-head attention with `d_model=256`, `n_heads=8`
4. Output: enriched token sequence where each vector token carries local semantic context (adjacent text labels, hatching textures, symbol proximity)

### Key Design Decisions
- **Raster is auxiliary, not primary.** If rasterization fails or isn't available, the pipeline continues with vector-only tokens (degraded but functional).
- **Attention window is spatially bounded.** Each vector token only attends to visual features within a radius proportional to the page diagonal / 20. This prevents global attention blowup on large sheets.

---

## Stage 3: Graph Diffusion Engine

**Agent:** `diffuse`  
**Module:** `src/diffusion/`

### Process
1. **Forward diffusion (training only):**
   - Inject Gaussian noise into node coordinate matrix X over T timesteps
   - Inject categorical noise into adjacency matrix A (edge flip probability)
   - Schedule: cosine schedule (Nichol & Dhariwal)
2. **Reverse denoising (inference):**
   - Start from noise, conditioned on cross-modal context c from Tokenizer
   - At each step t, neural network predicts ε (noise) for coordinates and edge logits
   - HDSE biases attention: nearby nodes attend strongly, distant nodes attend weakly, with hierarchical levels (wall → room → floor → building)
3. **Output:** Predicted structural graph G* with wall junction nodes and wall segment edges

### Architecture
- Backbone: Linear Transformer with HDSE-augmented attention scores
- Layers: 12 transformer blocks, d_model=512, n_heads=8
- Timestep embedding: sinusoidal, concatenated with context
- T=1000 steps (training), DDIM sampling with 50 steps (inference)

### Key Design Decisions
- **Joint continuous-discrete diffusion.** Coordinates use Gaussian noise; adjacency uses absorbing-state discrete diffusion. These are coupled through shared attention.
- **HDSE replaces standard positional encoding** in the graph transformer. It encodes shortest-path distance, random walk similarity, and hierarchical level simultaneously.

---

## Stage 4: Differentiable Constraint Enforcement

**Agent:** `constrain`  
**Module:** `src/constraints/`

### Process
At each reverse diffusion step t:
1. Extract predicted wall junctions and edges from denoiser output
2. Evaluate all axioms against current geometry:
   - **Orthogonal Integrity:** L_ortho = Σ (1 - |cos(θ_e1, θ_e2)|²) for expected-parallel or expected-perpendicular edge pairs
   - **Parallel Pair Constancy:** L_parallel = Σ max(0, |d(e1,e2) - μ_thickness| - IQR/2) for paired wall edges
   - **Junction Closure:** L_junction = λ_junction · ||L · x||² where L is graph Laplacian, penalizing nodes with degree < 2
   - **Spatial Non-Intersection:** L_intersect = Σ ReLU(overlap_area(e_i, e_j)) for non-adjacent edge pairs
3. Compute total constraint loss: L_SAT = w₁·L_ortho + w₂·L_parallel + w₃·L_junction + w₄·L_intersect
4. Backpropagate gradients through differentiable SAT
5. Project geometry: snap near-orthogonal angles to exact 90° when within tolerance

### Key Design Decisions
- **Soft constraints during training, hard snap during inference.** Training uses smooth penalties; inference applies hard projection after final denoising step.
- **Axiom weights are learned** (via meta-learning on validation set), not hand-tuned.

---

## Stage 5: Topological Integrity

**Agent:** `topo`  
**Module:** `src/topology/`

### Process
1. Construct cubical complex from predicted graph (rasterize to binary grid, then build filtration)
2. Compute persistence diagram PD_pred tracking Betti-0 (components) and Betti-1 (holes)
3. Compute ground-truth persistence diagram PD_gt
4. Calculate Wasserstein-1 distance using Sinkhorn-Knopp:
   - W₁(PD_pred, PD_gt) via entropy-regularized optimal transport
5. Topology-Aware Focal Loss:
   - TAFL = α · W₁ + β · |Betti₀_pred - Betti₀_gt| + γ · |Betti₁_pred - Betti₁_gt|

### Key Design Decisions
- **Cubical complex (not simplicial)** for computational efficiency on grid-aligned filtrations
- **Sinkhorn (not Hungarian)** for differentiability — Hungarian matching has no gradient
- **Betti number counts as auxiliary loss** alongside Wasserstein for direct topological feature count supervision

---

## Stage 6: Physics Validation

**Agent:** `physics`  
**Module:** `src/physics/`

### Process
1. Discretize wall graph into FEA mesh:
   - Walls → MITC4 quadrilateral shell elements (2D plane stress)
   - Slender walls → 1D Euler-Bernoulli beam-column elements
2. Apply loads:
   - Dead load: self-weight based on assumed material density and wall thickness
   - Live load: uniform distributed load per code (e.g., 40 psf residential)
3. Solve equilibrium via JAX-SSO:
   - K·u = F (stiffness × displacement = force)
   - Extract: max displacement, max shear stress, max bearing pressure
4. Compute physics loss:
   - L_PDE = MSE(u_predicted, u_reference) + λ · max(0, σ_max - σ_allowable)
5. Backpropagate via adjoint method

### Key Design Decisions
- **PE-PINN with sin activations** (not ReLU) to resolve high-frequency spatial derivatives in PDE solutions
- **Adjoint method** (not direct backprop through solve) for memory efficiency on large meshes
- **Physics loss is weighted lower early in training** (curriculum: visual → geometric → topological → physical)

---

## Stage 7: IFC Serialization

**Agent:** `serial`  
**Module:** `src/serializer/`

### Process
1. Tokenize finalized graph into compressed JSON vocabulary:
   - Structure hierarchy: vertices → wall segments → openings → rooms → floor
2. Map each wall to `IfcWallStandardCase`:
   - Extrusion axis from graph edge direction
   - Cross-section from parallel pair thickness
   - SweptSolid shape representation
3. Attach openings: `IfcRelVoidsElement`, `IfcOpeningElement`
4. Assign room semantics: `IfcSpace` with `IfcRelSpaceBoundary`
5. Export via IfcOpenShell to IFC-SPF format

---

## Composite Loss Function

The full training objective combines all differentiable losses:

```
L_total = L_diffusion                           (data likelihood)
        + λ₁ · L_SAT                            (geometric constraints)
        + λ₂ · TAFL                             (topological integrity)
        + λ₃ · L_PDE                            (physics viability)
        + λ₄ · L_reconstruction                 (pre-training / MPM)
```

Loss weights follow a curriculum schedule:
- Phase 1 (epochs 1-50): L_diffusion dominant, others warm up linearly
- Phase 2 (epochs 51-150): All losses active, λ values learned via meta-learning
- Phase 3 (epochs 151+): L_PDE and TAFL weights increase for fine structural precision

---

## Training Pipeline

### Phase A: Self-Supervised Pre-Training (MPM)
- Data: 100K+ unlabeled PDF floor plans
- Task: Reconstruct 75-85% masked vector tokens
- Loss: Chamfer Distance + coordinate regression
- Duration: ~200 epochs on 4×A100

### Phase B: Supervised Fine-Tuning (SFT)
- Data: CubiCasa5K, Floorplan-HQ-300K, MSD, ResPlan
- Task: Full pipeline with ground-truth graphs
- Loss: L_total (all components)
- Duration: ~150 epochs on 4×A100

### Phase C: Quality Annealing (GRPO)
- Data: Curated high-quality subset
- Task: Group Relative Policy Optimization
- Reward: composite metric (HIoU + GED + Betti + PINN MSE)
- Duration: ~50 epochs on 4×A100

---

## Evaluation Metrics

| Metric | Target | Module Responsible |
|--------|--------|-------------------|
| HIoU (Hierarchical IoU) | >0.92 | QA benchmarks |
| mAP@50 | >0.90 | QA benchmarks |
| Graph Edit Distance | <5.0 | QA benchmarks |
| Betti Number Error | <0.1 | QA benchmarks |
| PINN Stress MSE | <0.01 | QA benchmarks |
| Inference time (single page) | <10s on A100 | Integration perf tests |
| IFC import success rate | 100% (Revit + ArchiCAD) | Serializer validation |
