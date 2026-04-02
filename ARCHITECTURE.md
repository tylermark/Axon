# ARCHITECTURE.md — Axon Technical Architecture

## System Overview

Axon is a **two-layer system**: Layer 1 extracts a clean structural graph from any PDF floor plan. Layer 2 maps that graph to Capsule Manufacturing's real products using a Knowledge Graph and Deep Reinforcement Learning, then outputs a manufacture-ready BIM model.

---

## Layer 1: Extraction

### Stage 1: PDF Vector Parsing — ✅ DONE

**Agent:** `parse` | **Module:** `src/parser/`

Parses PDF PostScript operators (`m`, `l`, `c`, `h`, `q/Q`, `W`) via PyMuPDF. Extracts continuous coordinates, stroke properties, and CTM transforms. Constructs raw graph G₀. Bézier curves sampled to polylines. Vertices deduplicated via KD-tree. Decorative elements flagged (not removed). Validated at <2s on ~50K-path ARCH E sheets.

### Stage 2: Cross-Modal Tokenization

**Agent:** `token` | **Module:** `src/tokenizer/`

Dual-processes the PDF: vector tokens from parsed graph + raster features from rendered image via HRNet/Swin backbone. Bidirectional cross-attention (TEF) fuses modalities so vector tokens absorb semantic context (e.g., "Load Bearing" labels, hatching patterns). Spatial attention is windowed to prevent blowup on large sheets. Vector-only fallback if raster unavailable.

### Stage 3: Graph Diffusion Engine

**Agent:** `diffuse` | **Module:** `src/diffusion/`

DDPM adapted for joint continuous-discrete graphs. Forward: Gaussian noise into node coordinates, categorical noise into adjacency. Reverse: iterative denoising conditioned on cross-modal context. HDSE biases attention toward graph hierarchy (wall → room → floor → building). Cosine noise schedule. DDIM sampling at 50 steps for inference.

### Stage 4: Geometric Constraints

**Agent:** `constrain` | **Module:** `src/constraints/`

Differentiable SAT solver enforces four axioms at each denoising step: orthogonal integrity, parallel pair constancy, junction closure, spatial non-intersection. Gradient projection snaps geometry into compliance. Lightweight Betti number regularization ensures room enclosure during training.

**Composite loss:**
$$\mathcal{L}_{total} = \mathcal{L}_{diffusion} + \lambda_{SAT}\,\mathcal{L}_{constraints}$$

Soft constraints during training, hard snap during inference.

---

## Layer 2: Prefab Intelligence

### Stage 5: Knowledge Graph

**Agent:** `catalog` | **Module:** `src/knowledge_graph/`

Structured, deterministic database of Capsule's entire product world. Node types: Panel, Pod, Machine, Connection, Material, Compliance. Relationship types: FABRICATED_BY, COMPATIBLE_WITH, REQUIRES, RATED_FOR.

All Layer 2 agents query the KG — it is the single source of truth. No probabilistic guessing. If the KG says a panel doesn't exist in that gauge/length/fire-rating combination, it doesn't get placed.

Key queries:
- `get_valid_panels(wall_length, type, fire_rating)` → compatible panels with fabrication details
- `get_valid_pods(room_dims, function)` → compatible pod assemblies
- `get_bim_family(panel_spec)` → exact Revit/ArchiCAD family match

### Stage 6: Wall Classification

**Agent:** `classify` | **Module:** `src/classifier/`

Labels every wall edge: load-bearing, partition, shear, fire-rated, envelope. Uses thickness, adjacency, text annotations, fill colors from Layer 1 extraction. Confidence scoring flags ambiguous walls for human review.

### Stage 7: DRL Panelization & Placement

**Agent:** `drl` | **Module:** `src/drl/`

The core optimization engine. A Deep Reinforcement Learning agent operates in a floor plan environment:

**State:** Classified wall graph + room geometries + current assignments  
**Actions:** Panel cut-point selection, panel type assignment, pod placement position/orientation  
**Reward:** SPUR (standard part usage), waste minimization, catalog match rate  
**Penalties:** Overlaps, gaps, opening obstructions, code violations

Two sub-tasks run sequentially:
1. **Panelization** — divide each wall into discrete CFS panels from the KG catalog
2. **Placement** — populate eligible rooms with pod assemblies from the KG catalog

The DRL agent can ONLY place components validated by the KG. This is a hard constraint, not a soft penalty.

Training: thousands of episodes per floor plan on simulated graphs extracted by Layer 1.

### Stage 8: Feasibility & BOM

**Agents:** `feasibility` + `bom` | **Modules:** `src/feasibility/`, `src/bom/`

Feasibility: prefab coverage % (by wall length, area, cost), blocker identification, design modification suggestions ("straighten this wall for +12 panels").

BOM: quantity takeoff (studs, track, fasteners, sheathing, pod components), cost estimation from KG pricing, labor hours from production rates. Export to CSV, Excel, PDF.

### Stage 9: BIM Library Transplant & IFC Export

**Agent:** `transplant` | **Module:** `src/transplant/`

Each 2D panel slot from the DRL output is matched to its exact 3D BIM family via deterministic KG lookup (panel type + gauge + length + fire rating → Revit family). The 2D skeleton is replaced with high-LOD 3D models including seams, hardware, and SKUs. Openings attached via IfcRelVoidsElement. Output serialized to IFC per ISO 16739-1:2024. Must import clean in Revit 2024+ and ArchiCAD 27+.

---

## Training Pipeline

### Datasets

All datasets are local at `datasets/` (relative to repo root).

| Dataset | Count | Format | Training Use |
|---------|-------|--------|-------------|
| **archcad400k** | ~41K (4 shards processed of 400K) | Vector entities (LINE/ARC with coords) + PNG + SVG + point clouds + captions (JSON) | MPM pre-training, SFT — vector-native geometry |
| **FloorPlanCAD** | 15K+ (3 tar.xz archives) | CAD drawings | MPM pre-training — vector-native |
| **ResPlan** | 17,107 | Shapely polygons (walls, rooms, doors, windows) + room graphs (pickle) | DRL episode generation, graph supervision |
| **MLSTRUCT-FP** | 935 PNGs | Raster floor plan images | Cross-modal tokenizer raster branch |

**archcad400k** is the primary training dataset. JSON format has `entities` list with typed geometric primitives (`LINE`, `ARC`) containing start/end coordinates — maps directly to Axon's parser output format. Paired PNGs provide raster supervision for the cross-modal tokenizer. Pre-processed shards at `datasets/archcad400k/processed/shard_XXX.pt`.

**ResPlan** provides structured room graphs with typed polygons (wall, bedroom, kitchen, bathroom, etc.) and `wall_depth` — ideal for generating synthetic DRL training episodes with ground-truth room segmentation.

### Phase A: Self-Supervised Pre-Training (MPM)
- Data: archcad400k (vector entities) + FloorPlanCAD
- Task: reconstruct 75-85% masked vector tokens
- Loss: Chamfer Distance + coordinate regression

### Phase B: Supervised Fine-Tuning (SFT)
- Data: archcad400k (vector + raster pairs), MLSTRUCT-FP (raster)
- Loss: $\mathcal{L}_{total}$ (diffusion + SAT constraints)

### Phase C: Quality Annealing (GRPO)
- Reward: composite metric (HIoU + GED + Betti)

### Phase D: DRL Training
- Environment: ResPlan room graphs + Layer 1 synthetic floor plans
- Algorithm: PPO (Stable-Baselines3 or CleanRL)
- Reward: SPUR + waste ratio + violation count
- Episodes: thousands per floor plan until convergence

---

## Evaluation Metrics

| Metric | Target | What It Measures |
|--------|--------|-----------------|
| HIoU | >0.92 | Wall extraction accuracy |
| GED | <5.0 | Graph topology accuracy |
| Betti Error | <0.1 | Room enclosure integrity |
| SPUR | >0.85 | Standard panel utilization (prefab efficiency) |
| KG Match | >0.95 | Correct product selection from catalog |
| BIM Transplant | 100% | IFC import success in Revit + ArchiCAD |
| DRL Convergence | Stable reward | Optimization quality |
| Waste Ratio | <5% | Material waste from non-standard cuts |
| Prefab Coverage | Report | % of building achievable with Capsule products |
