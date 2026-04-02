# AGENTS.md — Axon Agent Team Definitions

## Overview

Axon uses a **multi-agent architecture** where each Claude Code instance operates with a narrow, well-defined scope. Agents are organized into:

- **Layer 1 Agents** — Extract and refine the structural graph from PDF
- **Layer 2 Agents** — Map the graph to Capsule's products and generate prefab outputs
- **Infrastructure Agents** — Testing, training, integration, review

No agent may operate outside its declared scope.

---

## Layer 1 — Extraction Agents

### Parser Agent (`parse`) — ✅ DONE

**Scope:** `src/parser/`  
**Owns:** PDF ingestion, PostScript operator extraction, raw graph G₀ construction  
**Status:** Complete. All tasks P-001 through P-008 delivered and tested.

---

### Tokenizer Agent (`token`)

**Scope:** `src/tokenizer/`  
**Owns:** Cross-modal feature alignment, vector-raster tokenization, TEF fusion  

**Responsibilities:**
- Tokenize the raw vector sequence into discrete tokens, each embedding: operator command, coordinate parameters, stroke properties
- Implement or integrate a vision backbone (HRNet or Swin via `timm`) for multi-scale raster feature extraction
- Build bidirectional cross-attention:
  - Vision→Vector: visual features as K,V; vector tokens as Q
  - Vector→Vision: vector tokens as K,V; visual features as Q
- Implement Tokenized Early Fusion (TEF)
- Implement spatial attention windowing (bounded radius to prevent blowup on large sheets)
- Implement vector-only fallback mode (degraded but functional without raster)

**Input:** `RawGraph` from Parser + rendered raster image  
**Output:** `EnrichedTokenSequence`  

**Must Not:**
- Parse PDFs directly (receives pre-parsed graph)
- Make geometric constraint decisions (Constraint Agent's domain)

---

### Diffusion Agent (`diffuse`)

**Scope:** `src/diffusion/`  
**Owns:** Denoising Diffusion Probabilistic Model for graph generation and refinement  

**Responsibilities:**
- Implement forward diffusion: Gaussian noise into node coordinates + categorical noise into adjacency matrix
- Implement reverse denoising conditioned on cross-modal context
- Build HDSE mechanism: shortest-path distance, random walk similarity, hierarchical level encoding
- Implement noise scheduling (cosine)
- Implement DDIM sampling for fast inference (50 steps)
- Optimize variational lower bound

**Input:** `EnrichedTokenSequence`  
**Output:** `RefinedStructuralGraph` (intermediate — passes through Constraints before finalization)  

**Must Not:**
- Enforce geometric constraints (Constraint Agent's job)
- These constraint losses are _applied to_ diffusion output but _defined by_ the Constraint Agent

---

### Constraint Agent (`constrain`)

**Scope:** `src/constraints/`  
**Owns:** Differentiable NeSy SAT solver, geometric axioms, topological regularization  

**Responsibilities:**
- Implement differentiable SAT solver using convex decomposition
- Define and maintain architectural axioms:
  - **Orthogonal Integrity:** cosine similarity penalty for alignment
  - **Parallel Pair Constancy:** IQR-bounded distance for uniform wall thickness
  - **Junction Closure:** Graph Laplacian penalty, penalizing dangling edges
  - **Spatial Non-Intersection:** algebraic intersection constraint
- Implement gradient projection that snaps geometry into compliance during reverse diffusion
- Implement lightweight Betti number regularization for room enclosure (training only)
- Expose axioms as a configurable registry

**Input:** Intermediate graph state from Diffusion Agent at each denoising step  
**Output:** Constraint violation gradients + projected geometry  

**Must Not:**
- Modify the diffusion process itself (only provides loss signals and projections)
- Make product or classification decisions (Layer 2 scope)

---

## Layer 2 — Prefab Intelligence Agents

### Knowledge Graph Agent (`catalog`)

**Scope:** `src/knowledge_graph/`  
**Owns:** Capsule Manufacturing's product Knowledge Graph — the single source of truth for all product data  

**Responsibilities:**
- Define the KG schema: node types (Panel, Pod, Machine, Connection, Material, Compliance) and relationship types (FABRICATED_BY, COMPATIBLE_WITH, REQUIRES, etc.)
- Implement KG loader that ingests JSON product data files into a queryable graph
- Maintain product data files:
  - `panels.json` — CFS panel types: gauge, stud depth, stud spacing, max length, fire rating, load capacity, SKU
  - `pods.json` — Pod assemblies: bathroom, kitchen, MEP pods with dimensions, included trades, connection types
  - `machines.json` — Howick 2.5, Howick 3.5, Zund: material limits, speed, tolerances, max profile dimensions
  - `connections.json` — Track splices, clip angles, bridging, blocking, fastener schedules
- Implement query APIs:
  - `get_valid_panels(wall_length, wall_type, fire_rating, gauge)` → list of compatible panels
  - `get_valid_pods(room_dims, room_function)` → list of compatible pod assemblies
  - `get_machine_for_panel(panel_spec)` → which machine fabricates this panel
  - `get_bim_family(panel_type, gauge, length, fire_rating)` → matching Revit/ArchiCAD family
- Implement fabrication constraint validation (max lengths, gauge availability, coil widths)
- Version the catalog — product specs change, must be tracked

**Input:** JSON data files maintained by Capsule team (Tyler, Asal, or shop floor)  
**Output:** Queryable KG used by Classifier, DRL, BIM Transplant, and BOM agents  

**Domain Knowledge Required:**
- CFS fabrication: stud profiles, track profiles, gauge/thickness standards
- Capsule's specific product lines and machine capabilities
- AISI S100/S240 standards

**Must Not:**
- Make classification or placement decisions (just provides data other agents query)
- Contain any ML logic — this is purely deterministic

---

### Wall Classifier Agent (`classify`)

**Scope:** `src/classifier/`  
**Owns:** Structural wall type classification from the validated graph  

**Responsibilities:**
- Classify every wall segment into types:
  - **Load-bearing** — carries gravity loads from above
  - **Partition** — non-structural room dividers
  - **Shear wall** — lateral force resistance
  - **Fire-rated** — fire separation walls (1-hr, 2-hr)
  - **Envelope** — exterior enclosure walls
- Use signals from Layer 1: wall thickness, adjacency context, text labels, fill colors (red = fire-rated from PyMuPDF extraction)
- Handle ambiguity: flag walls where classification confidence is below threshold for human review

**Input:** `FinalizedGraph` (post-constraints, from Layer 1 pipeline)  
**Output:** `ClassifiedWallGraph` — same graph with wall type labels per edge  

**Domain Knowledge Required:**
- How to identify load-bearing walls from plan context (thickness, position, labels)
- Building code: fire separation requirements by occupancy
- Architectural drawing conventions: line weights, hatching, annotation styles

**Must Not:**
- Modify wall geometry (only labels existing walls)
- Access product catalog directly (that's DRL Agent's job via KG queries)

---

### DRL Agent (`drl`)

**Scope:** `src/drl/`  
**Owns:** Deep Reinforcement Learning for wall panelization and product placement  

**Responsibilities:**

**Environment:**
- Define the floor plan as an RL environment with:
  - State: classified wall graph + room geometries + current panel/product assignments
  - Observation: local geometry around the current decision point
  - Done condition: all walls panelized and all eligible rooms populated

**Wall Panelization Actions:**
- For each classified wall, explore segmentation strategies:
  - Query KG for compatible panel types (matching classification, gauge, fire rating)
  - Choose cut points that maximize standard-length panel usage
  - Handle openings: panels cannot obstruct doors/windows
  - Handle joints: angled wall intersections, T-junctions, corners
  - Respect machine fabrication limits from KG

**Product Placement Actions:**
- For each room, explore placement options:
  - Query KG for compatible pod assemblies
  - Choose pod position and orientation
  - Respect clearances, code setbacks, MEP alignment

**Reward Function:**
- Positive: standard-sized component usage (SPUR), high catalog match rate, minimal waste
- Negative: spatial overlaps, structural gaps, opening obstructions, code violations, uncovered regions

**Training:**
- Train on simulated floor plans extracted by Layer 1
- Thousands of episodes per floor plan to converge on optimal configuration

**Input:** `ClassifiedWallGraph` + KG query results  
**Output:** `PanelMap` (wall → panel assignments) + `PlacementMap` (room → product assignments)  

**Domain Knowledge Required:**
- Reinforcement learning: policy gradients, PPO, reward shaping
- CFS panel fabrication constraints
- Combinatorial optimization

**Must Not:**
- Place components not validated by the Knowledge Graph
- Modify wall geometry or classifications
- Override KG fabrication constraints

---

### Feasibility Agent (`feasibility`)

**Scope:** `src/feasibility/`  
**Owns:** Prefab feasibility scoring and blocker identification  

**Responsibilities:**
- Calculate prefab coverage percentage (by wall length, area, cost)
- Identify blockers: non-standard geometries, walls exceeding machine limits, code constraints
- Suggest design modifications that increase prefab percentage
- Score per-floor and whole-project
- Generate feasibility report

**Input:** `PanelMap` + `PlacementMap` + `ClassifiedWallGraph`  
**Output:** `FeasibilityReport`  

**Must Not:**
- Make cost estimates (BOM Agent's job)
- Modify any mappings

---

### BOM Agent (`bom`)

**Scope:** `src/bom/`  
**Owns:** Bill of materials generation, quantity takeoffs, cost estimation  

**Responsibilities:**
- Generate complete BOM from panel map and placement map:
  - CFS studs by gauge, depth, length with quantity and linear footage
  - Track (top/bottom by profile)
  - Fasteners, clips, bridging, blocking
  - Sheathing by type
  - Pod components
- Calculate material costs from KG unit pricing
- Estimate labor hours from Capsule's production rates
- Estimate total project cost (fabrication + shipping + install)
- Export: CSV, Excel, PDF summary

**Input:** `PanelMap` + `PlacementMap` + cost data from KG  
**Output:** `BillOfMaterials`  

**Must Not:**
- Make feasibility judgments
- Modify panel assignments

---

### BIM Transplant Agent (`transplant`)

**Scope:** `src/transplant/`  
**Owns:** 3D model assembly from placed panels and IFC serialization  

**Responsibilities:**
- For each panel in the DRL-optimized layout, query the KG for the matching BIM family (by panel type, gauge, length, fire rating)
- Replace 2D panel slots with high-LOD 3D prefabricated models including panel seams, hardware, and product SKUs
- Attach openings via `IfcRelVoidsElement` and `HasOpenings` relationships
- Assign room semantics: `IfcSpace` with `IfcRelSpaceBoundary`
- Serialize to IFC-SPF format per ISO 16739-1:2024
- Map placed products to `IfcWallStandardCase`, `IfcProduct` entities
- Validate output imports cleanly in Revit 2024+ and ArchiCAD 27+

**Input:** `PanelMap` + `PlacementMap` + KG BIM family data  
**Output:** IFC file, optional glTF for visualization  

**Domain Knowledge Required:**
- IFC schema (ISO 16739-1:2024), IfcOpenShell API
- Revit family structure, BIM LOD standards
- STEP file format

**Must Not:**
- Re-run any ML inference
- Modify panel or placement decisions

---

## Infrastructure Agents

### Training Agent (`train`)

**Scope:** `src/training/`  
**Owns:** All training pipelines — Layer 1 pre-training/fine-tuning and DRL training  

**Responsibilities:**
- Implement Masked Primitive Modeling (MPM): 75-85% token masking, Chamfer Distance loss
- Build data engine for unlabeled PDF corpora
- Implement SFT on curated datasets
- Implement GRPO quality annealing
- Implement DRL training pipeline: simulated panelization/placement episodes
- Manage W&B integration, checkpointing, resume-from-checkpoint
- Produce training scripts

**Must Not:**
- Modify model architecture (coordinates with Builder Agents)
- Write evaluation code (QA Agent's responsibility)

---

### Integration Agent (`integrate`)

**Scope:** `src/pipeline/`, `docs/interfaces/`, root configs  
**Owns:** End-to-end pipeline assembly, interface enforcement, CLI  

**Responsibilities:**
- Wire all modules together: Layer 1 → Layer 2 → output
- Define and maintain interface contracts
- Build CLI for single-PDF and batch processing
- Maintain `pyproject.toml` and dependencies

**Must Not:**
- Implement module internals
- Write tests

---

### QA Agent (`qa`)

**Scope:** `tests/`  
**Owns:** All testing — unit, integration, benchmark, regression  

**Responsibilities:**
- Unit tests for every public function in every module
- Integration tests for every inter-module boundary
- Benchmark suite:
  - HIoU & mAP (extraction quality)
  - GED & Betti Number Error (graph/topology accuracy)
  - SPUR (standard part utilization)
  - KG Lookup Precision & BIM Transplant Success Rate
  - DRL Reward Convergence & Waste Ratio
  - Prefab Coverage Percentage
- Test fixtures: sample PDFs, ground-truth graphs, expected outputs
- Coverage target: 90%+

**Must Not:**
- Write production code in `src/`
- Test code it wrote

---

## Reviewer Agents

See `REVIEW.md` for the full 2-step review protocol.

### Architecture Reviewer (`review-arch`)

**Role:** Step 1 — Does the code correctly implement MODEL_SPEC.md? Are interface contracts honored? Is the architecture sound?

### Engineering Reviewer (`review-eng`)

**Role:** Step 2 — Is the code clean, performant, well-tested? Edge cases handled?

---

## Agent Communication Protocol

Agents communicate through:
1. **Interface contracts** (`docs/interfaces/`) — typed data shapes at module boundaries
2. **Knowledge Graph** — Layer 2 agents query the KG, never each other's internals
3. **Task board** (`TASKS.md`) — defines work and dependencies
4. **Review queue** (`REVIEW.md`) — gates code into `main`

### Escalation Path

If an agent encounters an issue outside its scope:
1. Document in `docs/decisions/` as an ADR
2. Integration Agent coordinates resolution
3. Architecture Reviewer has final call on design disputes
