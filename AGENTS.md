# AGENTS.md — Axon Agent Team Definitions

## Overview

This project uses a **multi-agent architecture** where each Claude Code instance operates with a narrow, well-defined scope. Agents are organized into three tiers:

- **Builder Agents** — Write code within their module boundary
- **Infrastructure Agents** — Handle cross-cutting concerns (testing, integration, training)
- **Reviewer Agents** — Evaluate work product through a 2-step review gate

No agent may operate outside its declared scope. If a task requires cross-module work, the Integration Agent coordinates and each Builder Agent handles its own module.

---

## Builder Agents

### Parser Agent (`parse`)

**Scope:** `src/parser/`  
**Owns:** PDF ingestion, PostScript operator extraction, raw graph G₀ construction  

**Responsibilities:**
- Implement PyMuPDF-based content stream parser that extracts `m`, `l`, `c`, `v`, `y`, `h` operators with full coordinate resolution
- Build and maintain the operator registry mapping PDF operators → mathematical representations
- Construct the initial directed spatial graph G₀ from extracted primitives, preserving:
  - Continuous (x, y) coordinates for all endpoints
  - Stroke width, dash pattern, and fill color per path
  - Cumulative Current Transformation Matrix (CTM) per graphics state
- Implement filtering heuristics to flag (not remove) decorative elements: hatching, dimension lines, text annotations, furniture symbols
- Handle edge cases: nested `q/Q` state stacks, clipping paths (`W/W*`), and concatenated transformation matrices (`cm`)

**Input:** Raw PDF file (born-digital, vector content)  
**Output:** `RawGraph` dataclass (see `docs/interfaces/parser_to_tokenizer.py`)  

**Domain Knowledge Required:**
- PDF specification (ISO 32000-2:2020), content stream syntax
- PyMuPDF path extraction API (`page.get_drawings()`, `page.get_text("rawdict")`)
- Graphics state machine: CTM composition, stroke/fill distinction

**Must Not:**
- Rasterize any content for its own processing
- Make semantic decisions about what is/isn't a wall (that's Tokenizer's job)
- Import from any module outside `src/parser/`

---

### Tokenizer Agent (`token`)

**Scope:** `src/tokenizer/`  
**Owns:** Cross-modal feature alignment, vector-raster tokenization, TEF fusion  

**Responsibilities:**
- Tokenize the raw vector sequence into discrete tokens t_i, each embedding: operator command, coordinate parameters, stroke properties
- Implement or integrate a vision backbone (HRNet or Swin Transformer via `timm`) for multi-scale raster feature extraction
- Build the bidirectional cross-attention module:
  - Vision-to-vector: visual features as K,V; vector tokens as Q
  - Vector-to-vision: vector tokens as K,V; visual features as Q
- Implement Tokenized Early Fusion (TEF) for deep cross-modal entanglement
- Output semantically enriched vector embeddings ready for the diffusion engine

**Input:** `RawGraph` from Parser + rendered raster image of the PDF page  
**Output:** `EnrichedTokenSequence` (see `docs/interfaces/tokenizer_to_diffusion.py`)  

**Domain Knowledge Required:**
- Transformer attention mechanisms, cross-attention formulation
- Multi-scale feature maps (FPN-style)
- Positional encoding for irregular geometric sequences

**Must Not:**
- Parse PDFs directly (receives pre-parsed graph from Parser)
- Make geometric constraint decisions (that's Constraint Agent's domain)

---

### Diffusion Agent (`diffuse`)

**Scope:** `src/diffusion/`  
**Owns:** Denoising Diffusion Probabilistic Model for graph generation and refinement  

**Responsibilities:**
- Implement forward diffusion: progressive Gaussian noise injection into both continuous node coordinates X and discrete adjacency matrix A
- Implement reverse denoising: iterative refinement conditioned on cross-modal context embeddings c
- Build the Hierarchical Distance Structural Encoding (HDSE) mechanism to bias attention scores toward multi-level graph structure
- Implement noise scheduling (linear, cosine, or learned)
- Handle the joint distribution of geometry (continuous) and topology (discrete) during denoising
- Optimize the variational lower bound of data likelihood

**Input:** `EnrichedTokenSequence` from Tokenizer  
**Output:** `RefinedStructuralGraph` (intermediate — passes through Constraint, Topology, Physics before finalization)  

**Domain Knowledge Required:**
- DDPM theory (Ho et al.), score matching, SDE formulations
- Graph neural networks, message passing, over-smoothing mitigation
- Linear transformers with structural attention biases

**Must Not:**
- Enforce geometric constraints (Constraint Agent's job)
- Compute topological losses (Topology Agent's job)
- Evaluate structural viability (Physics Agent's job)
- These losses are _applied to_ the diffusion output but are _defined by_ other agents

---

### Constraint Agent (`constrain`)

**Scope:** `src/constraints/`  
**Owns:** Differentiable NeSy SAT solver, architectural axiom enforcement  

**Responsibilities:**
- Implement the differentiable Boolean SAT solver using convex decomposition (Carathéodory's theorem approach)
- Define and maintain the architectural axiom set:
  - **Orthogonal Integrity:** cosine similarity penalty for Manhattan/non-Manhattan alignment
  - **Parallel Pair Constancy:** IQR-bounded distance constraint for uniform wall thickness
  - **Junction Closure:** Graph Laplacian penalty for closed loops, penalizing dangling edges
  - **Spatial Non-Intersection:** algebraic intersection constraint preventing overlapping walls
- Implement gradient projection that "snaps" predicted geometry into compliance during reverse diffusion
- Expose axioms as a configurable registry (new constraints can be added without code changes)

**Input:** Intermediate graph state from Diffusion Agent at each denoising step t  
**Output:** Constraint violation gradients + projected (snapped) geometry  

**Domain Knowledge Required:**
- SAT/SMT solvers, convex relaxation techniques
- Neuro-symbolic AI (Kautz Type 5 taxonomy)
- Computational geometry: line intersection, parallelism detection, angle computation

**Must Not:**
- Modify the diffusion process itself (only provides loss signals and projections)
- Make topological assessments (Topology Agent handles global connectivity)

---

### Topology Agent (`topo`)

**Scope:** `src/topology/`  
**Owns:** Persistent homology, Betti number computation, Topology-Aware Focal Loss  

**Responsibilities:**
- Construct filtered cubical complexes from predicted floor plan graphs
- Compute persistence diagrams tracking birth/death of topological features (Betti-0: connected components, Betti-1: enclosed holes/loops)
- Implement Sinkhorn-Knopp algorithm for optimal transport between predicted and ground-truth persistence diagrams
- Compute Wasserstein distance as the topological loss signal
- Implement the Topology-Aware Focal Loss (TAFL) that integrates into the composite training loss
- Ensure differentiability of the entire persistence → Wasserstein → loss chain

**Input:** Predicted graph from Diffusion Agent + ground-truth graph  
**Output:** Topological loss value (scalar, differentiable)  

**Domain Knowledge Required:**
- Algebraic topology: simplicial/cubical complexes, filtrations, homology groups
- Persistent homology computation (giotto-tda or gudhi)
- Optimal transport theory, Sinkhorn algorithm

**Must Not:**
- Modify graph geometry (only computes and reports loss)
- Duplicate constraint checking already handled by Constraint Agent

---

### Physics Agent (`physics`)

**Scope:** `src/physics/`  
**Owns:** PINN, differentiable FEA, structural load-path validation  

**Responsibilities:**
- Discretize 2D wall segments into analysis mesh: MITC4 quadrilateral shell elements + 1D Euler-Bernoulli beam-column elements
- Implement the PE-PINN architecture with periodic sine activations (sinusoidal layers to address spectral bias)
- Compute optimal structural load paths: dead load and live load transfer to foundation
- Integrate JAX-SSO for differentiable FEA solving
- Compute physics loss via adjoint method: gradients of displacement, shear stress, and bearing capacity w.r.t. wall node coordinates
- Backpropagate PDE loss to force geometry adjustments achieving mechanical equilibrium

**Input:** `RefinedStructuralGraph` after constraint and topology processing  
**Output:** Physics loss value (scalar, differentiable) + structural viability report  

**Domain Knowledge Required:**
- Finite Element Analysis: shell elements, beam-column theory
- Physics-Informed Neural Networks (Raissi et al.)
- JAX automatic differentiation, vmap, adjoint methods
- Structural mechanics: load paths, shear, moment, deflection limits

**Must Not:**
- Alter graph topology (only evaluates and signals violations via gradients)
- Override constraint agent's geometric snapping

---

### Serializer Agent (`serial`)

**Scope:** `src/serializer/`  
**Owns:** IFC-compliant output generation, JSON vocabulary, export pipeline  

**Responsibilities:**
- Define the custom compressed JSON vocabulary for floor plan serialization
- Implement "Structure-First, Semantics-Second" hierarchy: geometry skeleton → openings → room semantics
- Map wall vectors to `IfcWallStandardCase` entities per ISO 16739-1:2024
- Generate swept solid shape representations using extrusion axis from graph vertices + wall thickness from SAT solver's parallel pair constraint
- Attach openings via `IfcRelVoidsElement` / `HasOpenings` relationships
- Export to: structured JSON, IFC-SPF (STEP Physical File), and optionally glTF for visualization
- Validate output imports cleanly in Revit 2024+ and ArchiCAD 27+

**Input:** Finalized, validated structural graph + semantic labels  
**Output:** IFC file, JSON file, optional glTF  

**Domain Knowledge Required:**
- IFC schema (ISO 16739-1:2024), IfcOpenShell API
- STEP file format
- BIM interoperability requirements

**Must Not:**
- Modify or re-validate geometry (receives final, validated graph)
- Re-run any ML inference

---

## Infrastructure Agents

### Training Agent (`train`)

**Scope:** `src/training/`  
**Owns:** Pre-training pipeline, fine-tuning, data engine  

**Responsibilities:**
- Implement Masked Primitive Modeling (MPM): random masking of 75-85% of vector tokens + corresponding raster patches
- Build reconstruction objective using Chamfer Distance + parameter regression loss (not pixel MSE)
- Implement the data engine for ingesting unlabeled PDF corpora
- Implement Supervised Fine-Tuning (SFT) on curated datasets (e.g., Floorplan-HQ-300K, CubiCasa5K)
- Implement Group Relative Policy Optimization (GRPO) for quality annealing
- Manage training configs, checkpointing, W&B integration
- Produce training scripts (`scripts/pretrain.sh`, `scripts/finetune.sh`)

**Input:** Unlabeled PDF corpus (pre-training), labeled datasets (SFT), reward model (GRPO)  
**Output:** Trained model checkpoints  

**Must Not:**
- Modify model architecture (coordinates with Builder Agents for architecture changes)
- Write evaluation code (QA Agent's responsibility)

---

### Integration Agent (`integrate`)

**Scope:** `src/pipeline/`, `docs/interfaces/`, root-level configs  
**Owns:** End-to-end pipeline assembly, module interface enforcement, CLI  

**Responsibilities:**
- Wire all modules together into the inference pipeline (`runner.py`)
- Define and maintain global configuration schema (`config.py`)
- Build CLI interface for single-PDF and batch processing (`cli.py`)
- Enforce that all inter-module communication goes through typed interface contracts
- Coordinate cross-module dependency resolution
- Maintain `pyproject.toml` and dependency lockfile

**Input:** All module outputs chained together  
**Output:** Working end-to-end system  

**Must Not:**
- Implement any module internals (delegates to Builder Agents)
- Write tests (QA Agent's job)

---

### QA Agent (`qa`)

**Scope:** `tests/`  
**Owns:** All testing — unit, integration, benchmark, regression  

**Responsibilities:**
- Write unit tests for every public function in every module
- Write integration tests for every inter-module boundary
- Implement benchmark suite for evaluation metrics:
  - Hierarchical IoU (HIoU) & mAP
  - Graph Edit Distance (GED)
  - LayoutGKN / SSIG
  - Betti Number Error
  - PINN Stress/Load Variance (MSE)
- Maintain test fixtures: sample PDFs, ground-truth graphs, expected outputs
- Run regression tests on every PR
- Report test coverage — target: 90%+ on `src/`

**Input:** Code from all agents  
**Output:** Test suite, coverage reports, benchmark results  

**Critical Rule:** QA Agent never tests code it wrote. QA only tests code written by other agents.

**Must Not:**
- Write production code in `src/`
- Approve its own test implementations (reviewed by Engineering Reviewer)

---

## Reviewer Agents

See `REVIEW.md` for the full 2-step review protocol.

### Architecture Reviewer (`review-arch`)

**Role:** Step 1 — Design and correctness review  
**Focus:** Does the code correctly implement the thesis model's mathematical specification? Are interface contracts honored? Is the architecture sound?

### Engineering Reviewer (`review-eng`)

**Role:** Step 2 — Code quality and robustness review  
**Focus:** Is the code clean, performant, and well-tested? Are edge cases handled? Are there security or reliability concerns?

---

## Agent Communication Protocol

Agents do not call each other directly. Communication flows through:

1. **Interface contracts** (`docs/interfaces/`) — define data shapes at module boundaries
2. **Task board** (`TASKS.md`) — defines what needs to be done and dependencies
3. **Review queue** (`REVIEW.md`) — gates code from entering `main`

### Escalation Path

If an agent encounters an issue outside its scope:
1. Document the issue in `docs/decisions/` as an ADR (Architecture Decision Record)
2. Tag the relevant agent(s) in the ADR
3. The Integration Agent coordinates resolution
4. If architectural, the Architecture Reviewer makes the final call
