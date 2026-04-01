# CLAUDE.md — Axon

## Project Identity

**Name:** Axon  
**Tagline:** *From lines to load paths. Vector-native. Physics-proven. Zero rasterization.*  
**Full Title:** Axon — A Neuro-Symbolic and Physics-Informed Graph Diffusion Framework for Universal Vector-Native Wall Extraction in Architectural Floor Plans  
**Owner:** Tyler (Beteagu / Capsule Manufacturing)  
**Repo Root:** This directory  

---

## High-Level Architecture

This project implements an end-to-end pipeline that **never rasterizes**. It reads raw PDF PostScript operators, builds a geometric graph, refines it through diffusion + symbolic constraints + physics simulation, and outputs IFC-compliant JSON.

```
PDF Input
  → [1] Vector Parser (PostScript → raw graph G₀)
  → [2] Cross-Modal Tokenizer (vector ↔ raster fusion)
  → [3] Graph Diffusion Engine (DDPM denoising → refined graph G*)
  → [4] NeSy SAT Solver (differentiable constraint enforcement)
  → [5] Persistent Homology Loss (topological integrity)
  → [6] PINN / FEA Layer (structural viability)
  → [7] IFC Serializer (structured JSON / IFC output)
```

Pre-training pipeline (offline):
```
Unlabeled PDFs → [8] Masked Primitive Modeling (SSL)
                → [9] SFT on curated datasets
                → [10] GRPO quality annealing
```

---

## Agent System

This project uses **specialized Claude Code agents** organized into teams. Every agent has a narrow scope and communicates through well-defined interfaces. See `AGENTS.md` for full definitions.

### Agent Roster

| Agent ID | Name | Scope |
|----------|------|-------|
| `parse` | **Parser Agent** | PDF ingestion, PostScript operator extraction, raw graph G₀ construction |
| `token` | **Tokenizer Agent** | Cross-modal transformer, vector-raster alignment, TEF fusion |
| `diffuse` | **Diffusion Agent** | DDPM graph denoising, HDSE encoding, reverse process |
| `constrain` | **Constraint Agent** | Differentiable SAT solver, NeSy layer, architectural axiom enforcement |
| `topo` | **Topology Agent** | Persistent homology, Betti numbers, Wasserstein distance loss |
| `physics` | **Physics Agent** | PINN, FEA discretization, JAX-SSO, load-path validation |
| `serial` | **Serializer Agent** | IFC alignment, JSON vocabulary, IfcWallStandardCase mapping |
| `train` | **Training Agent** | SSL pre-training (MPM), SFT, GRPO, data engine |
| `integrate` | **Integration Agent** | End-to-end pipeline, module interfaces, CLI/API surface |
| `qa` | **QA Agent** | Unit tests, integration tests, metric validation, regression |
| `review-arch` | **Architecture Reviewer** | Step 1 review — design correctness, interface contracts |
| `review-eng` | **Engineering Reviewer** | Step 2 review — code quality, performance, edge cases |

### Dispatching Rules

1. **One agent per task.** Never assign overlapping scope.
2. **Interface-first.** Before an agent writes implementation, it must define its input/output contract in `docs/interfaces/`.
3. **No agent writes tests for its own code.** Testing is always `qa` agent's job.
4. **All code passes two review stages** before merging (see `REVIEW.md`).

---

## Directory Structure

```
axon/
├── claude.md                    # This file — project-level instructions
├── AGENTS.md                    # Agent team definitions and responsibilities
├── REVIEW.md                    # 2-step review system specification
├── ARCHITECTURE.md              # Technical architecture deep dive
├── TASKS.md                     # Task breakdown and dependency graph
├── MODEL_SPEC.md                # Original thesis model specification
├── src/
│   ├── parser/                  # [parse] PDF vector extraction
│   │   ├── operators.py         # PostScript operator registry
│   │   ├── extractor.py         # Content stream parser (PyMuPDF)
│   │   ├── graph_builder.py     # Raw graph G₀ construction
│   │   └── filters.py           # Decorative element filtering
│   ├── tokenizer/               # [token] Cross-modal alignment
│   │   ├── vector_tokenizer.py  # Operator → token sequence
│   │   ├── vision_backbone.py   # HRNet / Swin feature extraction
│   │   └── cross_attention.py   # Bidirectional TEF fusion
│   ├── diffusion/               # [diffuse] Graph DDPM
│   │   ├── forward.py           # Noise injection process
│   │   ├── reverse.py           # Denoising / inference
│   │   ├── hdse.py              # Hierarchical Distance Structural Encoding
│   │   └── scheduler.py         # Noise schedule configuration
│   ├── constraints/             # [constrain] NeSy SAT
│   │   ├── sat_solver.py        # Differentiable SAT via convex decomposition
│   │   ├── axioms.py            # Architectural constraint definitions
│   │   └── projector.py         # Gradient projection for snapping
│   ├── topology/                # [topo] Persistent homology
│   │   ├── persistence.py       # Cubical complex + persistence diagrams
│   │   ├── wasserstein.py       # Sinkhorn-Knopp optimal transport
│   │   └── tafl.py              # Topology-Aware Focal Loss
│   ├── physics/                 # [physics] PINN / FEA
│   │   ├── fem_mesh.py          # MITC4 + Euler-Bernoulli discretization
│   │   ├── pinn.py              # PE-PINN with sinusoidal activations
│   │   ├── load_paths.py        # Dead/live load transfer computation
│   │   └── jax_solver.py        # JAX-SSO integration
│   ├── serializer/              # [serial] IFC output
│   │   ├── json_vocab.py        # Custom compressed JSON vocabulary
│   │   ├── ifc_mapper.py        # IfcWallStandardCase + relations
│   │   └── exporter.py          # File output (JSON, IFC, STEP)
│   ├── training/                # [train] Pre-training + fine-tuning
│   │   ├── mpm.py               # Masked Primitive Modeling
│   │   ├── data_engine.py       # Unlabeled PDF corpus loader
│   │   ├── sft.py               # Supervised fine-tuning
│   │   └── grpo.py              # Group Relative Policy Optimization
│   └── pipeline/                # [integrate] End-to-end
│       ├── runner.py            # Full inference pipeline
│       ├── config.py            # Global configuration
│       └── cli.py               # Command-line interface
├── tests/                       # [qa] All testing
│   ├── unit/                    # Per-module unit tests
│   ├── integration/             # Cross-module tests
│   ├── benchmarks/              # Metric validation (HIoU, GED, Betti, etc.)
│   └── fixtures/                # Test PDFs and ground truth data
├── docs/
│   ├── interfaces/              # Module I/O contracts (written before code)
│   ├── decisions/               # Architecture Decision Records (ADRs)
│   └── metrics/                 # Evaluation metric specifications
└── scripts/
    ├── pretrain.sh              # Launch SSL pre-training
    ├── finetune.sh              # Launch SFT + GRPO
    └── evaluate.sh              # Run full benchmark suite
```

---

## Tech Stack

| Layer | Technology | Rationale |
|-------|-----------|-----------|
| PDF Parsing | PyMuPDF (fitz) | Direct PostScript operator access, fast C binding |
| Core ML | PyTorch 2.x | Graph operations, diffusion, autograd |
| Graph Networks | PyG (PyTorch Geometric) | Native sparse graph ops, message passing |
| Physics/FEA | JAX + JAX-SSO | Differentiable FEA, adjoint method, vmap |
| Topology | giotto-tda / gudhi | Persistent homology, persistence diagrams |
| Vision Backbone | timm (HRNet / Swin) | Pretrained multi-scale feature extraction |
| Serialization | IfcOpenShell | IFC schema compliance, ISO 16739-1:2024 |
| Experiment Tracking | Weights & Biases | Metric logging, hyperparameter sweeps |
| Package Management | uv | Fast, deterministic Python env |

---

## Conventions

### Code Style
- Python 3.11+, type hints on all public functions
- Docstrings: Google style
- Formatting: `ruff format`, linting: `ruff check`
- No wildcard imports. Explicit is better.

### Branching
- `main` — stable, reviewed code only
- `dev/<agent-id>/<feature>` — agent working branches
- `review/<agent-id>/<feature>` — branches under review

### Commit Messages
```
[agent-id] scope: short description

Body explaining what changed and why.
Refs: TASKS.md #task-number
```

### Interface Contracts
Every module boundary must have a typed contract in `docs/interfaces/` **before** implementation begins. Format:

```python
# docs/interfaces/parser_to_tokenizer.py
from dataclasses import dataclass
from typing import list

@dataclass
class RawGraph:
    """Output of Parser → Input of Tokenizer"""
    nodes: np.ndarray          # (N, 2) float64 — continuous coordinates
    edges: np.ndarray          # (E, 2) int64 — node index pairs
    stroke_widths: np.ndarray  # (E,) float64
    stroke_colors: np.ndarray  # (E, 4) float64 — RGBA
    operator_types: list[str]  # per-edge: 'lineto' | 'curveto' | 'closepath'
    transform_stack: list[np.ndarray]  # cumulative CTM per path
```

---

## Critical Constraints

1. **Vector-native only.** Rasterization is used solely for cross-modal context — never as a primary representation. The pipeline must work without raster fallback for born-digital PDFs.
2. **Differentiable end-to-end.** Every loss component (reconstruction, SAT, topology, physics) must be differentiable. No detach() hacks that break gradient flow.
3. **IFC compliance is non-negotiable.** Output must import cleanly into Revit 2024+ and ArchiCAD 27+.
4. **Test before merge.** No code enters `main` without passing QA agent's test suite and both review stages.
5. **Interfaces before implementation.** Agents define contracts first, get approval, then build.

---

## How to Run Agents

Each agent is invoked as a Claude Code sub-agent with its scope document loaded:

```bash
# Example: dispatch the Parser Agent
claude-code --agent parse --context "AGENTS.md#parser-agent" --task "TASKS.md#P-001"

# Example: dispatch QA after Parser completes
claude-code --agent qa --context "AGENTS.md#qa-agent" --scope src/parser/ --task "TASKS.md#Q-001"

# Example: trigger Step 1 review
claude-code --agent review-arch --context "REVIEW.md#step-1" --target "dev/parse/operator-registry"
```

Agents read their section of `AGENTS.md` as their system prompt injection. They do not operate outside their defined scope.
