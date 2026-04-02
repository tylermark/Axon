# CLAUDE.md — Axon

## Project Identity

**Name:** Axon  
**Tagline:** *From floor plan to fabrication. Any PDF. Your products. Zero guesswork.*  
**Full Title:** Axon — A Neuro-Symbolic Graph Diffusion Framework for Universal Vector-Native Wall Extraction and Automated Prefabricated Component Placement  
**Owner:** Tyler (Beteagu / Capsule Manufacturing)  
**Repo Root:** This directory  

---

## What Axon Does

A GC or architect uploads a PDF floor plan. Axon reads every wall, room, and structural element — then maps it against Capsule Manufacturing's products using a Knowledge Graph, panelizes walls and places products using Deep Reinforcement Learning, and exports a manufacture-ready BIM model with real product SKUs.

```
┌─────────────────────────────────────────────────────────────┐
│  LAYER 1 — EXTRACTION                                        │
│  "Read and understand any floor plan PDF"                    │
│                                                              │
│  PDF Input                                                   │
│    → [1] Vector Parser (PostScript → raw graph G₀)  ✅ DONE  │
│    → [2] Cross-Modal Tokenizer (vector ↔ raster fusion)      │
│    → [3] Graph Diffusion Engine (DDPM → refined graph G*)    │
│    → [4] NeSy SAT Solver (geometric constraint enforcement)  │
│                                                              │
│  Output: Clean structural graph + room semantics             │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  LAYER 2 — PREFAB INTELLIGENCE                               │
│  "Map it to Capsule's world and make it buildable"           │
│                                                              │
│  Structural Graph                                            │
│    → [5] Knowledge Graph (Capsule product catalog + rules)   │
│    → [6] Wall Classifier (bearing/partition/shear/fire)      │
│    → [7] DRL Panelizer (walls → CFS panel assemblies)        │
│    → [8] DRL Placer (rooms → Capsule pods/products)          │
│    → [9] Feasibility Scorer (prefab % + blockers)            │
│    → [10] BOM Generator (quantities + cost)                  │
│    → [11] BIM Transplant + IFC Export (3D model output)      │
│                                                              │
│  Output: Prefab report, panel schedule, BOM, IFC model       │
└─────────────────────────────────────────────────────────────┘
```

Training pipeline (offline):
```
Unlabeled PDFs → Masked Primitive Modeling (SSL)
               → SFT on curated datasets
               → GRPO quality annealing
DRL training   → Simulated panelization episodes on extracted graphs
```

---

## Agent System

### Layer 1 — Extraction Agents

| Agent ID | Name | Scope | Status |
|----------|------|-------|--------|
| `parse` | **Parser Agent** | PDF ingestion, PostScript extraction, raw graph G₀ | ✅ DONE |
| `token` | **Tokenizer Agent** | Cross-modal transformer, vector-raster TEF fusion | TODO |
| `diffuse` | **Diffusion Agent** | DDPM graph denoising, HDSE encoding | TODO |
| `constrain` | **Constraint Agent** | Differentiable SAT solver, geometric axioms, topo regularization | TODO |

### Layer 2 — Prefab Intelligence Agents

| Agent ID | Name | Scope |
|----------|------|-------|
| `catalog` | **Knowledge Graph Agent** | Product catalog, machine specs, fabrication constraints, compliance rules |
| `classify` | **Wall Classifier Agent** | Structural classification: bearing, partition, shear, fire-rated, envelope |
| `drl` | **DRL Agent** | Panelization + product placement via Deep Reinforcement Learning |
| `feasibility` | **Feasibility Agent** | Prefab percentage scoring, blocker ID, design suggestions |
| `bom` | **BOM Agent** | Bill of materials, quantity takeoffs, cost estimation |
| `transplant` | **BIM Transplant Agent** | KG → BIM family lookup, 3D model assembly, IFC serialization |

### Infrastructure Agents

| Agent ID | Name | Scope |
|----------|------|-------|
| `train` | **Training Agent** | SSL pre-training (MPM), SFT, GRPO, DRL training |
| `integrate` | **Integration Agent** | End-to-end pipeline, interfaces, CLI |
| `qa` | **QA Agent** | All testing, benchmarks, regression |
| `review-arch` | **Architecture Reviewer** | Step 1 — design correctness, interface contracts |
| `review-eng` | **Engineering Reviewer** | Step 2 — code quality, performance, edge cases |

### Dispatching Rules

1. **One agent per task.** Never assign overlapping scope.
2. **Interface-first.** Define input/output contract in `docs/interfaces/` before implementation.
3. **No agent writes tests for its own code.** Testing is always `qa` agent's job.
4. **All code passes two review stages** before merging (see `REVIEW.md`).

---

## Directory Structure

```
axon/
├── claude.md
├── AGENTS.md
├── REVIEW.md
├── ARCHITECTURE.md
├── TASKS.md
├── MODEL_SPEC.md
├── src/
│   │── # ═══ LAYER 1: EXTRACTION ═══
│   ├── parser/                  # [parse] ✅ DONE
│   ├── tokenizer/               # [token] Cross-modal alignment
│   ├── diffusion/               # [diffuse] Graph DDPM
│   ├── constraints/             # [constrain] NeSy SAT + topo regularization
│   │
│   │── # ═══ LAYER 2: PREFAB INTELLIGENCE ═══
│   ├── knowledge_graph/         # [catalog] Product KG
│   │   ├── schema.py            # KG node/edge types, entity definitions
│   │   ├── loader.py            # Ingest product data into graph
│   │   ├── query.py             # Query API for panel/pod/machine lookup
│   │   └── data/                # Product data files
│   │       ├── panels.json
│   │       ├── pods.json
│   │       ├── machines.json
│   │       └── connections.json
│   ├── classifier/              # [classify] Wall type classification
│   ├── drl/                     # [drl] Reinforcement learning
│   │   ├── env.py               # Floor plan environment (state, actions)
│   │   ├── agent.py             # DRL policy network
│   │   ├── panelizer.py         # Wall → panel segmentation actions
│   │   ├── placer.py            # Room → product placement actions
│   │   └── reward.py            # Reward function (SPUR, waste, violations)
│   ├── feasibility/             # [feasibility] Prefab scoring
│   ├── bom/                     # [bom] Bill of materials
│   ├── transplant/              # [transplant] BIM Library Transplant + IFC
│   │   ├── matcher.py           # KG lookup → BIM family matching
│   │   ├── assembler.py         # 3D model assembly from placed panels
│   │   └── ifc_export.py        # IFC serialization (ISO 16739-1:2024)
│   │
│   │── # ═══ INFRASTRUCTURE ═══
│   ├── training/                # [train] Pre-training + DRL training
│   └── pipeline/                # [integrate] End-to-end
├── tests/
├── docs/
│   ├── interfaces/
│   └── decisions/
└── scripts/
```

---

## Tech Stack

| Layer | Technology | Rationale |
|-------|-----------|-----------|
| PDF Parsing | PyMuPDF (fitz) | Direct PostScript operator access |
| Core ML | PyTorch 2.x | Graph ops, diffusion, autograd |
| Graph Networks | PyG (PyTorch Geometric) | Sparse graph ops, message passing |
| Knowledge Graph | Neo4j or NetworkX + JSON | Structured product catalog queries |
| Reinforcement Learning | Stable-Baselines3 / CleanRL | DRL for panelization + placement |
| Vision Backbone | timm (HRNet / Swin) | Multi-scale raster feature extraction |
| BIM/IFC | IfcOpenShell | IFC schema compliance |
| Experiment Tracking | Weights & Biases | Metric logging, sweeps |
| Package Management | uv | Fast, deterministic Python env |

---

## Conventions

### Code Style
- Python 3.11+, type hints on all public functions
- Docstrings: Google style
- Formatting: `ruff format`, linting: `ruff check`

### Branching
- `main` — stable, reviewed code only
- `dev/<agent-id>/<feature>` — agent working branches

### Commit Messages
```
[agent-id] scope: short description
```

### Interface Contracts
Every module boundary must have a typed contract in `docs/interfaces/` before implementation begins.

---

## Critical Constraints

1. **Vector-native only.** Rasterization is used solely for cross-modal context.
2. **Knowledge Graph is deterministic.** Product matching must be exact KG lookups, never probabilistic guessing.
3. **DRL must respect KG constraints.** The RL agent can only place components that the KG confirms are valid.
4. **IFC compliance is non-negotiable.** Output must import into Revit 2024+ and ArchiCAD 27+.
5. **Test before merge.** All code passes QA + both review stages.
6. **Interfaces before implementation.**
