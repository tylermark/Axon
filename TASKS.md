# TASKS.md — Axon Task Breakdown & Dependency Graph

## Task ID Convention

```
[Agent Prefix]-[Sequential Number]
```

| Prefix | Agent |
|--------|-------|
| P | Parser |
| T | Tokenizer |
| D | Diffusion |
| C | Constraint |
| H | Topology (Homology) |
| F | Physics (FEA) |
| S | Serializer |
| TR | Training |
| I | Integration |
| Q | QA |

---

## Phase 0: Foundations

> Interface contracts and project scaffolding. Must complete before any implementation.

| ID | Agent | Task | Depends On | Status |
|----|-------|------|------------|--------|
| I-001 | integrate | Create project scaffolding (directory structure, pyproject.toml, ruff config) | — | TODO |
| I-002 | integrate | Define `RawGraph` interface contract (`docs/interfaces/parser_to_tokenizer.py`) | — | TODO |
| I-003 | integrate | Define `EnrichedTokenSequence` interface contract (`docs/interfaces/tokenizer_to_diffusion.py`) | — | TODO |
| I-004 | integrate | Define `RefinedStructuralGraph` interface contract (`docs/interfaces/diffusion_output.py`) | — | TODO |
| I-005 | integrate | Define `ConstraintGradients` interface contract (`docs/interfaces/constraint_signals.py`) | — | TODO |
| I-006 | integrate | Define `TopologyLoss` interface contract (`docs/interfaces/topology_loss.py`) | — | TODO |
| I-007 | integrate | Define `PhysicsLoss` interface contract (`docs/interfaces/physics_loss.py`) | — | TODO |
| I-008 | integrate | Define `FinalizedGraph` interface contract (`docs/interfaces/graph_to_serializer.py`) | — | TODO |
| I-009 | integrate | Define global `Config` schema (`src/pipeline/config.py`) | — | TODO |

---

## Phase 1: Parser

> PDF vector extraction — the foundation of the entire pipeline.

| ID | Agent | Task | Depends On | Status |
|----|-------|------|------------|--------|
| P-001 | parse | Implement PostScript operator registry (`operators.py`) | I-001 | TODO |
| P-002 | parse | Implement PyMuPDF content stream extractor (`extractor.py`) | P-001 | TODO |
| P-003 | parse | Implement graphics state stack tracking (CTM, stroke, fill, clip) | P-002 | TODO |
| P-004 | parse | Implement Bézier curve → polyline sampling with metadata preservation | P-002 | TODO |
| P-005 | parse | Implement KD-tree vertex deduplication within tolerance | P-002 | TODO |
| P-006 | parse | Implement raw graph G₀ builder (`graph_builder.py`) | P-003, P-004, P-005 | TODO |
| P-007 | parse | Implement decorative element flagging heuristics (`filters.py`) | P-006 | TODO |
| P-008 | parse | Validate against ARCH E test sheet (~50K paths, <2s target) | P-006 | TODO |
| Q-001 | qa | Write unit tests for parser module | P-007 | TODO |
| Q-002 | qa | Write integration test: PDF → RawGraph end-to-end | P-007, I-002 | TODO |

---

## Phase 2: Tokenizer

> Cross-modal feature alignment. Can begin once Parser interface is stable.

| ID | Agent | Task | Depends On | Status |
|----|-------|------|------------|--------|
| T-001 | token | Implement vector token embedding layer | I-002, I-003 | TODO |
| T-002 | token | Implement 2D learned positional encoding | T-001 | TODO |
| T-003 | token | Integrate HRNet/Swin backbone via timm for raster features | I-001 | TODO |
| T-004 | token | Implement vision-to-vector cross-attention module | T-001, T-003 | TODO |
| T-005 | token | Implement vector-to-vision cross-attention module | T-001, T-003 | TODO |
| T-006 | token | Implement TEF fusion combining both attention directions | T-004, T-005 | TODO |
| T-007 | token | Implement spatial attention windowing (bounded radius) | T-006 | TODO |
| T-008 | token | Implement vector-only fallback mode (no raster available) | T-006 | TODO |
| Q-003 | qa | Write unit tests for tokenizer module | T-008 | TODO |
| Q-004 | qa | Write integration test: RawGraph → EnrichedTokenSequence | T-008, Q-002 | TODO |

---

## Phase 3: Diffusion Engine

> Core generative model. Can begin once Tokenizer interface is defined.

| ID | Agent | Task | Depends On | Status |
|----|-------|------|------------|--------|
| D-001 | diffuse | Implement cosine noise schedule | I-003, I-004 | TODO |
| D-002 | diffuse | Implement forward diffusion (continuous coordinates + discrete adjacency) | D-001 | TODO |
| D-003 | diffuse | Implement HDSE — shortest-path distance encoding | D-001 | TODO |
| D-004 | diffuse | Implement HDSE — random walk similarity encoding | D-003 | TODO |
| D-005 | diffuse | Implement HDSE — hierarchical level encoding | D-004 | TODO |
| D-006 | diffuse | Implement linear transformer backbone with HDSE attention bias | D-005 | TODO |
| D-007 | diffuse | Implement reverse denoising conditioned on context c | D-006 | TODO |
| D-008 | diffuse | Implement DDIM sampling for fast inference (50 steps) | D-007 | TODO |
| D-009 | diffuse | Implement variational lower bound loss computation | D-007 | TODO |
| Q-005 | qa | Write unit tests for diffusion module | D-009 | TODO |
| Q-006 | qa | Write integration test: EnrichedTokenSequence → RefinedStructuralGraph | D-009, Q-004 | TODO |

---

## Phase 4: Constraints & Topology

> Can proceed in parallel once Diffusion output interface is defined.

| ID | Agent | Task | Depends On | Status |
|----|-------|------|------------|--------|
| C-001 | constrain | Implement orthogonal integrity axiom (cosine similarity penalty) | I-004, I-005 | TODO |
| C-002 | constrain | Implement parallel pair constancy axiom (IQR-bounded distance) | C-001 | TODO |
| C-003 | constrain | Implement junction closure axiom (Graph Laplacian penalty) | C-001 | TODO |
| C-004 | constrain | Implement spatial non-intersection axiom | C-001 | TODO |
| C-005 | constrain | Implement differentiable SAT solver (convex decomposition) | C-001 | TODO |
| C-006 | constrain | Implement axiom weight meta-learning | C-005 | TODO |
| C-007 | constrain | Implement hard projection / snapping for inference | C-005 | TODO |
| C-008 | constrain | Build configurable axiom registry | C-001 thru C-004 | TODO |
| H-001 | topo | Implement cubical complex construction from graph | I-004, I-006 | TODO |
| H-002 | topo | Implement persistence diagram computation (Betti-0, Betti-1) | H-001 | TODO |
| H-003 | topo | Implement Sinkhorn-Knopp optimal transport | H-002 | TODO |
| H-004 | topo | Implement Wasserstein distance computation | H-003 | TODO |
| H-005 | topo | Implement Topology-Aware Focal Loss (TAFL) | H-004 | TODO |
| Q-007 | qa | Write unit tests for constraint module | C-008 | TODO |
| Q-008 | qa | Write unit tests for topology module | H-005 | TODO |

---

## Phase 5: Physics Layer

> Depends on Constraint module for wall thickness data.

| ID | Agent | Task | Depends On | Status |
|----|-------|------|------------|--------|
| F-001 | physics | Implement wall → MITC4 shell element discretization | I-004, I-007 | TODO |
| F-002 | physics | Implement wall → Euler-Bernoulli beam-column element discretization | F-001 | TODO |
| F-003 | physics | Implement load application (dead + live loads) | F-002 | TODO |
| F-004 | physics | Implement PE-PINN with sinusoidal activations | F-001 | TODO |
| F-005 | physics | Integrate JAX-SSO solver | F-003 | TODO |
| F-006 | physics | Implement adjoint method for gradient computation | F-005 | TODO |
| F-007 | physics | Implement physics loss (displacement MSE + stress constraint) | F-006 | TODO |
| F-008 | physics | Implement structural viability report output | F-007 | TODO |
| Q-009 | qa | Write unit tests for physics module | F-008 | TODO |

---

## Phase 6: Serializer

> Can begin interface design early; implementation needs finalized graph.

| ID | Agent | Task | Depends On | Status |
|----|-------|------|------------|--------|
| S-001 | serial | Define compressed JSON vocabulary | I-008 | TODO |
| S-002 | serial | Implement graph → JSON tokenization (structure-first hierarchy) | S-001 | TODO |
| S-003 | serial | Implement IfcWallStandardCase mapping | S-002 | TODO |
| S-004 | serial | Implement SweptSolid shape representation from graph geometry | S-003 | TODO |
| S-005 | serial | Implement opening attachment (IfcRelVoidsElement) | S-004 | TODO |
| S-006 | serial | Implement room semantics (IfcSpace, IfcRelSpaceBoundary) | S-005 | TODO |
| S-007 | serial | Implement IFC-SPF export via IfcOpenShell | S-006 | TODO |
| S-008 | serial | Implement JSON export | S-002 | TODO |
| S-009 | serial | Validate Revit 2024+ import | S-007 | TODO |
| S-010 | serial | Validate ArchiCAD 27+ import | S-007 | TODO |
| Q-010 | qa | Write unit tests for serializer module | S-008 | TODO |
| Q-011 | qa | Write IFC round-trip integration test | S-007 | TODO |

---

## Phase 7: Training Pipeline

> Can begin once all model components exist.

| ID | Agent | Task | Depends On | Status |
|----|-------|------|------------|--------|
| TR-001 | train | Implement Masked Primitive Modeling (MPM) pre-training loop | T-008, D-009 | TODO |
| TR-002 | train | Implement unlabeled PDF data engine | P-007 | TODO |
| TR-003 | train | Implement Chamfer Distance + coordinate regression loss | TR-001 | TODO |
| TR-004 | train | Implement SFT training loop with L_total | All Phase 3-5 | TODO |
| TR-005 | train | Implement GRPO quality annealing | TR-004 | TODO |
| TR-006 | train | Implement curriculum loss scheduling | TR-004 | TODO |
| TR-007 | train | Implement W&B integration and checkpointing | TR-004 | TODO |
| TR-008 | train | Write training scripts (pretrain.sh, finetune.sh) | TR-007 | TODO |

---

## Phase 8: Integration & Benchmarking

> Final assembly and evaluation.

| ID | Agent | Task | Depends On | Status |
|----|-------|------|------------|--------|
| I-010 | integrate | Wire full inference pipeline (runner.py) | All phases | TODO |
| I-011 | integrate | Implement CLI interface | I-010 | TODO |
| I-012 | integrate | Implement batch processing mode | I-011 | TODO |
| Q-012 | qa | Implement HIoU benchmark | I-010 | TODO |
| Q-013 | qa | Implement Graph Edit Distance benchmark | I-010 | TODO |
| Q-014 | qa | Implement Betti Number Error benchmark | I-010 | TODO |
| Q-015 | qa | Implement PINN Stress/Load Variance benchmark | I-010 | TODO |
| Q-016 | qa | Implement LayoutGKN / SSIG benchmark | I-010 | TODO |
| Q-017 | qa | Run full evaluation on CubiCasa5K test set | Q-012 thru Q-016 | TODO |
| Q-018 | qa | Run full evaluation on MSD test set | Q-012 thru Q-016 | TODO |

---

## Dependency Graph (Simplified)

```
Phase 0 (Interfaces)
    │
    ├── Phase 1 (Parser)
    │       │
    │       └── Phase 2 (Tokenizer)
    │               │
    │               └── Phase 3 (Diffusion)
    │                       │
    │                       ├── Phase 4a (Constraints) ──┐
    │                       ├── Phase 4b (Topology) ─────┤
    │                       │                            │
    │                       └── Phase 5 (Physics) ───────┤
    │                                                    │
    │                       Phase 6 (Serializer) ◄───────┘
    │                               │
    │                       Phase 7 (Training)
    │                               │
    └───────────────────── Phase 8 (Integration & Benchmarks)
```

**Parallelization opportunities:**
- Phases 4a, 4b can run in parallel
- Phase 6 interface design can start during Phase 3
- Phase 7 (data engine TR-002) can start during Phase 1
- QA tasks run continuously as modules complete
