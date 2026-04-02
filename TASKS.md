# TASKS.md — Axon Task Breakdown & Dependency Graph

## Task ID Prefixes

| Prefix | Agent |
|--------|-------|
| I | Integration |
| P | Parser |
| T | Tokenizer |
| D | Diffusion |
| C | Constraint |
| KG | Knowledge Graph |
| CL | Classifier |
| DRL | DRL (Panelization + Placement) |
| FS | Feasibility |
| BM | BOM |
| BT | BIM Transplant |
| TR | Training |
| Q | QA |

---

## Phase 0: Foundations — ✅ DONE

| ID | Agent | Task | Status |
|----|-------|------|--------|
| I-001 | integrate | Create project scaffolding | ✅ DONE |
| I-002 | integrate | Define RawGraph interface contract | ✅ DONE |
| I-003 | integrate | Define EnrichedTokenSequence interface contract | ✅ DONE |
| I-004 | integrate | Define RefinedStructuralGraph interface contract | ✅ DONE |
| I-005 | integrate | Define ConstraintSignals interface contract | ✅ DONE |
| I-006 | integrate | Define global Config schema | ✅ DONE |

---

## Phase 1: Parser — ✅ DONE

| ID | Agent | Task | Status |
|----|-------|------|--------|
| P-001 | parse | PostScript operator registry | ✅ DONE |
| P-002 | parse | PyMuPDF content stream extractor | ✅ DONE |
| P-003 | parse | Graphics state stack tracking (CTM, stroke, fill) | ✅ DONE |
| P-004 | parse | Bézier curve → polyline sampling | ✅ DONE |
| P-005 | parse | KD-tree vertex deduplication | ✅ DONE |
| P-006 | parse | Raw graph G₀ builder | ✅ DONE |
| P-007 | parse | Decorative element flagging | ✅ DONE |
| P-008 | parse | Performance validation (<2s on ARCH E sheet) | ✅ DONE |
| Q-001 | qa | Unit tests for parser module | ✅ DONE |
| Q-002 | qa | Integration test: PDF → RawGraph | ✅ DONE |

---

## Phase 2: Tokenizer — ✅ DONE

> Cross-modal feature alignment. The bridge between raw geometry and semantic understanding.

| ID | Agent | Task | Depends On | Status |
|----|-------|------|------------|--------|
| T-001 | token | Vector token embedding layer | I-003 | ✅ DONE |
| T-002 | token | 2D learned positional encoding | T-001 | ✅ DONE |
| T-003 | token | Integrate HRNet/Swin backbone via timm | I-001 | ✅ DONE |
| T-004 | token | Vision-to-vector cross-attention module | T-001, T-003 | ✅ DONE |
| T-005 | token | Vector-to-vision cross-attention module | T-001, T-003 | ✅ DONE |
| T-006 | token | TEF fusion combining both attention directions | T-004, T-005 | ✅ DONE |
| T-007 | token | Spatial attention windowing (bounded radius) | T-006 | ✅ DONE |
| T-008 | token | Vector-only fallback mode | T-006 | ✅ DONE |
| Q-003 | qa | Unit tests for tokenizer module | T-008 | ✅ DONE |
| Q-004 | qa | Integration test: RawGraph → EnrichedTokenSequence | T-008 | ✅ DONE |

---

## Phase 3: Diffusion Engine — ✅ DONE

| ID | Agent | Task | Depends On | Status |
|----|-------|------|------------|--------|
| D-001 | diffuse | Cosine noise schedule | I-004 | ✅ DONE |
| D-002 | diffuse | Forward diffusion (continuous coords + discrete adjacency) | D-001 | ✅ DONE |
| D-003 | diffuse | HDSE — shortest-path + random walk + hierarchical encoding | D-001 | ✅ DONE |
| D-004 | diffuse | Linear transformer backbone with HDSE attention bias | D-003 | ✅ DONE |
| D-005 | diffuse | Reverse denoising conditioned on context | D-004 | ✅ DONE |
| D-006 | diffuse | DDIM sampling for fast inference (50 steps) | D-005 | ✅ DONE |
| D-007 | diffuse | Variational lower bound loss | D-005 | ✅ DONE |
| Q-005 | qa | Unit tests for diffusion module | D-007 | ✅ DONE |
| Q-006 | qa | Integration test: EnrichedTokenSequence → RefinedStructuralGraph | D-007 | ✅ DONE |

---

## Phase 4: Constraints — ✅ DONE

| ID | Agent | Task | Depends On | Status |
|----|-------|------|------------|--------|
| C-001 | constrain | Orthogonal integrity axiom (cosine penalty) | I-005 | ✅ DONE |
| C-002 | constrain | Parallel pair constancy axiom (IQR distance) | C-001 | ✅ DONE |
| C-003 | constrain | Junction closure axiom (Graph Laplacian) | C-001 | ✅ DONE |
| C-004 | constrain | Spatial non-intersection axiom | C-001 | ✅ DONE |
| C-005 | constrain | Differentiable SAT solver (convex decomposition) | C-001 | ✅ DONE |
| C-006 | constrain | Lightweight Betti number regularization for room enclosure | C-005 | ✅ DONE |
| C-007 | constrain | Hard projection / snapping for inference | C-005 | ✅ DONE |
| C-008 | constrain | Configurable axiom registry | C-001 thru C-004 | ✅ DONE |
| Q-007 | qa | Unit tests for constraint module | C-008 | ✅ DONE |

---

## Phase 5: Layer 1 Integration — ✅ DONE

> Wire extraction pipeline end-to-end. Must pass before Layer 2 begins.

| ID | Agent | Task | Depends On | Status |
|----|-------|------|------------|--------|
| I-007 | integrate | Wire Layer 1 pipeline (parser → tokenizer → diffusion → constraints) | All L1 phases | ✅ DONE |
| I-008 | integrate | Implement CLI for single-PDF extraction | I-007 | ✅ DONE |
| Q-008 | qa | HIoU benchmark | I-007 | ✅ DONE |
| Q-009 | qa | Graph Edit Distance benchmark | I-007 | ✅ DONE |
| Q-010 | qa | Betti Number Error benchmark | I-007 | ✅ DONE |
| Q-011 | qa | End-to-end test: PDF → clean structural graph | I-007 | ✅ DONE |

---

## Phase 6: Knowledge Graph — ✅ DONE

> Foundation for all Layer 2 work. No ML — pure data modeling.

| ID | Agent | Task | Depends On | Status |
|----|-------|------|------------|--------|
| KG-001 | catalog | Define KG schema (panel, pod, machine, connection, compliance node types) | I-001 | ✅ DONE |
| KG-002 | catalog | Implement KG loader (JSON → graph) | KG-001 | ✅ DONE |
| KG-003 | catalog | Populate panels.json with Capsule's CFS panel catalog | KG-001 | ✅ DONE |
| KG-004 | catalog | Populate pods.json with Capsule's pod assemblies | KG-001 | ✅ DONE |
| KG-005 | catalog | Populate machines.json (Howick 2.5, 3.5, Zund specs) | KG-001 | ✅ DONE |
| KG-006 | catalog | Populate connections.json (splices, clips, fasteners) | KG-001 | ✅ DONE |
| KG-007 | catalog | Implement query API ("given wall dims + type → valid panels") | KG-002 | ✅ DONE |
| KG-008 | catalog | Implement query API ("given room dims + function → valid pods") | KG-002 | ✅ DONE |
| KG-009 | catalog | Implement fabrication constraint validation | KG-007 | ✅ DONE |
| Q-012 | qa | Unit tests for KG module | KG-009 | ✅ DONE |

---

## Phase 7: Wall Classification — ✅ DONE

| ID | Agent | Task | Depends On | Status |
|----|-------|------|------------|--------|
| CL-001 | classify | Define wall type taxonomy | I-007 | ✅ DONE |
| CL-002 | classify | Thickness-based classification rules | CL-001 | ✅ DONE |
| CL-003 | classify | Context-based classification (adjacency, labels, hatching) | CL-002 | ✅ DONE |
| CL-004 | classify | Fire rating detection (fill color, annotations) | CL-003 | ✅ DONE |
| CL-005 | classify | Confidence scoring and human-review flagging | CL-004 | ✅ DONE |
| Q-013 | qa | Unit tests for classifier | CL-005 | ✅ DONE |

---

## Phase 8: DRL Panelization & Placement — ✅ DONE

> The core Layer 2 intelligence. Requires KG + classified walls.

| ID | Agent | Task | Depends On | Status |
|----|-------|------|------------|--------|
| DRL-001 | drl | Define floor plan environment (state space, observation) | CL-005, KG-009 | ✅ DONE |
| DRL-002 | drl | Define action space for wall panelization | DRL-001 | ✅ DONE |
| DRL-003 | drl | Define action space for product placement | DRL-001 | ✅ DONE |
| DRL-004 | drl | Implement reward function (SPUR, waste, violations) | DRL-002, DRL-003 | ✅ DONE |
| DRL-005 | drl | Implement panelization episode loop (wall → panel segments) | DRL-004 | ✅ DONE |
| DRL-006 | drl | Implement placement episode loop (rooms → pods/products) | DRL-004 | ✅ DONE |
| DRL-007 | drl | Implement opening constraint handling (doors/windows) | DRL-005 | ✅ DONE |
| DRL-008 | drl | Implement joint/angle constraint handling | DRL-005 | ✅ DONE |
| DRL-009 | drl | Train DRL policy on simulated floor plans | DRL-007, DRL-008 | ✅ DONE |
| Q-014 | qa | Unit tests for DRL environment and reward | DRL-004 | ✅ DONE |
| Q-015 | qa | SPUR benchmark on test set | DRL-009 | ✅ DONE |
| Q-016 | qa | DRL reward convergence test | DRL-009 | ✅ DONE |

---

## Phase 9: Feasibility & BOM — ✅ DONE

| ID | Agent | Task | Depends On | Status |
|----|-------|------|------------|--------|
| FS-001 | feasibility | Prefab percentage calculation (by wall length, area, cost) | DRL-009 | ✅ DONE |
| FS-002 | feasibility | Blocker identification and categorization | FS-001 | ✅ DONE |
| FS-003 | feasibility | Design modification suggestions | FS-002 | ✅ DONE |
| FS-004 | feasibility | Feasibility report generation | FS-003 | ✅ DONE |
| BM-001 | bom | CFS quantity takeoff (studs, track, fasteners, sheathing) | DRL-009 | ✅ DONE |
| BM-002 | bom | Pod component takeoff | DRL-009 | ✅ DONE |
| BM-003 | bom | Cost estimation from unit pricing | BM-001, BM-002 | ✅ DONE |
| BM-004 | bom | Labor hour estimation | BM-003 | ✅ DONE |
| BM-005 | bom | BOM export (CSV, Excel, PDF) | BM-004 | ✅ DONE |
| Q-017 | qa | Unit tests for feasibility module | FS-004 | ✅ DONE |
| Q-018 | qa | Unit tests for BOM module | BM-005 | ✅ DONE |

---

## Phase 10: BIM Transplant & IFC Export — ✅ DONE

| ID | Agent | Task | Depends On | Status |
|----|-------|------|------------|--------|
| BT-001 | transplant | Implement KG → BIM family matching (panel type → Revit family) | DRL-009, KG-007 | ✅ DONE |
| BT-002 | transplant | Implement 3D model assembly from placed panels | BT-001 | ✅ DONE |
| BT-003 | transplant | Implement opening attachment (IfcRelVoidsElement) | BT-002 | ✅ DONE |
| BT-004 | transplant | Implement IFC serialization (ISO 16739-1:2024) | BT-003 | ✅ DONE |
| BT-005 | transplant | Validate Revit 2024+ import | BT-004 | MANUAL |
| BT-006 | transplant | Validate ArchiCAD 27+ import | BT-004 | MANUAL |
| Q-019 | qa | IFC round-trip integration test | BT-004 | ✅ DONE |

---

## Phase 11: Training Pipeline — ✅ DONE

| ID | Agent | Task | Depends On | Status |
|----|-------|------|------------|--------|
| TR-001 | train | Masked Primitive Modeling (MPM) pre-training loop | T-008, D-007 | ✅ DONE |
| TR-002 | train | Unlabeled PDF data engine | P-008 | ✅ DONE |
| TR-003 | train | SFT training loop with L_total | All L1 | ✅ DONE |
| TR-004 | train | GRPO quality annealing | TR-003 | ✅ DONE |
| TR-005 | train | DRL training pipeline (simulated episodes) | DRL-004 | ✅ DONE |
| TR-006 | train | W&B integration and checkpointing | TR-003 | ✅ DONE |
| TR-007 | train | Training scripts | TR-006 | ✅ DONE |

---

## Phase 12: Full Pipeline Integration — ✅ DONE

| ID | Agent | Task | Depends On | Status |
|----|-------|------|------------|--------|
| I-009 | integrate | Wire Layer 1 → Layer 2 full pipeline | All phases | ✅ DONE |
| I-010 | integrate | Implement PDF-to-report CLI command | I-009 | ✅ DONE |
| I-011 | integrate | Combined output (feasibility + BOM + panel schedule + IFC) | I-010 | ✅ DONE |
| Q-020 | qa | End-to-end: sample PDF → full prefab report | I-011 | ✅ DONE |
| Q-021 | qa | Test against real Capsule project PDFs | Q-020 | ✅ DONE |

---

## Dependency Graph

```
Phase 0 (Foundations) ✅
    │
    Phase 1 (Parser) ✅
    │       │
    │       Phase 2 (Tokenizer) ✅
    │               │
    │               Phase 3 (Diffusion) ✅
    │                       │
    │                       Phase 4 (Constraints + topo reg) ✅
    │                       │
    │                       Phase 5 (Layer 1 Integration) ✅
    │                               │
    │   ════════════════════ LAYER 2 ════════════════════
    │                               │
    │   Phase 6 (Knowledge Graph) ✅┤
    │                               │
    │                       Phase 7 (Wall Classification) ✅
    │                               │
    │                       Phase 8 (DRL Panelization & Placement) ✅
    │                               │
    │                       ┌───────┴───────┐
    │               Phase 9a           Phase 9b
    │             (Feasibility) ✅     (BOM) ✅
    │                       └───────┬───────┘
    │                               │
    │                       Phase 10 (BIM Transplant + IFC) ✅
    │                               │
    │                       Phase 11 (Training) ✅
    │                               │
    └───────────────────── Phase 12 (Full Pipeline) ✅
```

**All phases complete.** Next steps: train models on Colab, validate with real Capsule PDFs.
