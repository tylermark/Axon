# Axon

**From floor plan to fabrication. Any PDF. Your products. Zero guesswork.**

Axon reads architectural PDF floor plans, extracts every wall, room, and structural element, maps them against Capsule Manufacturing's product catalog, and outputs manufacture-ready panel schedules, BOMs, feasibility reports, and IFC/BIM models with real product SKUs.

## How It Works

```
PDF Floor Plan
  │
  ├─ Layer 1: Extraction
  │   Vector Parser → Cross-Modal Tokenizer → Graph Diffusion → SAT Solver
  │   Output: Clean structural graph + room semantics
  │
  └─ Layer 2: Prefab Intelligence
      Knowledge Graph → Wall Classifier → DRL Panelizer → DRL Placer
      → Feasibility Scorer → BOM Generator → IFC Export
      Output: Panel schedule, BOM, feasibility report, IFC model
```

**Layer 1** converts raw PDF geometry into a clean structural graph using self-supervised pre-training (Masked Primitive Modeling), graph diffusion denoising, and differentiable geometric constraints.

**Layer 2** takes that graph and decides how to build it with Capsule's products. A Knowledge Graph stores the full product catalog (CFS panels, pods, machines, connections). A Deep Reinforcement Learning agent learns optimal panel selection and placement policies. The output is a complete fabrication package.

## Quick Start

```bash
# Install
uv pip install -e .

# Extract structure from a floor plan
axon extract plan.pdf --output result.json

# Full prefab report (panels + BOM + feasibility + IFC)
axon report plan.pdf --output-dir results/

# Batch process a directory of PDFs
axon batch plans/ --output-dir results/
```

## Training

Training runs on Google Colab with GPU. Open `notebooks/axon_training_colab.ipynb` and run all cells:

| Phase | What it trains | Time (RTX 6000 Pro) |
|-------|---------------|---------------------|
| MPM | Self-supervised primitive reconstruction | ~10 min |
| SFT | Supervised graph extraction (tokenizer + diffusion + constraints) | ~30-90 min |
| GRPO | Quality annealing with reward-based optimization | TBD |
| DRL | Panel selection + pod placement policy | ~5 min |

Shell scripts for non-Colab training:
```bash
scripts/pretrain.sh    # MPM pre-training
scripts/finetune.sh    # SFT + GRPO
scripts/train_drl.sh   # DRL panelization
```

## Project Structure

```
src/
├── parser/           # PDF vector extraction
├── tokenizer/        # Cross-modal transformer (vector + raster fusion)
├── diffusion/        # Graph DDPM denoising
├── constraints/      # Differentiable SAT solver (geometric axioms)
├── knowledge_graph/  # Capsule product catalog + query APIs
├── classifier/       # Wall type classification (bearing/partition/shear/fire)
├── drl/              # Reinforcement learning (panelization + placement)
├── feasibility/      # Prefab percentage scoring + blockers
├── bom/              # Bill of materials + cost estimation
├── transplant/       # BIM assembly + IFC export
├── training/         # MPM, SFT, GRPO, DRL training pipelines
└── pipeline/         # End-to-end orchestration + CLI
```

## Tech Stack

| Component | Technology |
|-----------|-----------|
| PDF Parsing | PyMuPDF |
| ML Framework | PyTorch 2.x |
| Graph Networks | PyTorch Geometric |
| Knowledge Graph | NetworkX + JSON |
| Reinforcement Learning | Stable-Baselines3 (MaskablePPO) |
| Vision Backbone | timm (HRNet / Swin) |
| BIM/IFC Export | IfcOpenShell |
| Experiment Tracking | Weights & Biases |
| Package Management | uv |

## Requirements

- Python 3.11+
- CUDA GPU recommended for training
- `uv` for package management (or `pip`)

```bash
# Development install with linting/testing tools
pip install -e ".[dev]"

# Run tests
pytest tests/

# Lint
ruff check src/ && ruff format --check src/
```

## License

MIT
