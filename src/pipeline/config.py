"""Global configuration schema for the Axon pipeline.

Centralizes all hyperparameters and settings across all pipeline stages.
Uses Pydantic for validation and serialization.

Each sub-config corresponds to one pipeline stage, matching the agent
boundaries defined in AGENTS.md.

Reference: ARCHITECTURE.md (all stages), MODEL_SPEC.md (hyperparameters).
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Stage 1: Parser
# ---------------------------------------------------------------------------


class ParserConfig(BaseModel):
    """Configuration for the PDF vector parser (Stage 1)."""

    bezier_sample_resolution: int = Field(
        default=8,
        ge=2,
        le=64,
        description="Number of polyline segments per sampled Bézier curve.",
    )
    vertex_merge_tolerance: float = Field(
        default=0.5,
        ge=0.0,
        description="KD-tree vertex deduplication tolerance in PDF user units.",
    )
    min_stroke_width: float = Field(
        default=0.1,
        ge=0.0,
        description="Minimum stroke width to include (filters hairlines).",
    )
    wall_stroke_width_range: tuple[float, float] = Field(
        default=(0.5, 3.0),
        description="Stroke width range [min, max] for wall confidence heuristic.",
    )
    max_paths_per_page: int = Field(
        default=200_000,
        ge=1,
        description="Safety limit on number of paths to process per page.",
    )


# ---------------------------------------------------------------------------
# Stage 2: Tokenizer
# ---------------------------------------------------------------------------


class VisionBackbone(str, Enum):
    """Supported vision backbones for raster feature extraction."""

    HRNET_W32 = "hrnet_w32"
    HRNET_W48 = "hrnet_w48"
    SWIN_TINY = "swin_tiny_patch4_window7_224"
    SWIN_SMALL = "swin_small_patch4_window7_224"
    SWIN_BASE = "swin_base_patch4_window7_224"


class TokenizerConfig(BaseModel):
    """Configuration for cross-modal tokenization (Stage 2)."""

    d_model: int = Field(
        default=256,
        description="Token embedding dimension.",
    )
    n_heads: int = Field(
        default=8,
        ge=1,
        description="Number of attention heads in cross-modal fusion.",
    )
    vision_backbone: VisionBackbone = Field(
        default=VisionBackbone.HRNET_W48,
        description="Vision backbone for raster feature extraction.",
    )
    raster_dpi: int = Field(
        default=300,
        ge=72,
        le=600,
        description="DPI for PDF-to-raster rendering.",
    )
    attention_radius_fraction: float = Field(
        default=0.05,
        ge=0.01,
        le=0.5,
        description=(
            "Spatial attention window radius as fraction of page diagonal. "
            "Each token attends to visual features within this radius."
        ),
    )
    vector_only_fallback: bool = Field(
        default=True,
        description="Enable vector-only mode when raster is unavailable.",
    )
    dropout: float = Field(
        default=0.1,
        ge=0.0,
        le=0.5,
        description="Dropout rate in attention layers.",
    )


# ---------------------------------------------------------------------------
# Stage 3: Diffusion
# ---------------------------------------------------------------------------


class NoiseSchedule(str, Enum):
    """Noise schedule type for the forward diffusion process."""

    LINEAR = "linear"
    COSINE = "cosine"
    LEARNED = "learned"


class DiffusionConfig(BaseModel):
    """Configuration for the graph diffusion engine (Stage 3)."""

    d_model: int = Field(
        default=512,
        description="Transformer backbone embedding dimension.",
    )
    n_heads: int = Field(
        default=8,
        ge=1,
        description="Number of attention heads in transformer blocks.",
    )
    n_layers: int = Field(
        default=12,
        ge=1,
        description="Number of transformer blocks in the denoising network.",
    )
    timesteps_train: int = Field(
        default=1000,
        ge=100,
        description="Number of diffusion timesteps T for training.",
    )
    timesteps_inference: int = Field(
        default=50,
        ge=10,
        description="Number of DDIM sampling steps for inference.",
    )
    noise_schedule: NoiseSchedule = Field(
        default=NoiseSchedule.COSINE,
        description="Noise schedule type (Nichol & Dhariwal cosine recommended).",
    )
    max_nodes: int = Field(
        default=512,
        ge=16,
        description="Maximum number of junction nodes per graph.",
    )
    use_hdse: bool = Field(
        default=True,
        description="Enable Hierarchical Distance Structural Encoding.",
    )
    hdse_max_distance: int = Field(
        default=10,
        ge=1,
        description="Maximum shortest-path distance for HDSE encoding.",
    )
    dropout: float = Field(
        default=0.1,
        ge=0.0,
        le=0.5,
        description="Dropout rate in transformer blocks.",
    )


# ---------------------------------------------------------------------------
# Stage 4: Constraints
# ---------------------------------------------------------------------------


class ConstraintConfig(BaseModel):
    """Configuration for the NeSy SAT constraint solver (Stage 4)."""

    ortho_tolerance_deg: float = Field(
        default=5.0,
        ge=0.0,
        le=45.0,
        description="Angle tolerance (degrees) for orthogonal snap during inference.",
    )
    parallel_iqr_scale: float = Field(
        default=1.5,
        ge=0.0,
        description="IQR multiplier for parallel pair distance outlier detection.",
    )
    junction_min_degree: int = Field(
        default=2,
        ge=1,
        description="Minimum node degree enforced by junction closure axiom.",
    )
    snap_at_step: int = Field(
        default=5,
        ge=0,
        description=(
            "Apply hard geometric projection when denoising step <= this value. "
            "0 = snap only at final step."
        ),
    )
    learn_weights: bool = Field(
        default=True,
        description="Meta-learn axiom weights on validation set.",
    )
    initial_weights: dict[str, float] = Field(
        default={
            "orthogonal": 1.0,
            "parallel_pair": 1.0,
            "junction_closure": 0.5,
            "non_intersection": 2.0,
        },
        description="Initial axiom weights before meta-learning.",
    )


# ---------------------------------------------------------------------------
# Stage 5: Topology
# ---------------------------------------------------------------------------


class TopologyConfig(BaseModel):
    """Configuration for persistent homology loss (Stage 5)."""

    alpha: float = Field(
        default=1.0,
        ge=0.0,
        description="Weight for Wasserstein distance in TAFL.",
    )
    beta: float = Field(
        default=0.5,
        ge=0.0,
        description="Weight for Betti-0 error in TAFL.",
    )
    gamma: float = Field(
        default=0.5,
        ge=0.0,
        description="Weight for Betti-1 error in TAFL.",
    )
    sinkhorn_iterations: int = Field(
        default=100,
        ge=10,
        description="Maximum iterations for Sinkhorn-Knopp OT solver.",
    )
    sinkhorn_epsilon: float = Field(
        default=0.01,
        gt=0.0,
        description="Entropy regularization for Sinkhorn-Knopp.",
    )
    filtration_resolution: int = Field(
        default=256,
        ge=32,
        description="Grid resolution for cubical complex construction.",
    )


# ---------------------------------------------------------------------------
# Stage 6: Physics
# ---------------------------------------------------------------------------


class PhysicsConfig(BaseModel):
    """Configuration for PINN / FEA structural validation (Stage 6)."""

    lambda_bc: float = Field(
        default=1.0,
        ge=0.0,
        description="Weight for boundary condition loss.",
    )
    lambda_phys: float = Field(
        default=1.0,
        ge=0.0,
        description="Weight for PDE residual loss.",
    )
    live_load_psf: float = Field(
        default=40.0,
        ge=0.0,
        description="Live load in pounds per square foot (40 psf = residential).",
    )
    material_density: float = Field(
        default=2400.0,
        ge=0.0,
        description="Wall material density in kg/m³ (2400 = concrete).",
    )
    allowable_displacement_ratio: float = Field(
        default=360.0,
        ge=1.0,
        description="L/ratio for allowable displacement (360 = L/360 standard).",
    )
    pinn_hidden_layers: int = Field(
        default=4,
        ge=1,
        description="Number of hidden layers in PE-PINN.",
    )
    pinn_hidden_dim: int = Field(
        default=128,
        ge=16,
        description="Hidden dimension in PE-PINN layers.",
    )
    use_sin_activation: bool = Field(
        default=True,
        description="Use sin activations (True) vs ReLU (False) in PE-PINN.",
    )
    mesh_elements_per_wall: int = Field(
        default=4,
        ge=1,
        description="Number of FEA elements per wall segment.",
    )


# ---------------------------------------------------------------------------
# Stage 7: Serializer
# ---------------------------------------------------------------------------


class ExportFormat(str, Enum):
    """Supported output file formats."""

    JSON = "json"
    IFC_SPF = "ifc"
    GLTF = "gltf"


class SerializerConfig(BaseModel):
    """Configuration for IFC serialization (Stage 7)."""

    export_formats: list[ExportFormat] = Field(
        default=[ExportFormat.JSON, ExportFormat.IFC_SPF],
        description="Output formats to generate.",
    )
    default_wall_height_mm: float = Field(
        default=2700.0,
        ge=0.0,
        description="Default wall height for 3D extrusion in millimeters.",
    )
    ifc_schema: str = Field(
        default="IFC4",
        description="IFC schema version (IFC4 = ISO 16739-1:2024).",
    )
    coordinate_precision: int = Field(
        default=6,
        ge=1,
        le=15,
        description="Decimal precision for coordinate values in JSON output.",
    )
    compress_json: bool = Field(
        default=True,
        description="Use compressed JSON vocabulary for smaller output files.",
    )


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


class CurriculumPhase(BaseModel):
    """Configuration for a single curriculum learning phase."""

    start_epoch: int
    end_epoch: int
    lambda_diffusion: float = 1.0
    lambda_sat: float = 0.0
    lambda_topo: float = 0.0
    lambda_pde: float = 0.0
    lambda_reconstruction: float = 0.0


class TrainingConfig(BaseModel):
    """Configuration for the training pipeline (Stages 8-10)."""

    # Pre-training (MPM)
    mpm_mask_ratio: float = Field(
        default=0.80,
        ge=0.0,
        le=1.0,
        description="Fraction of tokens masked during MPM pre-training.",
    )
    mpm_epochs: int = Field(
        default=200,
        ge=1,
        description="Number of MPM pre-training epochs.",
    )

    # SFT
    sft_epochs: int = Field(
        default=150,
        ge=1,
        description="Number of supervised fine-tuning epochs.",
    )

    # GRPO
    grpo_epochs: int = Field(
        default=50,
        ge=1,
        description="Number of GRPO quality annealing epochs.",
    )

    # Optimization
    learning_rate: float = Field(
        default=1e-4,
        gt=0.0,
        description="Peak learning rate.",
    )
    weight_decay: float = Field(
        default=0.01,
        ge=0.0,
        description="AdamW weight decay.",
    )
    batch_size: int = Field(
        default=16,
        ge=1,
        description="Training batch size per GPU.",
    )
    grad_clip_norm: float = Field(
        default=1.0,
        gt=0.0,
        description="Gradient clipping max norm.",
    )

    # Curriculum schedule
    curriculum: list[CurriculumPhase] = Field(
        default=[
            CurriculumPhase(
                start_epoch=1,
                end_epoch=50,
                lambda_diffusion=1.0,
                lambda_sat=0.1,
                lambda_topo=0.1,
                lambda_pde=0.0,
            ),
            CurriculumPhase(
                start_epoch=51,
                end_epoch=150,
                lambda_diffusion=1.0,
                lambda_sat=1.0,
                lambda_topo=1.0,
                lambda_pde=0.5,
            ),
            CurriculumPhase(
                start_epoch=151,
                end_epoch=400,
                lambda_diffusion=1.0,
                lambda_sat=1.0,
                lambda_topo=2.0,
                lambda_pde=2.0,
            ),
        ],
        description="Curriculum loss weight schedule across training phases.",
    )

    # Infrastructure
    wandb_project: str = Field(
        default="axon",
        description="Weights & Biases project name.",
    )
    checkpoint_dir: Path = Field(
        default=Path("checkpoints"),
        description="Directory for model checkpoints.",
    )
    num_workers: int = Field(
        default=4,
        ge=0,
        description="DataLoader worker processes.",
    )
    seed: int = Field(
        default=42,
        description="Global random seed for reproducibility.",
    )


# ---------------------------------------------------------------------------
# Global Config
# ---------------------------------------------------------------------------


class AxonConfig(BaseModel):
    """Top-level configuration for the Axon pipeline.

    Aggregates all stage-specific configs into a single validated schema.
    Load from YAML/JSON or construct programmatically.

    Example:
        config = AxonConfig()
        config = AxonConfig.model_validate_json(Path("config.json").read_text())
        config = AxonConfig(diffusion=DiffusionConfig(n_layers=16))
    """

    parser: ParserConfig = Field(default_factory=ParserConfig)
    tokenizer: TokenizerConfig = Field(default_factory=TokenizerConfig)
    diffusion: DiffusionConfig = Field(default_factory=DiffusionConfig)
    constraints: ConstraintConfig = Field(default_factory=ConstraintConfig)
    topology: TopologyConfig = Field(default_factory=TopologyConfig)
    physics: PhysicsConfig = Field(default_factory=PhysicsConfig)
    serializer: SerializerConfig = Field(default_factory=SerializerConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)

    # Global settings
    device: str = Field(
        default="cuda",
        description="Compute device: 'cuda', 'cpu', or specific 'cuda:N'.",
    )
    dtype: str = Field(
        default="float32",
        description="Default tensor dtype: 'float32' or 'float16'.",
    )
    verbose: bool = Field(
        default=False,
        description="Enable verbose logging across all stages.",
    )
