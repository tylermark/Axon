"""Training pipeline — SSL pre-training, SFT, GRPO, DRL training, and tracking.

Provides the data engine, MPM pre-trainer, SFT/GRPO fine-tuners, DRL
training pipeline wrapper (TR-005), and unified experiment tracking +
checkpoint management (TR-006).
"""

from src.training.data_engine import (
    ArchCAD400KDataset,
    CombinedFloorPlanDataset,
    DatasetSpec,
    FloorPlanCADDataset,
    MLStructDataset,
    ResPlanDataset,
    build_combined_dataset,
)
from src.training.drl_training import DRLTrainingConfig, DRLTrainingPipeline
from src.training.grpo import (
    GRPOConfig,
    GRPOTrainer,
    compute_composite_reward,
)
from src.training.mpm import (
    MPMConfig,
    MPMModel,
    MPMPreTrainer,
    chamfer_distance,
)
from src.training.sft import (
    SFTConfig,
    SFTTrainer,
    WarmupCosineScheduler,
)
from src.training.tracking import CheckpointManager, ExperimentTracker

__all__ = [
    "ArchCAD400KDataset",
    "CheckpointManager",
    "CombinedFloorPlanDataset",
    "DRLTrainingConfig",
    "DRLTrainingPipeline",
    "DatasetSpec",
    "ExperimentTracker",
    "FloorPlanCADDataset",
    "GRPOConfig",
    "GRPOTrainer",
    "MLStructDataset",
    "MPMConfig",
    "MPMModel",
    "MPMPreTrainer",
    "ResPlanDataset",
    "SFTConfig",
    "SFTTrainer",
    "WarmupCosineScheduler",
    "build_combined_dataset",
    "chamfer_distance",
    "compute_composite_reward",
]
