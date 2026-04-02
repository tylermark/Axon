"""DRL Agent — Deep Reinforcement Learning for panelization and placement.

Provides a Gymnasium-compatible environment that processes a ClassifiedWallGraph
and uses the Knowledge Graph to panelize walls with CFS panels and place prefab
pods into rooms.

Modules:
    env: PanelizationEnv (Gymnasium.Env) — step/reset/observe/act
    state: State encoding — wall graph + assignments to observation tensors
    actions: Action space definitions — panel selection and pod placement
    reward: SPUR-based reward function with waste/violation penalties
    constraints: Opening and junction constraint handling (DRL-007/008)
    train: Training pipeline for MaskablePPO on synthetic floor plans (DRL-009)
"""

from src.drl.actions import (
    PanelAction,
    PlacementAction,
    decode_panel_action,
    decode_placement_action,
)
from src.drl.constraints import (
    JunctionInfo,
    WallSubSegment,
    compute_junction_map,
    compute_junction_penalties,
    compute_wall_sub_segments,
    get_corner_thickness_deduction,
)
from src.drl.env import PanelizationEnv
from src.drl.reward import compute_reward
from src.drl.state import encode_observation
from src.drl.train import (
    DRLTrainingConfig,
    EvalResult,
    RandomFloorPlanWrapper,
    evaluate_policy,
    generate_classified_graph,
    generate_synthetic_floor_plan,
    greedy_policy,
    make_env,
    train_drl,
)

__all__ = [  # noqa: RUF022
    "PanelizationEnv",
    "PanelAction",
    "PlacementAction",
    "compute_reward",
    "decode_panel_action",
    "decode_placement_action",
    "encode_observation",
    "WallSubSegment",
    "JunctionInfo",
    "compute_junction_map",
    "compute_junction_penalties",
    "compute_wall_sub_segments",
    "get_corner_thickness_deduction",
    "DRLTrainingConfig",
    "EvalResult",
    "RandomFloorPlanWrapper",
    "evaluate_policy",
    "generate_classified_graph",
    "generate_synthetic_floor_plan",
    "greedy_policy",
    "make_env",
    "train_drl",
]
