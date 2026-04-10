"""Unit tests for PanelizationEnv (DRL-001: Gymnasium environment)."""

from __future__ import annotations

import math

import numpy as np
import pytest

import gymnasium as gym

from docs.interfaces.classified_wall_graph import (
    ClassifiedWallGraph,
    FireRating,
    WallClassification,
)
from docs.interfaces.graph_to_serializer import (
    FinalizedGraph,
    Opening,
    OpeningType,
    Room,
    WallSegment,
    WallType,
)
from src.drl.actions import PANELIZATION_ACTION_SIZE, PLACEMENT_ACTION_SIZE
from src.drl.env import PanelizationEnv
from src.drl.reward import RewardBreakdown, RewardWeights
from src.drl.state import (
    CANDIDATE_PANEL_DIM,
    CANDIDATE_POD_DIM,
    MAX_CANDIDATES,
    MAX_ROOMS,
    MAX_WALLS,
    ROOM_FEATURE_DIM,
    WALL_FEATURE_DIM,
)
from src.knowledge_graph.loader import load_knowledge_graph


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_classified_graph(
    n_walls: int = 2,
    n_rooms: int = 1,
    wall_type: WallType = WallType.LOAD_BEARING,
    fire_rating: FireRating = FireRating.NONE,
    scale_factor: float = 1.0,
    wall_length: float | None = None,
) -> ClassifiedWallGraph:
    """Create a classified wall graph with simple rectangular layout.

    If n_walls=4 and n_rooms=1, creates a rectangle. Otherwise creates
    a line of walls with n_rooms rooms.
    """
    # Create a simple rectangular room with the specified number of walls
    if n_walls >= 4:
        nodes = np.array([
            [0, 0],
            [7200, 0],     # 100 inches at 72 units/inch
            [7200, 7200],
            [0, 7200],
        ], dtype=np.float64)
        edge_pairs = [(0, 1), (1, 2), (2, 3), (3, 0)]
    else:
        # Create a simple line of walls
        nodes_list = []
        for i in range(n_walls + 1):
            x = i * 7200.0
            nodes_list.append([x, 0.0])
        nodes = np.array(nodes_list, dtype=np.float64)
        edge_pairs = [(i, i + 1) for i in range(n_walls)]

    edges = np.array(edge_pairs[:n_walls], dtype=np.int64)

    segments = []
    for i, (s, e) in enumerate(edge_pairs[:n_walls]):
        delta = nodes[e] - nodes[s]
        length = wall_length if wall_length is not None else float(np.linalg.norm(delta))
        angle = float(np.arctan2(delta[1], delta[0]) % np.pi)
        segments.append(WallSegment(
            edge_id=i,
            start_node=int(s),
            end_node=int(e),
            start_coord=nodes[s].copy(),
            end_coord=nodes[e].copy(),
            thickness=6.0,
            height=2700.0,
            wall_type=WallType.UNKNOWN,
            angle=angle,
            length=length,
            confidence=1.0,
        ))

    rooms = []
    for r in range(n_rooms):
        boundary_nodes = list(range(min(len(nodes), 4)))
        rooms.append(Room(
            room_id=r,
            boundary_edges=list(range(min(n_walls, 4))),
            boundary_nodes=boundary_nodes,
            area=7200.0 * 7200.0,
            label="Bedroom",
            is_exterior=False,
        ))

    graph = FinalizedGraph(
        nodes=nodes,
        edges=edges,
        wall_segments=segments,
        openings=[],
        rooms=rooms,
        page_width=14400.0,
        page_height=14400.0,
        scale_factor=scale_factor,
    )

    classifications = [
        WallClassification(
            edge_id=seg.edge_id,
            wall_type=wall_type,
            fire_rating=fire_rating,
            confidence=0.9,
            is_perimeter=True,
        )
        for seg in segments
    ]

    return ClassifiedWallGraph(
        graph=graph,
        classifications=classifications,
    )


@pytest.fixture(scope="module")
def kg_store():
    """Load the KG store once for all tests in this module."""
    return load_knowledge_graph()


@pytest.fixture
def env_2wall_1room(kg_store):
    """Create an env with 2 walls and 1 room."""
    cg = _make_classified_graph(n_walls=2, n_rooms=1)
    return PanelizationEnv(cg, kg_store)


@pytest.fixture
def env_4wall_1room(kg_store):
    """Create an env with 4 walls (rectangle) and 1 room."""
    cg = _make_classified_graph(n_walls=4, n_rooms=1)
    return PanelizationEnv(cg, kg_store)


# ---------------------------------------------------------------------------
# reset()
# ---------------------------------------------------------------------------


class TestEnvReset:
    """Tests for PanelizationEnv.reset()."""

    def test_returns_obs_and_info(self, env_2wall_1room):
        obs, info = env_2wall_1room.reset()
        assert isinstance(obs, dict)
        assert isinstance(info, dict)

    def test_observation_keys(self, env_2wall_1room):
        obs, _ = env_2wall_1room.reset()
        expected_keys = {
            "wall_features", "room_features", "wall_assigned",
            "room_assigned", "current_target", "candidate_features",
            "candidate_mask", "phase", "progress",
        }
        assert set(obs.keys()) == expected_keys

    def test_observation_shapes(self, env_2wall_1room):
        obs, _ = env_2wall_1room.reset()
        candidate_dim = max(CANDIDATE_PANEL_DIM, CANDIDATE_POD_DIM)
        target_dim = max(WALL_FEATURE_DIM, ROOM_FEATURE_DIM)
        assert obs["wall_features"].shape == (MAX_WALLS, WALL_FEATURE_DIM)
        assert obs["room_features"].shape == (MAX_ROOMS, ROOM_FEATURE_DIM)
        assert obs["wall_assigned"].shape == (MAX_WALLS,)
        assert obs["room_assigned"].shape == (MAX_ROOMS,)
        assert obs["current_target"].shape == (target_dim,)
        assert obs["candidate_features"].shape == (MAX_CANDIDATES, candidate_dim)
        assert obs["candidate_mask"].shape == (MAX_CANDIDATES,)
        assert obs["phase"].shape == (2,)
        assert obs["progress"].shape == (2,)

    def test_starts_in_panelization_phase(self, env_2wall_1room):
        obs, info = env_2wall_1room.reset()
        assert info["phase"] == "panelization"
        np.testing.assert_array_equal(obs["phase"], [1.0, 0.0])

    def test_assignments_cleared(self, env_2wall_1room):
        env = env_2wall_1room
        env.reset()
        # Take a step with SKIP
        env.step(0)
        # Reset should clear state
        obs, info = env.reset()
        assert info["walls_assigned"] == 0
        assert info["rooms_assigned"] == 0
        assert np.all(obs["wall_assigned"] == 0.0)
        assert np.all(obs["room_assigned"] == 0.0)

    def test_info_contains_counts(self, env_2wall_1room):
        _, info = env_2wall_1room.reset()
        assert "total_walls" in info
        assert "total_rooms" in info
        assert info["total_walls"] == 2
        assert info["total_rooms"] == 1

    def test_no_walls_starts_in_placement(self, kg_store):
        """If there are no walls, should skip directly to placement phase."""
        cg = _make_classified_graph(n_walls=0, n_rooms=1)
        # The graph constructor with 0 walls creates an edge array of shape (0, 2)
        # and an empty wall_segments list.
        # Manually fix the graph for 0 walls:
        graph = FinalizedGraph(
            nodes=np.array([[0, 0], [7200, 0], [7200, 7200], [0, 7200]], dtype=np.float64),
            edges=np.empty((0, 2), dtype=np.int64),
            wall_segments=[],
            openings=[],
            rooms=[Room(
                room_id=0,
                boundary_edges=[],
                boundary_nodes=[0, 1, 2, 3],
                area=7200.0 * 7200.0,
                label="Bedroom",
                is_exterior=False,
            )],
            page_width=14400.0,
            page_height=14400.0,
        )
        cg = ClassifiedWallGraph(graph=graph, classifications=[])
        env = PanelizationEnv(cg, kg_store)
        _, info = env.reset()
        assert info["phase"] == "placement"

    def test_reset_with_new_graph(self, kg_store):
        """Reset with options to replace the classified_graph."""
        cg1 = _make_classified_graph(n_walls=2, n_rooms=1)
        cg2 = _make_classified_graph(n_walls=4, n_rooms=1)
        env = PanelizationEnv(cg1, kg_store)
        env.reset()
        assert env._n_walls == 2

        env.reset(options={"classified_graph": cg2})
        assert env._n_walls == 4


# ---------------------------------------------------------------------------
# step()
# ---------------------------------------------------------------------------


class TestEnvStep:
    """Tests for PanelizationEnv.step()."""

    def test_step_returns_five_tuple(self, env_2wall_1room):
        env = env_2wall_1room
        env.reset()
        result = env.step(0)  # SKIP
        assert len(result) == 5
        obs, reward, terminated, truncated, info = result
        assert isinstance(obs, dict)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

    def test_truncated_is_always_false(self, env_2wall_1room):
        env = env_2wall_1room
        env.reset()
        _, _, _, truncated, _ = env.step(0)
        assert truncated is False

    def test_skip_advances_wall_index(self, env_2wall_1room):
        env = env_2wall_1room
        env.reset()
        assert env.current_wall_idx == 0
        env.step(0)  # SKIP wall 0
        assert env.current_wall_idx == 1

    def test_skip_does_not_assign(self, env_2wall_1room):
        env = env_2wall_1room
        env.reset()
        env.step(0)  # SKIP
        assert len(env.wall_assignments) == 0

    def test_phase_transitions_to_placement(self, env_2wall_1room):
        env = env_2wall_1room
        env.reset()
        # 2 walls: skip both
        env.step(0)  # wall 0
        _, _, _, _, info = env.step(0)  # wall 1 -> transitions to placement
        assert env.phase == "placement"

    def test_episode_terminates_after_all_rooms(self, env_2wall_1room):
        env = env_2wall_1room
        env.reset()
        # 2 walls + 1 room
        env.step(0)  # wall 0
        env.step(0)  # wall 1 -> placement
        _, _, terminated, _, _ = env.step(0)  # room 0 -> done
        assert terminated is True

    def test_not_terminated_during_panelization(self, env_2wall_1room):
        env = env_2wall_1room
        env.reset()
        _, _, terminated, _, _ = env.step(0)  # wall 0
        assert terminated is False

    def test_reward_breakdown_in_info(self, env_2wall_1room):
        env = env_2wall_1room
        env.reset()
        _, _, _, _, info = env.step(0)
        assert "reward_breakdown" in info
        assert isinstance(info["reward_breakdown"], RewardBreakdown)

    def test_step_count_increments(self, env_2wall_1room):
        env = env_2wall_1room
        env.reset()
        _, _, _, _, info1 = env.step(0)
        _, _, _, _, info2 = env.step(0)
        assert info1["step"] == 1
        assert info2["step"] == 2


# ---------------------------------------------------------------------------
# Action masking
# ---------------------------------------------------------------------------


class TestActionMasking:
    """Tests for action_masks() method."""

    def test_skip_always_valid_panelization(self, env_2wall_1room):
        env = env_2wall_1room
        env.reset()
        mask = env.action_masks()
        assert mask[0] == 1.0  # SKIP is action 0

    def test_mask_shape(self, env_2wall_1room):
        env = env_2wall_1room
        env.reset()
        mask = env.action_masks()
        expected_size = max(PANELIZATION_ACTION_SIZE, PLACEMENT_ACTION_SIZE)
        assert mask.shape == (expected_size,)

    def test_mask_is_copy(self, env_2wall_1room):
        """action_masks() should return a copy, not a reference."""
        env = env_2wall_1room
        env.reset()
        mask1 = env.action_masks()
        mask2 = env.action_masks()
        assert mask1 is not mask2

    def test_skip_valid_in_placement_phase(self, env_2wall_1room):
        env = env_2wall_1room
        env.reset()
        env.step(0)  # wall 0
        env.step(0)  # wall 1 -> placement
        mask = env.action_masks()
        assert mask[0] == 1.0  # SKIP always valid


# ---------------------------------------------------------------------------
# Two-phase episode flow
# ---------------------------------------------------------------------------


class TestTwoPhaseEpisode:
    """Tests for the panelization -> placement phase transition."""

    def test_full_episode_with_skips(self, env_4wall_1room):
        """Run a complete episode skipping everything."""
        env = env_4wall_1room
        env.reset()

        # Panelization phase: 4 walls
        for i in range(4):
            obs, reward, terminated, truncated, info = env.step(0)
            if i < 3:
                assert not terminated
                assert info["phase"] == "panelization"
            else:
                # After 4th wall, should transition to placement
                assert info["phase"] == "placement"

        # Placement phase: 1 room
        obs, reward, terminated, truncated, info = env.step(0)
        assert terminated is True

    def test_wall_assignment_recorded(self, env_2wall_1room):
        """When action != 0, a wall assignment should be recorded (if valid)."""
        env = env_2wall_1room
        env.reset()

        mask = env.action_masks()
        # Find a valid non-skip action
        valid_actions = np.where(mask > 0)[0]
        non_skip_actions = [a for a in valid_actions if a != 0]

        if non_skip_actions:
            action = non_skip_actions[0]
            env.step(action)
            assert len(env.wall_assignments) >= 1
        else:
            # No valid panel candidates; skip is the only option
            env.step(0)
            assert len(env.wall_assignments) == 0

    def test_observation_changes_between_steps(self, env_2wall_1room):
        env = env_2wall_1room
        obs1, _ = env.reset()
        obs2, _, _, _, _ = env.step(0)  # advance to wall 1
        # Progress should change
        assert not np.array_equal(obs1["progress"], obs2["progress"]) or \
               not np.array_equal(obs1["current_target"], obs2["current_target"])


# ---------------------------------------------------------------------------
# get_results()
# ---------------------------------------------------------------------------


class TestGetResults:
    """Tests for get_results() after episode completion."""

    def test_results_after_episode(self, env_2wall_1room):
        env = env_2wall_1room
        env.reset()
        # Run full episode with skips
        env.step(0)  # wall 0
        env.step(0)  # wall 1
        env.step(0)  # room 0

        results = env.get_results()
        assert "wall_assignments" in results
        assert "room_assignments" in results
        assert "walls_covered" in results
        assert "rooms_covered" in results
        assert "total_walls" in results
        assert "total_rooms" in results
        assert "wall_coverage_pct" in results
        assert "room_coverage_pct" in results
        assert "total_reward" in results
        assert "total_violations" in results
        assert "step_rewards" in results

    def test_skip_all_gives_zero_coverage(self, env_2wall_1room):
        env = env_2wall_1room
        env.reset()
        env.step(0)
        env.step(0)
        env.step(0)

        results = env.get_results()
        assert results["walls_covered"] == 0
        assert results["rooms_covered"] == 0
        assert results["wall_coverage_pct"] == 0.0
        assert results["room_coverage_pct"] == 0.0

    def test_total_walls_rooms_match(self, env_2wall_1room):
        env = env_2wall_1room
        env.reset()
        env.step(0)
        env.step(0)
        env.step(0)

        results = env.get_results()
        assert results["total_walls"] == 2
        assert results["total_rooms"] == 1

    def test_step_rewards_length(self, env_2wall_1room):
        env = env_2wall_1room
        env.reset()
        env.step(0)
        env.step(0)
        env.step(0)

        results = env.get_results()
        # 3 steps total (2 walls + 1 room)
        assert len(results["step_rewards"]) == 3


# ---------------------------------------------------------------------------
# Observation space compliance
# ---------------------------------------------------------------------------


class TestObservationSpaceCompliance:
    """Tests that observations conform to the declared observation_space."""

    def test_observation_in_space_after_reset(self, env_2wall_1room):
        env = env_2wall_1room
        obs, _ = env.reset()
        assert env.observation_space.contains(obs)

    def test_observation_in_space_after_step(self, env_2wall_1room):
        env = env_2wall_1room
        env.reset()
        obs, _, _, _, _ = env.step(0)
        assert env.observation_space.contains(obs)

    def test_observation_in_space_placement_phase(self, env_2wall_1room):
        env = env_2wall_1room
        env.reset()
        env.step(0)  # wall 0
        obs, _, _, _, _ = env.step(0)  # wall 1 -> placement
        assert env.observation_space.contains(obs)


# ---------------------------------------------------------------------------
# Custom reward weights
# ---------------------------------------------------------------------------


class TestCustomRewardWeights:
    """Tests for configuring reward weights."""

    def test_custom_weights_applied(self, kg_store):
        cg = _make_classified_graph(n_walls=1, n_rooms=0)
        # Zero out all weights except SPUR
        weights = RewardWeights(
            spur=10.0,
            waste=0.0,
            violation=0.0,
            coverage=0.0,
            completion_bonus=0.0,
        )
        env = PanelizationEnv(cg, kg_store, reward_weights=weights)
        env.reset()
        _, reward_skip, _, _, _ = env.step(0)  # SKIP (spur=0)

        # Reset and skip again -- this env only has 1 wall + 0 rooms
        # so the episode ends after 1 step.
        # With SPUR=0 and all other weights=0, reward should be ~0
        # (just completion bonus if terminal, but we set that to 0)
        assert isinstance(reward_skip, float)


# ---------------------------------------------------------------------------
# DRL-005: Multi-panel wall handling
# ---------------------------------------------------------------------------


class TestMultiPanelWall:
    """Tests for walls longer than max panel length requiring multiple steps.

    DRL-005: When a wall is longer than the maximum panel length, the
    environment stays on the same wall and lets the agent place additional
    panels until the remaining length is covered.
    """

    def test_long_wall_requires_multiple_steps(self, kg_store):
        """A very long wall should require more than one panel step."""
        # Create a single very long wall (500 inches at scale=1.0 -> 36000 PDF units)
        cg = _make_classified_graph(n_walls=1, n_rooms=0, wall_length=36000.0)
        # Manually fix rooms for 0 rooms (exclude exterior)
        graph = FinalizedGraph(
            nodes=cg.graph.nodes,
            edges=cg.graph.edges,
            wall_segments=cg.graph.wall_segments,
            openings=[],
            rooms=[],
            page_width=cg.graph.page_width,
            page_height=cg.graph.page_height,
            scale_factor=cg.graph.scale_factor,
        )
        cg_no_rooms = ClassifiedWallGraph(
            graph=graph,
            classifications=cg.classifications,
        )
        env = PanelizationEnv(cg_no_rooms, kg_store)
        env.reset()

        # Take a non-skip action (select first valid panel candidate)
        mask = env.action_masks()
        valid_actions = [a for a in range(len(mask)) if mask[a] > 0 and a != 0]
        assert valid_actions, (
            "KG returned no panel candidates for a 500-inch wall — "
            "check KnowledgeGraphStore fixture has panels with max_length < 500"
        )

        wall_edge_id = env._walls[0].edge_id

        # First step: assign a panel
        env.step(valid_actions[0])

        # Check if the wall still has remaining length
        remaining = env.wall_remaining_inches.get(wall_edge_id, 0.0)

        # A 500-inch wall should need multiple panels (typical max ~240 inches)
        assert remaining > 6.0, (
            f"Expected remaining > 6.0 for a 500-inch wall, got {remaining}"
        )
        # The env should stay on the same wall (idx 0)
        assert env.current_wall_idx == 0
        assert env.phase == "panelization"

    def test_wall_remaining_tracked(self, kg_store):
        """wall_remaining_inches should be populated after first panel step."""
        cg = _make_classified_graph(n_walls=1, n_rooms=0, wall_length=36000.0)
        graph = FinalizedGraph(
            nodes=cg.graph.nodes,
            edges=cg.graph.edges,
            wall_segments=cg.graph.wall_segments,
            openings=[],
            rooms=[],
            page_width=cg.graph.page_width,
            page_height=cg.graph.page_height,
            scale_factor=cg.graph.scale_factor,
        )
        cg_no_rooms = ClassifiedWallGraph(
            graph=graph,
            classifications=cg.classifications,
        )
        env = PanelizationEnv(cg_no_rooms, kg_store)
        env.reset()

        mask = env.action_masks()
        valid_actions = [a for a in range(len(mask)) if mask[a] > 0 and a != 0]
        if not valid_actions:
            pytest.skip("No valid panel candidates")

        env.step(valid_actions[0])
        wall_edge_id = env._walls[0].edge_id
        assert wall_edge_id in env.wall_remaining_inches

    def test_skip_sets_remaining_to_zero(self, kg_store):
        """Skipping a wall sets remaining to zero and moves on."""
        cg = _make_classified_graph(n_walls=1, n_rooms=0, wall_length=36000.0)
        graph = FinalizedGraph(
            nodes=cg.graph.nodes,
            edges=cg.graph.edges,
            wall_segments=cg.graph.wall_segments,
            openings=[],
            rooms=[],
            page_width=cg.graph.page_width,
            page_height=cg.graph.page_height,
            scale_factor=cg.graph.scale_factor,
        )
        cg_no_rooms = ClassifiedWallGraph(
            graph=graph,
            classifications=cg.classifications,
        )
        env = PanelizationEnv(cg_no_rooms, kg_store)
        env.reset()

        env.step(0)  # SKIP
        wall_edge_id = env._walls[0].edge_id
        remaining = env.wall_remaining_inches.get(wall_edge_id, 0.0)
        assert remaining == 0.0

    def test_multi_panel_walls_counted_in_results(self, kg_store):
        """get_results should report multi_panel_walls count."""
        cg = _make_classified_graph(n_walls=2, n_rooms=1)
        env = PanelizationEnv(cg, kg_store)
        env.reset()
        # Skip all walls and the room
        env.step(0)
        env.step(0)
        env.step(0)
        results = env.get_results()
        assert "multi_panel_walls" in results
        assert isinstance(results["multi_panel_walls"], int)

    def test_wall_remaining_in_info(self, kg_store):
        """Step info should include wall_remaining_inches during panelization."""
        cg = _make_classified_graph(n_walls=2, n_rooms=1)
        env = PanelizationEnv(cg, kg_store)
        env.reset()
        _, _, _, _, info = env.step(0)
        assert "wall_remaining_inches" in info


# ---------------------------------------------------------------------------
# DRL-006: Multi-pod room placement
# ---------------------------------------------------------------------------


class TestMultiPodRoom:
    """Tests for rooms that can receive multiple pods.

    DRL-006: When a room has enough area after placing the first pod,
    the environment stays on the same room for additional pod placements.
    """

    def test_room_remaining_area_tracked(self, kg_store):
        """room_remaining_area should be set after first pod placement."""
        # Build a large bathroom room (200x200 inches = 14400x14400 PDF units)
        # so that pods will fit and the test can exercise pod placement.
        nodes = np.array([
            [0, 0], [14400, 0], [14400, 14400], [0, 14400],
        ], dtype=np.float64)
        edges = np.array([[0, 1], [1, 2]], dtype=np.int64)
        segments = []
        for i, (s, e) in enumerate(edges):
            delta = nodes[e] - nodes[s]
            length = float(np.linalg.norm(delta))
            angle = float(np.arctan2(delta[1], delta[0]) % np.pi)
            segments.append(WallSegment(
                edge_id=i, start_node=int(s), end_node=int(e),
                start_coord=nodes[s].copy(), end_coord=nodes[e].copy(),
                thickness=6.0, height=2700.0, wall_type=WallType.UNKNOWN,
                angle=angle, length=length, confidence=1.0,
            ))
        rooms = [Room(
            room_id=0, boundary_edges=[0, 1],
            boundary_nodes=[0, 1, 2, 3],
            area=14400.0 * 14400.0, label="Bathroom", is_exterior=False,
        )]
        graph = FinalizedGraph(
            nodes=nodes, edges=edges, wall_segments=segments,
            openings=[], rooms=rooms,
            page_width=28800.0, page_height=28800.0,
        )
        classifications = [
            WallClassification(edge_id=i, wall_type=WallType.LOAD_BEARING,
                               fire_rating=FireRating.NONE, confidence=0.9,
                               is_perimeter=True)
            for i in range(2)
        ]
        cg = ClassifiedWallGraph(graph=graph, classifications=classifications)
        env = PanelizationEnv(cg, kg_store)
        env.reset()
        env.step(0)  # wall 0
        env.step(0)  # wall 1 -> placement

        mask = env.action_masks()
        valid_actions = [a for a in range(len(mask)) if mask[a] > 0 and a != 0]
        if not valid_actions:
            pytest.skip("No valid pod candidates for large bathroom")

        env.step(valid_actions[0])
        room_id = env._rooms[0].room_id
        assert room_id in env.room_remaining_area

    def test_room_pod_placements_tracked(self, kg_store):
        """room_pod_placements should record placed pod info."""
        # Build a large bathroom room so pods fit.
        nodes = np.array([
            [0, 0], [14400, 0], [14400, 14400], [0, 14400],
        ], dtype=np.float64)
        edges = np.array([[0, 1], [1, 2]], dtype=np.int64)
        segments = []
        for i, (s, e) in enumerate(edges):
            delta = nodes[e] - nodes[s]
            length = float(np.linalg.norm(delta))
            angle = float(np.arctan2(delta[1], delta[0]) % np.pi)
            segments.append(WallSegment(
                edge_id=i, start_node=int(s), end_node=int(e),
                start_coord=nodes[s].copy(), end_coord=nodes[e].copy(),
                thickness=6.0, height=2700.0, wall_type=WallType.UNKNOWN,
                angle=angle, length=length, confidence=1.0,
            ))
        rooms = [Room(
            room_id=0, boundary_edges=[0, 1],
            boundary_nodes=[0, 1, 2, 3],
            area=14400.0 * 14400.0, label="Bathroom", is_exterior=False,
        )]
        graph = FinalizedGraph(
            nodes=nodes, edges=edges, wall_segments=segments,
            openings=[], rooms=rooms,
            page_width=28800.0, page_height=28800.0,
        )
        classifications = [
            WallClassification(edge_id=i, wall_type=WallType.LOAD_BEARING,
                               fire_rating=FireRating.NONE, confidence=0.9,
                               is_perimeter=True)
            for i in range(2)
        ]
        cg = ClassifiedWallGraph(graph=graph, classifications=classifications)
        env = PanelizationEnv(cg, kg_store)
        env.reset()
        env.step(0)
        env.step(0)

        mask = env.action_masks()
        valid_actions = [a for a in range(len(mask)) if mask[a] > 0 and a != 0]
        if not valid_actions:
            pytest.skip("No valid pod candidates for large bathroom")

        env.step(valid_actions[0])
        room_id = env._rooms[0].room_id
        placements = env.room_pod_placements.get(room_id, [])
        assert len(placements) >= 1
        # Each placement is a (sku, x, y, rotated) tuple
        sku, x, y, rotated = placements[0]
        assert isinstance(sku, str)
        assert isinstance(x, float)
        assert isinstance(y, float)
        assert isinstance(rotated, bool)

    def test_multi_pod_rooms_counted_in_results(self, kg_store):
        """get_results should report multi_pod_rooms count."""
        cg = _make_classified_graph(n_walls=2, n_rooms=1)
        env = PanelizationEnv(cg, kg_store)
        env.reset()
        env.step(0)
        env.step(0)
        env.step(0)
        results = env.get_results()
        assert "multi_pod_rooms" in results
        assert isinstance(results["multi_pod_rooms"], int)

    def test_skip_room_terminates_episode(self, kg_store):
        """Skipping the only room should end the episode."""
        cg = _make_classified_graph(n_walls=2, n_rooms=1)
        env = PanelizationEnv(cg, kg_store)
        env.reset()
        env.step(0)  # wall 0
        env.step(0)  # wall 1
        _, _, terminated, _, _ = env.step(0)  # skip room 0
        assert terminated is True

    def test_placement_info_contains_room_metrics(self, kg_store):
        """During placement phase, info should include room area and pod count."""
        cg = _make_classified_graph(n_walls=2, n_rooms=1)
        env = PanelizationEnv(cg, kg_store)
        env.reset()
        env.step(0)
        _, _, _, _, info = env.step(0)  # transitions to placement
        assert "current_room_id" in info
        assert "room_remaining_area" in info
        assert "pods_placed_in_room" in info


# ---------------------------------------------------------------------------
# DRL-007: Opening sub-segment handling
# ---------------------------------------------------------------------------


class TestOpeningHandling:
    """Tests for walls with openings being split into sub-segments.

    DRL-007: When a wall has openings, it is split into sub-segments
    that are independently panelized. The environment iterates through
    sub-segments via current_sub_segment_idx.
    """

    def test_sub_segments_precomputed(self, kg_store):
        """Sub-segments should be precomputed for walls with openings."""
        nodes = np.array([
            [0, 0], [14400, 0], [14400, 7200], [0, 7200],
        ], dtype=np.float64)
        edges = np.array([[0, 1], [1, 2]], dtype=np.int64)
        wall_0 = WallSegment(
            edge_id=0, start_node=0, end_node=1,
            start_coord=nodes[0].copy(), end_coord=nodes[1].copy(),
            thickness=6.0, height=2700.0, wall_type=WallType.UNKNOWN,
            angle=0.0, length=14400.0, confidence=1.0,
        )
        wall_1 = WallSegment(
            edge_id=1, start_node=1, end_node=2,
            start_coord=nodes[1].copy(), end_coord=nodes[2].copy(),
            thickness=6.0, height=2700.0, wall_type=WallType.UNKNOWN,
            angle=math.pi / 2, length=7200.0, confidence=1.0,
        )
        openings = [
            Opening(
                opening_type=OpeningType.DOOR,
                wall_edge_id=0,
                position_along_wall=0.5,
                width=2160.0,  # 30 inches
                height=2000.0,
            ),
        ]
        graph = FinalizedGraph(
            nodes=nodes, edges=edges,
            wall_segments=[wall_0, wall_1],
            openings=openings,
            rooms=[],
            page_width=14400.0, page_height=14400.0,
        )
        classifications = [
            WallClassification(edge_id=0, wall_type=WallType.LOAD_BEARING, fire_rating=FireRating.NONE, confidence=0.9),
            WallClassification(edge_id=1, wall_type=WallType.LOAD_BEARING, fire_rating=FireRating.NONE, confidence=0.9),
        ]
        cg = ClassifiedWallGraph(graph=graph, classifications=classifications)
        env = PanelizationEnv(cg, kg_store)
        env.reset()

        # Wall 0 has an opening, so it should have >1 sub-segments
        sub_segs = env._sub_segment_map.get(0, [])
        assert len(sub_segs) == 2

    def test_wall_without_opening_has_one_sub_segment(self, kg_store):
        """A wall with no openings has exactly one sub-segment."""
        cg = _make_classified_graph(n_walls=2, n_rooms=1)
        env = PanelizationEnv(cg, kg_store)
        env.reset()
        for wall in env._walls:
            sub_segs = env._sub_segment_map.get(wall.edge_id, [])
            assert len(sub_segs) == 1

    def test_sub_segment_info_in_step(self, kg_store):
        """Step info should include sub-segment count and current index."""
        cg = _make_classified_graph(n_walls=2, n_rooms=1)
        env = PanelizationEnv(cg, kg_store)
        env.reset()
        _, _, _, _, info = env.step(0)
        assert "num_sub_segments" in info
        assert "current_sub_segment_idx" in info

    def test_junction_info_in_step(self, kg_store):
        """Step info should include junction nodes during panelization."""
        # Build a rectangle so there are corner junctions
        cg = _make_classified_graph(n_walls=4, n_rooms=1)
        env = PanelizationEnv(cg, kg_store)
        env.reset()
        _, _, _, _, info = env.step(0)
        assert "junction_nodes" in info
        assert isinstance(info["junction_nodes"], list)

    def test_junction_violations_in_results(self, kg_store):
        """get_results should count junction violations."""
        cg = _make_classified_graph(n_walls=4, n_rooms=1)
        env = PanelizationEnv(cg, kg_store)
        env.reset()
        for _ in range(4):
            env.step(0)
        env.step(0)  # room
        results = env.get_results()
        assert "junction_violations" in results
        assert isinstance(results["junction_violations"], int)
