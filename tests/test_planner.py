"""Planner checks replay native actions and observe the original piece's lock."""

import copy
import json

import gymnasium as gym
import numpy as np
import pytest
from tetris_gymnasium.envs.tetris import Tetris
from tetris_gymnasium.mappings.actions import ActionsMapping

from tetris_agent import (
    PlannerStatus, TacticalPlanner, TargetPlacement, TargetSpace, TetrisAdapter,
    target_to_env_origin,
)


@pytest.fixture
def env():
    game = Tetris(gravity=True)
    game.reset(seed=42)
    yield game
    game.close()


def select_piece(env, index, orientation=0):
    env.active_tetromino = env.tetrominoes[index]
    for _ in range(orientation):
        env.active_tetromino = env.rotate(env.active_tetromino)
    env.reset_tetromino_position()


def piece_signature(piece):
    if piece is None:
        return None
    return piece.id, tuple(piece.color_rgb), piece.matrix.shape, piece.matrix.tobytes()


def state_signature(env):
    """All TetrisState fields plus base Gym RNG/configuration, by value."""
    state = env  # Inspect live objects: get_state() itself drops deque maxlen.
    randomizer = state.randomizer
    return (
        state.board.shape, state.board.dtype.str, state.board.tobytes(),
        piece_signature(state.active_tetromino), state.x, state.y,
        state.queue.size, tuple(state.queue.get_queue()),
        state.queue.queue.maxlen,
        state.queue.randomizer is state.randomizer,
        state.holder.size, tuple(map(piece_signature, state.holder.get_tetrominoes())),
        state.holder.queue.maxlen,
        type(randomizer), randomizer.size,
        json.dumps(randomizer.rng.bit_generator.state, sort_keys=True),
        getattr(randomizer, "index", None),
        getattr(randomizer, "bag", np.array([])).tobytes(),
        state.has_swapped, state.game_over, state.score,
        "score" in vars(env),
        "place_active_tetromino" in vars(env), vars(env).get("place_active_tetromino"),
        json.dumps(env.np_random.bit_generator.state, sort_keys=True),
        env.gravity_enabled, tuple(map(piece_signature, env.tetrominoes)),
    )


def assert_replay(env, target, result):
    """Independent oracle: record cells at native placement, before row clearing."""
    assert result.status == PlannerStatus.REACHABLE and result.reachable
    assert result.path_length == len(result.actions)
    assert result.elapsed_seconds >= 0
    allowed = {env.actions.move_left, env.actions.move_right, env.actions.move_down,
               env.actions.rotate_clockwise, env.actions.rotate_counterclockwise,
               env.actions.no_op}
    assert set(result.actions) <= allowed
    initial = TetrisAdapter(env).read_current_piece()
    expected_matrix = np.rot90(initial.spawn_matrix, target.orientation)
    erows, ecols = np.nonzero(expected_matrix > 0)
    expected_cells = set(zip((erows - erows.min() + target.y).tolist(),
                            (ecols - ecols.min() + target.x).tolist()))
    locks = []
    place = env.place_active_tetromino

    def record():
        active = env.active_tetromino
        rows, cols = np.nonzero(active.matrix > 0)
        locks.append((active.id, active.matrix.copy(),
                      set(zip((rows + env.y).tolist(), (cols + env.x).tolist()))))
        place()

    env.place_active_tetromino = record
    try:
        info = None
        for i, action in enumerate(result.actions):
            _, _, _, _, info = env.step(action)
            assert len(locks) == (1 if not result.needs_hard_drop
                                 and i == len(result.actions) - 1 else 0)
        if result.needs_hard_drop:
            _, _, _, _, info = env.step(env.actions.hard_drop)
        assert len(locks) == 1
        assert locks[0][0] == initial.env_id
        np.testing.assert_array_equal(locks[0][1], expected_matrix)
        assert locks[0][2] == expected_cells
        return info
    finally:
        del env.place_active_tetromino


@pytest.mark.parametrize("index,target,expected_actions", [
    (1, TargetPlacement(10, 18, 0), (1, 1)),  # RIGHT, RIGHT
    (2, TargetPlacement(8, 17, 1), (3,)),  # CW
    (2, TargetPlacement(10, 17, 1), (1, 1, 3)),
    (1, TargetPlacement(8, 18, 0), ()),  # Immediate HARD_DROP
    (1, TargetPlacement(4, 18, 0), (0, 0, 0, 0)),
    (1, TargetPlacement(12, 18, 0), (1, 1, 1, 1)),
    (2, TargetPlacement(9, 17, 3), (4,)),  # CCW has min_col=1.
])
def test_empty_board_native_paths(env, index, target, expected_actions):
    select_piece(env, index)
    before = state_signature(env)
    result = TacticalPlanner().plan(env, target)
    assert state_signature(env) == before
    assert result.actions == expected_actions
    assert result.needs_hard_drop
    assert_replay(env, target, result)


@pytest.mark.parametrize("orientation,x,y,raw_x,raw_y", [
    (0, 4, 19, 4, 18), (0, 10, 19, 10, 18),
    (1, 4, 16, 3, 16), (1, 13, 16, 12, 16),
])
def test_i_offsets_and_extreme_edges(env, orientation, x, y, raw_x, raw_y):
    select_piece(env, 0)
    target = TargetPlacement(x, y, orientation)
    spawn = TetrisAdapter(env).read_current_piece().spawn_matrix
    assert target_to_env_origin(target, spawn) == (raw_x, raw_y)
    before = state_signature(env)
    result = TacticalPlanner().plan(env, target)
    assert state_signature(env) == before
    assert_replay(env, target, result)


def test_sealed_cavity_valid_but_unreachable(env):
    select_piece(env, 1)
    env.board[17, 4:14] = 2
    target = TargetPlacement(7, 18, 0)
    adapter = TetrisAdapter(env)
    assert TargetSpace(adapter.geometry).is_valid(env.board, env.active_tetromino.matrix, target)
    before = state_signature(env)
    result = TacticalPlanner().plan(env, target)
    assert result.status == PlannerStatus.UNREACHABLE
    assert not result.reachable and not result.actions and not result.needs_hard_drop
    assert result.expanded_states == result.visited_states > 0
    assert state_signature(env) == before


def test_visited_expands_each_exact_pose_once(env, monkeypatch):
    select_piece(env, 1)
    env.gravity_enabled = False
    env.board[17, 4:14] = 2
    expanded = []
    non_drop_count = 0
    step = env.step

    def counted(action):
        nonlocal non_drop_count
        if action == env.actions.move_left:  # First candidate per expanded node
            expanded.append((env.x, env.y, env.active_tetromino.matrix.tobytes()))
        if action != env.actions.hard_drop:
            non_drop_count += 1
        return step(action)

    monkeypatch.setattr(env, "step", counted)
    result = TacticalPlanner().plan(env, TargetPlacement(7, 18, 0))
    # O can occupy x=4..12, y=0..15 above the sealed roof.
    assert not result.reachable
    assert result.expanded_states == result.visited_states == 9 * 16
    assert len(expanded) == len(set(expanded)) == result.expanded_states
    assert non_drop_count == 6 * result.expanded_states


@pytest.mark.parametrize("with_blocks", [False, True])
def test_right_at_floor_can_only_succeed_by_auto_lock(env, with_blocks):
    select_piece(env, 1)
    env.x, env.y = 7, 18
    if with_blocks:
        env.board[18:20, 4:14] = 2
        env.board[18:20, 7:10] = 0  # Three-cell opening permits a RIGHT tuck.
    target = TargetPlacement(8, 18, 0)
    before = state_signature(env)
    result = TacticalPlanner().plan(env, target)
    assert state_signature(env) == before
    assert result.actions == (env.actions.move_right,)
    assert not result.needs_hard_drop
    info = assert_replay(env, target, result)
    assert info["lines_cleared"] == 0


@pytest.mark.parametrize("auto_lock", [False, True])
def test_line_clear_before_next_piece_does_not_hide_success(env, auto_lock):
    if auto_lock:
        # Vertical I -> horizontal I. The rotation locks in that same step,
        # clears its entire row, and spawns another piece of the SAME type.
        select_piece(env, 0, orientation=1)
        env.x, env.y = 7, 16
        env.board[17, 4:14] = 3
        env.board[17, 7:11] = 0
        env.board[18, 7] = 3
        env.queue.queue[0] = 0
        target = TargetPlacement(7, 17, 0)
    else:
        select_piece(env, 1)
        # One cleared row and two surviving cells from the original O.
        env.board[19, 4:14] = 2
        env.board[19, 8:10] = 0
        target = TargetPlacement(8, 18, 0)
    before = state_signature(env)
    result = TacticalPlanner().plan(env, target)
    assert state_signature(env) == before
    assert result.needs_hard_drop == (not auto_lock)
    if auto_lock:
        assert result.actions == (env.actions.rotate_counterclockwise,)
    info = assert_replay(env, target, result)
    assert info["lines_cleared"] == 1
    if auto_lock:
        assert not np.any(env.board[:20, 4:14] == 2)
        assert env.active_tetromino.id == 2
    else:
        assert np.count_nonzero(env.board[:20, 4:14] == 3) == 2


def test_two_row_clear_success(env):
    select_piece(env, 1)
    env.board[18:20, 4:14] = 2
    env.board[18:20, 8:10] = 0
    target = TargetPlacement(8, 18, 0)
    result = TacticalPlanner().plan(env, target)
    assert assert_replay(env, target, result)["lines_cleared"] == 2
    assert not np.any(env.board[:20, 4:14])


def test_tuck_requires_descent_and_both_horizontal_directions(env):
    select_piece(env, 1)
    env.board[16, 8:10] = 2  # Must go around this roof before moving underneath.
    target = TargetPlacement(8, 18, 0)
    before = state_signature(env)
    result = TacticalPlanner().plan(env, target)
    assert state_signature(env) == before
    assert env.actions.move_left in result.actions
    assert env.actions.move_right in result.actions
    assert env.actions.move_down in result.actions
    assert_replay(env, target, result)


def test_wrong_auto_lock_cannot_continue_with_next_piece(env):
    select_piece(env, 1)
    env.x, env.y = 7, 18
    env.queue.queue[0] = 1  # Even the next piece having the same ID must stop search.
    before = state_signature(env)
    result = TacticalPlanner().plan(env, TargetPlacement(10, 18, 0))
    assert not result.reachable and result.actions == ()
    assert result.visited_states == result.expanded_states == 1
    assert state_signature(env) == before


def test_translated_i_half_turn_requires_absolute_target_rotation(env):
    select_piece(env, 0, orientation=2)
    target = TargetPlacement(7, 19, 0)
    result = TacticalPlanner().plan(env, target)
    assert result.path_length == 2
    assert_replay(env, target, result)


def test_installed_primitive_gravity_semantics(env):
    select_piece(env, 1)
    state = env.get_state()
    env.step(env.actions.move_right)
    assert (env.x, env.y) == (state.x + 1, state.y + 1)
    env.set_state(state)
    env.step(env.actions.move_down)
    assert (env.x, env.y) == (state.x, state.y + 2)


def test_negative_target_raw_origin_is_safe_unreachable(env, monkeypatch):
    select_piece(env, 0)
    env.board[1, 4:14] = 2
    target = TargetPlacement(4, 0, 0)
    adapter = TetrisAdapter(env)
    assert TargetSpace(adapter.geometry).is_valid(env.board, env.active_tetromino.matrix, target)
    assert target_to_env_origin(target, env.active_tetromino.matrix) == (4, -1)
    before = state_signature(env)

    def forbidden_step(action):
        pytest.fail("Unsupported target must be rejected before native slicing")

    monkeypatch.setattr(env, "step", forbidden_step)
    result = TacticalPlanner().plan(env, target)
    assert not result.reachable and result.visited_states == 0
    assert state_signature(env) == before


@pytest.mark.parametrize("x,y", [(-1, 0), (7, -1)])
def test_negative_current_raw_origin_is_safe(env, x, y):
    select_piece(env, 1)
    env.x, env.y = x, y
    before = state_signature(env)
    result = TacticalPlanner().plan(env, TargetPlacement(8, 18, 0))
    assert not result.reachable
    assert state_signature(env) == before


def test_restoration_even_when_native_step_raises_after_lock(env, monkeypatch):
    select_piece(env, 1)
    step = env.step
    place = env.place_active_tetromino
    monkeypatch.setattr(env, "place_active_tetromino", place)

    def broken_step(action):
        step(action)  # The first goal probe mutates board/queue/RNG by locking.
        raise RuntimeError("injected transition failure")

    monkeypatch.setattr(env, "step", broken_step)
    before = state_signature(env)
    refs = {name: getattr(env, name) for name in
            ("board", "active_tetromino", "queue", "holder", "randomizer")}
    with pytest.raises(RuntimeError, match="injected"):
        TacticalPlanner().plan(env, TargetPlacement(8, 18, 0))
    assert state_signature(env) == before
    assert env.place_active_tetromino is place
    assert all(getattr(env, name) is value for name, value in refs.items())


def test_queue_rng_continuation_is_unchanged(env):
    select_piece(env, 1)
    # Force the next probe's queue refill to reshuffle the bag.
    env.randomizer.index = len(env.randomizer.bag) - 1
    state = env.get_state()
    control = Tetris()
    try:
        control.reset(seed=1)
        control.set_state(copy.deepcopy(state))
        before = state_signature(env)
        TacticalPlanner().plan(env, TargetPlacement(12, 18, 0))
        assert state_signature(env) == before
        assert [env.queue.get_next_tetromino() for _ in range(40)] == [
            control.queue.get_next_tetromino() for _ in range(40)
        ]
    finally:
        control.close()


@pytest.mark.parametrize("reachable", [False, True])
def test_original_objects_and_metadata_are_restored(env, reachable):
    select_piece(env, 1)
    if not reachable:
        env.board[17, 4:14] = 2
    # Include a populated holder and an existing score override, although
    # neither may be used to search with HOLD or modify reward behavior.
    env.holder.queue.append(env.tetrominoes[2])
    env.has_swapped = True
    env.score = lambda rows: rows * 10
    attrs = ("board", "active_tetromino", "queue", "holder", "randomizer", "score")
    refs = {name: getattr(env, name) for name in attrs}
    before = state_signature(env)
    result = TacticalPlanner().plan(env, TargetPlacement(7, 18, 0))
    assert result.reachable == reachable
    assert state_signature(env) == before
    assert all(getattr(env, name) is value for name, value in refs.items())


def test_game_over_after_correct_lock_is_reachable(env):
    select_piece(env, 1)
    env.board[2, 4:14] = 2
    target = TargetPlacement(8, 0, 0)
    before = state_signature(env)
    result = TacticalPlanner().plan(env, target)
    assert state_signature(env) == before
    assert_replay(env, target, result)
    assert env.game_over


def test_already_game_over_does_not_step(env):
    select_piece(env, 1)
    env.game_over = True
    before = state_signature(env)
    result = TacticalPlanner().plan(env, TargetPlacement(8, 18, 0))
    assert not result.reachable and result.expanded_states == 0
    assert state_signature(env) == before


def test_wrappers_are_untouched_and_native_actions_are_returned(env):
    select_piece(env, 1)
    wrapped = gym.wrappers.TimeLimit(env, max_episode_steps=1)
    wrapped.reset(seed=42)
    select_piece(env, 1)
    before = state_signature(env)
    result = TacticalPlanner().plan(wrapped, TargetPlacement(10, 18, 0))
    assert wrapped._elapsed_steps == 0
    assert state_signature(env) == before
    assert_replay(env, TargetPlacement(10, 18, 0), result)


def test_absolute_orientation_does_not_rebase_after_rotation(env):
    select_piece(env, 2, orientation=3)
    target = TargetPlacement(8, 17, 1)
    result = TacticalPlanner().plan(env, target)
    assert_replay(env, target, result)


def test_non_default_geometry_and_action_mapping():
    mapping = ActionsMapping(7, 0, 1, 2, 3, 4, 5, 6)
    env = Tetris(width=6, height=12, actions_mapping=mapping)
    try:
        env.reset(seed=42)
        select_piece(env, 1)
        target = TargetPlacement(8, 10, 0)
        before = state_signature(env)
        result = TacticalPlanner().plan(env, target)
        assert state_signature(env) == before
        assert result.actions == (mapping.move_right, mapping.move_right)
        assert_replay(env, target, result)
    finally:
        env.close()


@pytest.mark.parametrize("target", [
    TargetPlacement(13, 18, 0), TargetPlacement(8, 17, 0),
    TargetPlacement(8, 18, 4), TargetPlacement(8, 18, 1),
])
def test_invalid_targets_are_unreachable(env, target):
    select_piece(env, 1)
    before = state_signature(env)
    result = TacticalPlanner().plan(env, target)
    assert not result.reachable and result.actions == ()
    assert result.visited_states == result.expanded_states == 0
    assert state_signature(env) == before


def test_reset_required():
    env = Tetris()
    try:
        with pytest.raises(RuntimeError, match="Reset"):
            TacticalPlanner().plan(env, TargetPlacement(8, 18, 0))
    finally:
        env.close()
