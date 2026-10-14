"""Live plan execution, independently observed pre-clear locks and stop rules."""

from dataclasses import replace

import gymnasium as gym
import numpy as np
import pytest
from tetris_gymnasium.envs.tetris import Tetris
from tetris_gymnasium.mappings.actions import ActionsMapping

from tetris_agent import (
    ExecutionFailure, Executor, PlannerResult, PlannerStatus,
    TacticalPlanner, TargetPlacement, TetrisAdapter,
)
from test_planner import select_piece, state_signature


@pytest.fixture
def env():
    game = Tetris(gravity=True)
    game.reset(seed=42)
    yield game
    game.close()


def controlled_plan(actions=(), needs_hard_drop=True):
    return PlannerResult(PlannerStatus.REACHABLE, tuple(actions),
                         needs_hard_drop, 0, 0, 0.0)


def spy_steps(env, monkeypatch):
    calls, returns = [], []
    step = env.step

    def observed(action):
        calls.append(action)
        output = step(action)
        returns.append(output)
        return output

    monkeypatch.setattr(env, "step", observed)
    return calls, returns


@pytest.mark.parametrize("index,target,prefix", [
    (1, TargetPlacement(10, 18, 0), (1, 1)),
    (2, TargetPlacement(8, 17, 1), (3,)),
    (2, TargetPlacement(10, 17, 1), (1, 1, 3)),
    (1, TargetPlacement(8, 18, 0), ()),
    (2, TargetPlacement(9, 17, 3), (4,)),
    (0, TargetPlacement(4, 16, 1), (0, 0, 0, 3, 0)),
])
def test_real_plan_replays_and_changes_live_state(env, monkeypatch, index, target, prefix):
    select_piece(env, index)
    initial = TetrisAdapter(env).read()
    before = state_signature(env)
    plan = TacticalPlanner().plan(env, target)
    assert state_signature(env) == before  # Planner is read-only.
    assert plan.actions == prefix and plan.needs_hard_drop
    calls, outputs = spy_steps(env, monkeypatch)
    result = Executor().execute(env, target, plan)
    expected = prefix + (env.actions.hard_drop,)
    assert result.success and result.target_verified
    assert result.failure_reason is None and result.error_message is None
    assert result.executed_actions == tuple(calls) == expected
    assert not result.terminated and not result.truncated
    assert len(result.steps) == len(outputs)
    for action, record, output in zip(expected, result.steps, outputs):
        assert record.action == action
        assert record.observation is output[0]
        assert record.reward == output[1]
        assert (record.terminated, record.truncated, record.info) == output[2:]
    assert result.raw_reward_sum == sum(o[1] for o in outputs)
    assert result.final_observation is outputs[-1][0]
    assert result.final_info is outputs[-1][4]
    # Independent cell oracle, without target/origin conversion helpers.
    expected_board = initial.board.copy()
    rows, cols = np.nonzero(np.rot90(initial.current_piece.spawn_matrix, target.orientation))
    expected_board[rows - rows.min() + target.y,
                   cols - cols.min() + target.x] = initial.current_piece.env_id
    np.testing.assert_array_equal(env.board, expected_board)
    assert state_signature(env) != before  # Executor really advances the game.
    assert env.active_tetromino.id == env.tetrominoes[2].id  # Seed 42 next: T.
    assert env.x == env.padding + (env.width // 2) - 1 and env.y == 0
    assert "place_active_tetromino" not in vars(env)


def test_auto_lock_right_does_not_drop_same_type_next_piece(env, monkeypatch):
    select_piece(env, 1)
    env.x, env.y = 7, 18
    env.queue.queue[0] = 1  # Next O has the SAME environment ID.
    target = TargetPlacement(8, 18, 0)
    before = state_signature(env)
    plan = TacticalPlanner().plan(env, target)
    assert state_signature(env) == before
    assert plan.actions == (env.actions.move_right,) and not plan.needs_hard_drop
    calls, outputs = spy_steps(env, monkeypatch)
    result = Executor().execute(env, target, plan)
    assert result.success and result.target_verified
    assert tuple(calls) == result.executed_actions == plan.actions
    assert env.active_tetromino.id == env.tetrominoes[1].id
    assert env.y == 0 and np.count_nonzero(env.board[:20, 4:14]) == 4
    assert result.raw_reward_sum == outputs[0][1]


@pytest.mark.parametrize("auto_lock,rows_cleared", [(True, 1), (False, 1), (False, 2)])
def test_line_clear_verified_before_clear_and_same_type_spawn(
        env, monkeypatch, auto_lock, rows_cleared):
    if auto_lock:
        select_piece(env, 0, orientation=1)
        env.x, env.y = 7, 16
        env.board[17, 4:14] = 3
        env.board[17, 7:11] = 0
        env.board[18, 7] = 3
        env.queue.queue[0] = 0
        target = TargetPlacement(7, 17, 0)
    else:
        select_piece(env, 1)
        env.board[20 - rows_cleared:20, 4:14] = 2
        env.board[20 - rows_cleared:20, 8:10] = 0
        target = TargetPlacement(8, 18, 0)
    original = TetrisAdapter(env).read_current_piece()
    before = state_signature(env)
    plan = TacticalPlanner().plan(env, target)
    assert state_signature(env) == before
    assert plan.needs_hard_drop == (not auto_lock)
    # Independent observer sees actual cells at native placement.
    locks = []
    place = env.place_active_tetromino

    def record_lock():
        active = env.active_tetromino
        rows, cols = np.nonzero(active.matrix)
        locks.append((active.id, set(zip((rows + env.y).tolist(),
                                        (cols + env.x).tolist()))))
        place()

    monkeypatch.setattr(env, "place_active_tetromino", record_lock)
    calls, outputs = spy_steps(env, monkeypatch)
    result = Executor().execute(env, target, plan)
    expected_matrix = np.rot90(original.spawn_matrix, target.orientation)
    rows, cols = np.nonzero(expected_matrix)
    expected_cells = set(zip((rows - rows.min() + target.y).tolist(),
                             (cols - cols.min() + target.x).tolist()))
    assert locks == [(original.env_id, expected_cells)]
    assert result.success and result.target_verified
    assert result.final_info["lines_cleared"] == rows_cleared
    assert result.raw_reward_sum == sum(o[1] for o in outputs)
    assert env.place_active_tetromino is record_lock
    assert tuple(calls) == plan.actions + (() if auto_lock else (env.actions.hard_drop,))
    if auto_lock:
        assert plan.actions == (env.actions.rotate_counterclockwise,)
        assert env.active_tetromino.id == original.env_id
        assert not np.any(env.board[:20, 4:14] == original.env_id)
        assert env.y == 0
    elif rows_cleared == 2:
        assert not np.any(env.board[:20, 4:14])


def test_unreachable_never_steps_or_changes_environment(env, monkeypatch):
    select_piece(env, 1)
    env.board[17, 4:14] = 2
    target = TargetPlacement(7, 18, 0)
    plan = TacticalPlanner().plan(env, target)
    assert not plan.reachable
    # Even a contradictory UNREACHABLE plan may never dispatch these actions.
    plan = replace(plan, actions=(env.actions.move_right,), needs_hard_drop=True)
    calls, _ = spy_steps(env, monkeypatch)
    before = state_signature(env)
    refs = {name: getattr(env, name) for name in
            ("board", "active_tetromino", "queue", "holder", "randomizer")}
    result = Executor().execute(env, target, plan)
    assert not result.success and not result.target_verified
    assert result.failure_reason == ExecutionFailure.UNREACHABLE
    assert calls == [] and result.executed_actions == () and result.steps == ()
    assert result.raw_reward_sum == 0
    assert result.final_observation is None and result.final_info is None
    assert state_signature(env) == before
    assert all(getattr(env, name) is value for name, value in refs.items())


@pytest.mark.parametrize("needs_hard_drop", [True, False])
@pytest.mark.parametrize("early_target_matches", [True, False])
def test_unexpected_early_lock_stops_before_same_type_next_piece(
        env, monkeypatch, needs_hard_drop, early_target_matches):
    select_piece(env, 1)
    target = TargetPlacement(8 if early_target_matches else 10, 18, 0)
    plan = controlled_plan((env.actions.move_right, env.actions.move_right,
                            env.actions.rotate_clockwise), needs_hard_drop)
    # Simulate live-state divergence after planning: already resting on floor.
    env.x, env.y = 7, 18
    env.queue.queue[0] = 1
    calls, _ = spy_steps(env, monkeypatch)
    result = Executor().execute(env, target, plan)
    assert result.failure_reason == ExecutionFailure.EARLY_LOCK
    assert not result.success
    assert result.target_verified == early_target_matches
    assert tuple(calls) == result.executed_actions == (env.actions.move_right,)
    assert env.active_tetromino.id == env.tetrominoes[1].id
    assert env.y == 0


def test_lock_on_last_primitive_before_promised_drop_is_early(env, monkeypatch):
    select_piece(env, 1)
    env.x, env.y = 7, 18
    target = TargetPlacement(8, 18, 0)
    plan = controlled_plan((env.actions.move_right,), needs_hard_drop=True)
    calls, _ = spy_steps(env, monkeypatch)
    result = Executor().execute(env, target, plan)
    assert not result.success and result.target_verified
    assert result.failure_reason == ExecutionFailure.EARLY_LOCK
    assert calls == [env.actions.move_right]  # Never execute final drop on next piece.


@pytest.mark.parametrize("index,target", [
    (1, TargetPlacement(10, 18, 0)),  # Wrong x.
    (1, TargetPlacement(8, 17, 0)),  # Wrong y.
    (2, TargetPlacement(8, 17, 1)),  # Wrong absolute matrix orientation.
])
def test_mismatching_drop_is_failure(env, monkeypatch, index, target):
    select_piece(env, index)
    calls, _ = spy_steps(env, monkeypatch)
    result = Executor().execute(env, target, controlled_plan())
    assert not result.success and not result.target_verified
    assert result.failure_reason == ExecutionFailure.TARGET_MISMATCH
    assert calls == [env.actions.hard_drop]
    assert np.count_nonzero(env.board[:20, 4:14]) == 4


def test_missing_promised_auto_lock_is_divergence(env, monkeypatch):
    select_piece(env, 1)
    calls, _ = spy_steps(env, monkeypatch)
    result = Executor().execute(env, TargetPlacement(9, 18, 0),
                                controlled_plan((env.actions.move_right,), False))
    assert result.failure_reason == ExecutionFailure.PLAN_DIVERGED
    assert not result.target_verified
    assert calls == [env.actions.move_right]
    assert not np.any(env.board[:20, 4:14])  # No recovery/fallback drop.


def test_correct_target_lock_can_terminate(env, monkeypatch):
    select_piece(env, 1)
    env.board[2, 4:14] = 2
    target = TargetPlacement(8, 0, 0)
    plan = TacticalPlanner().plan(env, target)
    calls, outputs = spy_steps(env, monkeypatch)
    result = Executor().execute(env, target, plan)
    assert result.success and result.target_verified and result.terminated
    assert not result.truncated and result.failure_reason is None
    assert result.raw_reward_sum == outputs[-1][1]
    assert calls == [env.actions.hard_drop] and env.game_over


@pytest.mark.parametrize("lock_on_limit", [False, True])
def test_time_limit_stops_prefix_but_correct_final_lock_still_succeeds(
        env, monkeypatch, lock_on_limit):
    wrapped = gym.wrappers.TimeLimit(env, max_episode_steps=1)
    wrapped.reset(seed=42)
    select_piece(env, 1)
    target = TargetPlacement(8 if lock_on_limit else 10, 18, 0)
    plan = TacticalPlanner().plan(wrapped, target)
    assert wrapped._elapsed_steps == 0
    calls, _ = spy_steps(wrapped, monkeypatch)
    result = Executor().execute(wrapped, target, plan)
    assert result.truncated and not result.terminated
    assert result.success == result.target_verified == lock_on_limit
    assert result.failure_reason == (None if lock_on_limit else ExecutionFailure.EPISODE_ENDED)
    assert len(calls) == wrapped._elapsed_steps == 1
    assert calls == [env.actions.hard_drop if lock_on_limit else env.actions.move_right]


def test_termination_without_lock_stops_remaining_actions(env, monkeypatch):
    select_piece(env, 1)
    target = TargetPlacement(10, 18, 0)
    plan = TacticalPlanner().plan(env, target)
    step = env.step
    calls = []

    def stop(action):
        calls.append(action)
        obs, reward, _, truncated, info = step(action)
        env.game_over = True
        return obs, reward, True, truncated, info

    monkeypatch.setattr(env, "step", stop)
    result = Executor().execute(env, target, plan)
    assert result.failure_reason == ExecutionFailure.EPISODE_ENDED
    assert result.terminated and not result.target_verified
    assert calls == [env.actions.move_right]


@pytest.mark.parametrize("after_lock", [False, True])
def test_raising_transition_returns_failure_and_restores_hook_only(env, monkeypatch, after_lock):
    select_piece(env, 1)
    target = TargetPlacement(8 if after_lock else 10, 18, 0)
    plan = TacticalPlanner().plan(env, target)
    place, step = env.place_active_tetromino, env.step
    monkeypatch.setattr(env, "place_active_tetromino", place)
    calls = []

    def broken(action):
        calls.append(action)
        step(action)  # May mutate/lock before failing; executor must not roll back.
        raise RuntimeError("injected step failure")

    monkeypatch.setattr(env, "step", broken)
    before = state_signature(env)
    result = Executor().execute(env, target, plan)
    assert result.failure_reason == ExecutionFailure.TRANSITION_ERROR
    assert not result.success and result.target_verified == after_lock
    assert result.error_message == "RuntimeError: injected step failure"
    assert tuple(calls) == result.executed_actions
    assert len(calls) == 1 and result.steps == ()
    assert result.final_observation is None and result.final_info is None
    assert result.raw_reward_sum == 0  # Raising step returned no reward.
    assert state_signature(env) != before
    assert env.place_active_tetromino is place


def test_failure_keeps_previous_returned_step_and_reward(env, monkeypatch):
    select_piece(env, 1)
    target = TargetPlacement(10, 18, 0)
    plan = TacticalPlanner().plan(env, target)
    step = env.step
    calls, outputs = [], []

    def broken_second(action):
        calls.append(action)
        if len(calls) == 2:
            raise ValueError("bad action transition")
        output = step(action)
        outputs.append(output)
        return output

    monkeypatch.setattr(env, "step", broken_second)
    result = Executor().execute(env, target, plan)
    assert result.failure_reason == ExecutionFailure.TRANSITION_ERROR
    assert result.executed_actions == tuple(calls) == plan.actions
    assert len(result.steps) == 1
    assert result.final_observation is outputs[0][0]
    assert result.final_info is outputs[0][4]
    assert result.raw_reward_sum == outputs[0][1]
    assert env.x == 9  # Only first RIGHT took effect.
    assert "place_active_tetromino" not in vars(env)


@pytest.mark.parametrize("action_name", ["hard_drop", "swap", "invalid"])
def test_bad_plan_prefix_is_rejected_before_dispatch(env, monkeypatch, action_name):
    action = -1 if action_name == "invalid" else getattr(env.actions, action_name)
    calls, _ = spy_steps(env, monkeypatch)
    before = state_signature(env)
    result = Executor().execute(env, TargetPlacement(8, 18, 0),
                                controlled_plan((env.actions.move_right, action)))
    assert result.failure_reason == ExecutionFailure.PLAN_DIVERGED
    assert calls == [] and state_signature(env) == before


def test_empty_auto_lock_plan_does_not_fallback(env, monkeypatch):
    calls, _ = spy_steps(env, monkeypatch)
    result = Executor().execute(env, TargetPlacement(8, 18, 0), controlled_plan((), False))
    assert result.failure_reason == ExecutionFailure.PLAN_DIVERGED
    assert calls == []


def test_already_game_over_does_not_step(env, monkeypatch):
    env.game_over = True
    calls, _ = spy_steps(env, monkeypatch)
    before = state_signature(env)
    result = Executor().execute(env, TargetPlacement(8, 18, 0), controlled_plan())
    assert result.failure_reason == ExecutionFailure.EPISODE_ENDED and result.terminated
    assert calls == [] and state_signature(env) == before


def test_reset_required_returns_state_error():
    game = Tetris()
    try:
        result = Executor().execute(game, TargetPlacement(8, 18, 0), controlled_plan())
        assert result.failure_reason == ExecutionFailure.TRANSITION_ERROR
        assert "Reset" in result.error_message
        assert result.executed_actions == ()
    finally:
        game.close()


def test_action_remapping_wrapper_is_rejected_before_execution(env, monkeypatch):
    class Remap(gym.ActionWrapper):
        def action(self, action):
            return self.env.actions.hard_drop

    select_piece(env, 1)
    wrapped = Remap(env)
    target = TargetPlacement(10, 18, 0)
    plan = TacticalPlanner().plan(wrapped, target)
    calls, _ = spy_steps(env, monkeypatch)
    before = state_signature(env)
    result = Executor().execute(wrapped, target, plan)
    assert result.failure_reason == ExecutionFailure.PLAN_DIVERGED
    assert calls == [] and state_signature(env) == before


def test_non_default_geometry_and_native_action_mapping():
    mapping = ActionsMapping(7, 0, 1, 2, 3, 4, 5, 6)
    game = Tetris(width=6, height=12, actions_mapping=mapping)
    try:
        game.reset(seed=42)
        select_piece(game, 1)
        target = TargetPlacement(8, 10, 0)
        plan = TacticalPlanner().plan(game, target)
        result = Executor().execute(game, target, plan)
        assert result.success and result.target_verified
        assert result.executed_actions == (mapping.move_right, mapping.move_right, mapping.hard_drop)
        assert np.count_nonzero(game.board[:12, 4:10]) == 4
    finally:
        game.close()
