from dataclasses import replace

import gymnasium as gym
import numpy as np
import pytest
from tetris_gymnasium.envs.tetris import Tetris

from tetris_agent import (
    ExecutionFailure, Executor, GameRunner, GameStopReason,
    RandomDecisionAgent, TacticalPlanner, TargetPlacement, TargetSpace, TetrisAdapter,
)
from test_planner import select_piece, state_signature


@pytest.fixture
def env():
    game = Tetris(gravity=True)
    game.reset(seed=42)
    yield game
    game.close()


def seed_for_targets(env, *targets):
    """Choose a reproducible RNG seed for a controlled geometric candidate set."""
    state = TetrisAdapter(env).read()
    space = TargetSpace(TetrisAdapter(env).geometry)
    ids = np.flatnonzero(space.action_mask(state.board, state.current_piece.spawn_matrix)).tolist()
    for seed in range(10000):
        agent = RandomDecisionAgent(seed)
        remaining = ids.copy()
        selected = []
        for _ in targets:
            decision = agent.choose(space, remaining)
            selected.append(decision.target)
            remaining.remove(decision.action_id)
        if tuple(selected) == targets:
            return seed
    raise AssertionError("No test seed found")


def one_step_env(env):
    wrapped = gym.wrappers.TimeLimit(env, max_episode_steps=1)
    wrapped.reset(seed=42)
    return wrapped


def test_one_random_piece_advances_real_queue_and_truncates(env):
    wrapped = one_step_env(env)
    select_piece(env, 1)
    initial_queue = tuple(env.queue.get_queue())
    seed = seed_for_targets(env, TargetPlacement(8, 18, 0))
    result = GameRunner(RandomDecisionAgent(seed)).run(wrapped, reset=False)
    assert result.pieces_placed == result.decision_count == 1
    assert result.unreachable_count == result.executor_failure_count == result.lines_cleared == 0
    assert result.stop_reason == GameStopReason.TRUNCATED
    assert result.truncated and not result.terminated
    assert result.attempts[0].execution_success
    assert env.active_tetromino.id == env.tetrominoes[initial_queue[0]].id
    assert tuple(env.queue.get_queue())[:-1] == initial_queue[1:]
    assert env.y == 0 and np.count_nonzero(env.board[:20, 4:14]) == 4
    assert "clear_filled_rows" not in vars(env)


def test_multiple_random_placements_and_seeded_reproduction(env):
    first = GameRunner(RandomDecisionAgent(42)).run(env, env_seed=42)
    board = env.board.copy()
    second = GameRunner(RandomDecisionAgent(42)).run(env, env_seed=42)
    assert first.pieces_placed > 1
    assert first.stop_reason == second.stop_reason == GameStopReason.TERMINATED
    assert first.executor_failure_count == second.executor_failure_count == 0
    assert [(a.turn, a.decision, a.plan.status, a.execution_success) for a in first.attempts] == [
        (a.turn, a.decision, a.plan.status, a.execution_success) for a in second.attempts
    ]
    np.testing.assert_array_equal(board, env.board)
    assert first.lines_cleared == second.lines_cleared
    assert first.decision_count == first.pieces_placed + first.unreachable_count
    assert first.unreachable_rate == first.unreachable_count / first.decision_count
    assert first.game_duration > 0
    assert 0 <= first.planner_avg_ms <= first.planner_max_ms
    assert 0 <= first.planner_p95_ms <= first.planner_max_ms
    assert len(first.planner_times_ms) == first.decision_count
    for turn in {a.turn for a in first.attempts}:
        ids = [a.decision.action_id for a in first.attempts if a.turn == turn]
        assert len(ids) == len(set(ids))


def test_real_unreachable_is_removed_and_remaining_candidate_executes(env, capsys):
    wrapped = one_step_env(env)
    select_piece(env, 1)
    env.board[17, 4:14] = 2
    cavity, reachable = TargetPlacement(7, 18, 0), TargetPlacement(8, 15, 0)
    seed = seed_for_targets(env, cavity, reachable)
    before = state_signature(env)

    class CheckedPlanner(TacticalPlanner):
        def plan(self, game, target):
            # The first rejected choice must leave the exact live state intact.
            assert state_signature(game.unwrapped) == before
            return replace(super().plan(game, target),
                           elapsed_seconds=0.004 if target == cavity else 0.002)

    result = GameRunner(RandomDecisionAgent(seed), planner=CheckedPlanner()).run(
        wrapped, reset=False, verbose=True)
    assert result.decision_count == 2 and result.unreachable_count == 1
    assert result.unreachable_rate == 0.5 and result.pieces_placed == 1
    assert result.executor_failure_count == 0
    assert [a.decision.target for a in result.attempts] == [cavity, reachable]
    assert [a.attempt for a in result.attempts] == [1, 2]
    assert [a.execution_success for a in result.attempts] == [None, True]
    assert result.planner_avg_ms == 3 and result.planner_p95_ms == result.planner_max_ms == 4
    assert "remaining candidates:" in capsys.readouterr().out


def test_all_geometric_candidates_actually_tried_once_without_retry_cap(env):
    select_piece(env, 1)
    # More than ten resting placements, but the active spawn already collides.
    # Each real Planner call returns UNREACHABLE; no fake reachability mask.
    env.board[8, 4:14] = 2
    env.board[17, 4:14] = 2
    env.board[0, 8] = 2
    state = TetrisAdapter(env).read()
    space = TargetSpace(TetrisAdapter(env).geometry)
    valid_ids = set(np.flatnonzero(space.action_mask(state.board, state.current_piece.spawn_matrix)))
    before = state_signature(env)
    result = GameRunner(RandomDecisionAgent(7)).run(env, reset=False)
    assert len(valid_ids) > 10
    assert result.stop_reason == GameStopReason.ALL_VALID_TARGETS_UNREACHABLE
    chosen = [a.decision.action_id for a in result.attempts]
    assert len(chosen) == len(set(chosen)) == len(valid_ids)
    assert set(chosen) == valid_ids
    assert result.unreachable_count == result.decision_count == len(valid_ids)
    assert result.unreachable_rate == 1
    assert result.pieces_placed == result.lines_cleared == result.executor_failure_count == 0
    assert state_signature(env) == before


def test_no_valid_targets_safe_empty_statistics(env):
    env.board[:20, 4:14] = 2
    result = GameRunner(RandomDecisionAgent(0)).run(env, reset=False)
    assert result.stop_reason == GameStopReason.NO_VALID_TARGETS
    assert result.decision_count == result.unreachable_count == result.pieces_placed == 0
    assert result.unreachable_rate == result.planner_avg_ms == result.planner_p95_ms == result.planner_max_ms == 0
    assert result.attempts == ()


def test_correct_lock_game_over_is_not_execution_failure(env):
    select_piece(env, 1)
    env.board[2, 4:14] = 2
    seed = seed_for_targets(env, TargetPlacement(8, 0, 0))
    result = GameRunner(RandomDecisionAgent(seed)).run(env, reset=False)
    assert result.stop_reason == GameStopReason.TERMINATED
    assert result.terminated and not result.truncated
    assert result.pieces_placed == result.decision_count == 1
    assert result.executor_failure_count == 0 and result.executor_failures == {}


@pytest.mark.parametrize("rows", [1, 2])
def test_direct_line_clear_signal_and_non_reward_count(env, rows):
    wrapped = one_step_env(env)
    select_piece(env, 1)
    env.board[20 - rows:20, 4:14] = 2
    env.board[20 - rows:20, 8:10] = 0
    # Reward cannot encode the row count: it is deliberately constant here.
    env.score = lambda _: 9876.0
    seed = seed_for_targets(env, TargetPlacement(8, 18, 0))
    result = GameRunner(RandomDecisionAgent(seed)).run(wrapped, reset=False)
    assert result.lines_cleared == result.attempts[0].lines_cleared == rows
    assert result.pieces_placed == 1 and result.executor_failure_count == 0
    assert np.count_nonzero(env.board[:20, 4:14]) == 4 - rows * 2


def test_auto_lock_clear_and_existing_clear_observer_restored(env, monkeypatch):
    wrapped = one_step_env(env)
    select_piece(env, 0, orientation=1)
    env.x, env.y = 7, 16
    env.board[17, 4:14] = 3
    env.board[17, 7:11] = 0
    env.board[18, 7] = 3
    env.queue.queue[0] = 0
    clear = env.clear_filled_rows
    monkeypatch.setattr(env, "clear_filled_rows", clear)
    seed = seed_for_targets(env, TargetPlacement(7, 17, 0))
    result = GameRunner(RandomDecisionAgent(seed)).run(wrapped, reset=False)
    assert result.lines_cleared == result.pieces_placed == 1
    assert not result.attempts[0].plan.needs_hard_drop
    assert env.clear_filled_rows is clear
    assert env.active_tetromino.id == env.tetrominoes[0].id and env.y == 0


@pytest.mark.parametrize("failure", [ExecutionFailure.EARLY_LOCK, ExecutionFailure.PLAN_DIVERGED,
                                     ExecutionFailure.TARGET_MISMATCH, ExecutionFailure.TRANSITION_ERROR])
def test_execution_failures_stop_without_another_decision_or_replanning(env, failure):
    select_piece(env, 1)
    seed = seed_for_targets(env, TargetPlacement(10, 18, 0))

    class FailingExecutor(Executor):
        calls = 0

        def execute(self, game, target, plan):
            self.calls += 1
            # Controlled result exercises each reason at the orchestration boundary.
            return replace(super().execute(game, target, plan),
                           success=False, failure_reason=failure)

    executor = FailingExecutor()
    result = GameRunner(RandomDecisionAgent(seed), executor=executor).run(env, reset=False)
    assert executor.calls == result.decision_count == 1
    assert result.pieces_placed == 0  # Only Executor success counts.
    assert result.stop_reason == GameStopReason.EXECUTOR_FAILURE
    assert result.executor_failure_count == 1
    assert result.executor_failures == {failure.value: 1}
    assert result.attempts[0].failure_reason == failure


def test_real_time_limit_execution_failure_preserves_flags(env):
    wrapped = one_step_env(env)
    select_piece(env, 1)
    seed = seed_for_targets(env, TargetPlacement(10, 18, 0))
    result = GameRunner(RandomDecisionAgent(seed)).run(wrapped, reset=False)
    assert result.truncated and not result.terminated
    assert result.stop_reason == GameStopReason.EXECUTOR_FAILURE
    assert result.executor_failures == {"EPISODE_ENDED": 1}
    assert result.pieces_placed == 0 and result.decision_count == 1


def test_native_clear_count_survives_wrapper_error_after_actual_clear(env):
    select_piece(env, 1)
    env.board[18:20, 4:14] = 2
    env.board[18:20, 8:10] = 0
    seed = seed_for_targets(env, TargetPlacement(8, 18, 0))

    class RaisesAfterStep(gym.Wrapper):
        def step(self, action):
            self.env.step(action)
            raise RuntimeError("wrapper failed after clear")

    result = GameRunner(RandomDecisionAgent(seed)).run(RaisesAfterStep(env), reset=False)
    assert result.lines_cleared == 2  # Even with no returned ExecutionStep/info.
    assert result.pieces_placed == 0 and result.executor_failure_count == 1
    assert result.executor_failures == {"TRANSITION_ERROR": 1}
    assert "wrapper failed" in result.attempts[0].error_message
    assert "clear_filled_rows" not in vars(env)
    assert not np.any(env.board[:20, 4:14])


def test_single_call_percentile_and_formatted_summary(env):
    wrapped = one_step_env(env)
    select_piece(env, 1)
    seed = seed_for_targets(env, TargetPlacement(8, 18, 0))
    result = GameRunner(RandomDecisionAgent(seed)).run(wrapped, reset=False)
    assert result.planner_avg_ms == result.planner_max_ms == result.planner_p95_ms
    assert result.planner_p95_ms > 0
    # Nontrivial nearest-rank percentile, including all calls.
    attempts = tuple(replace(result.attempts[0], plan=replace(result.attempts[0].plan,
                            elapsed_seconds=i / 1000)) for i in range(1, 21))
    many = replace(result, attempts=attempts, decision_count=20)
    assert many.planner_avg_ms == 10.5
    assert many.planner_p95_ms == 19 and many.planner_max_ms == 20
    output = many.format()
    assert "Game Summary" in output and "planner p95: 19.000 ms" in output
    assert "stop reason: TRUNCATED" in output


def test_already_game_over_and_reset_required(env):
    env.game_over = True
    result = GameRunner().run(env, reset=False)
    assert result.stop_reason == GameStopReason.TERMINATED
    assert result.decision_count == 0
    game = Tetris()
    try:
        with pytest.raises(RuntimeError, match="Reset"):
            GameRunner().run(game, reset=False)
    finally:
        game.close()
