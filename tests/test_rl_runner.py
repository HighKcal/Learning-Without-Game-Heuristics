"""Real Planner/Executor integration and failures excluded from training data."""
from dataclasses import replace
from random import Random

import gymnasium as gym
import numpy as np
import pytest

from tetris_agent import Executor, ExecutionFailure, TacticalPlanner, TargetPlacement
from tetris_agent.rl.config import TrainingConfig
from tetris_agent.rl.dqn import DQNTrainer
from tetris_agent.rl.policy import FormalRandomPolicy
from tetris_agent.rl.replay import ReplayBuffer
from tetris_agent.rl.runner import RLRunner
from tetris_agent.rl.state import SPACE, read_state, geometric_mask, assert_canonical
from tetris_agent.rl.training import train
from test_planner import select_piece, state_signature
from test_rl_foundation import env  # Existing fixture style, real native environment.


class FixedPolicy:
    def __init__(self, target):
        self.action_id = SPACE.encode(target)
        self.calls = 0

    def select(self, state, mask):
        self.calls += 1
        assert mask[self.action_id]
        return self.action_id


@pytest.mark.parametrize('mode', ['minimal', 'shaped'])
def test_formal_random_repeats_unreachable_with_exact_restoration(env, mode):
    select_piece(env, 1)
    env.board[17, 4:14] = 2
    env.randomizer.index = len(env.randomizer.bag) - 1
    target = TargetPlacement(7, 18, 0)
    action = SPACE.encode(target)
    mask = geometric_mask(read_state(env))
    ids = np.flatnonzero(mask).tolist()
    seed = next(s for s in range(10000)
                if (lambda r: [r.choice(ids), r.choice(ids)])(Random(s)) == [action, action])
    before = state_signature(env)
    transitions = []
    replay = ReplayBuffer(seed=1)
    class NeverExecute:
        def execute(self, *args):
            pytest.fail('UNREACHABLE must never invoke Executor')
    class CheckedRandom(FormalRandomPolicy):
        def select(self, state, received):
            assert_canonical(env)
            assert state_signature(env) == before
            np.testing.assert_array_equal(received, mask)
            return super().select(state, received)
    def store(t):
        transitions.append(t)
        replay.add(t)
    result = RLRunner(CheckedRandom(seed), executor=NeverExecute(), reward_mode=mode).run(
        env, reset=False, max_decisions=2, on_transition=store)
    assert state_signature(env) == before
    assert len(replay) == result.stored_transitions == result.decision_count == 2
    assert [t.action_id for t in transitions] == [action, action]
    assert not transitions[0].truncated and transitions[1].truncated
    assert all(not t.terminated for t in transitions)
    assert all(t.state is t.next_state for t in transitions)
    assert result.unreachable_rate == result.self_loop_ratio == 1
    assert result.max_consecutive_unreachable == result.consecutive_unreachable == 2
    assert result.pieces_placed == result.lines_cleared == 0
    assert result.truncated and not result.terminated and result.diagnostic is None
    assert result.reward_totals['unreachable'] == -.1
    assert result.reward_totals['death'] == 0
    assert result.reward_totals['shaping'] > 0 if mode == 'shaped' else result.reward_totals['shaping'] == 0


@pytest.mark.parametrize('rows', [0, 1, 2, 3, 4])
def test_actual_line_reward_and_last_real_truncated_next_state(env, rows):
    if rows > 2:
        select_piece(env, 0)
        env.board[20 - rows:20, 4:14] = 2
        env.board[20 - rows:20, 9] = 0
        target = TargetPlacement(9, 16, 1)
    else:
        select_piece(env, 1)
        env.board[20 - rows:20, 4:14] = 2
        env.board[20 - rows:20, 8:10] = 0
        target = TargetPlacement(8, 18, 0)
    env.score = lambda _: 12345  # Native game reward is not the RL line reward.
    transitions = []
    before = read_state(env)
    result = RLRunner(FixedPolicy(target)).run(env, reset=False, max_decisions=1,
                                               on_transition=transitions.append)
    assert result.diagnostic is None
    assert result.lines_cleared == rows and result.episode_return == rows
    assert result.pieces_placed == result.decision_count == 1
    assert result.truncated and not result.terminated
    t, = transitions
    assert t.truncated and not t.terminated and t.reward == rows
    np.testing.assert_array_equal(t.state.board, before.board)
    np.testing.assert_array_equal(t.next_state.board, read_state(env).board)
    assert t.next_state.current_piece == before.next_queue[0]
    assert_canonical(env)
    assert 'clear_filled_rows' not in vars(env)


@pytest.mark.parametrize('clear_and_die', [False, True])
def test_actual_terminal_success_and_simultaneous_line_death(env, clear_and_die):
    select_piece(env, 1)
    if clear_and_die:
        # O falls down columns 8/9; next I collides at spawn (row 1).
        env.board[0, 7] = 2
        env.board[19, 4:14] = 2
        env.board[19, 8:10] = 0
        env.queue.queue[0] = 0
        target = TargetPlacement(8, 18, 0)
    else:
        env.board[2, 8:10] = 2  # Support without an already-full row to clear.
        target = TargetPlacement(8, 0, 0)
    transitions = []
    result = RLRunner(FixedPolicy(target), reward_mode='shaped').run(
        env, reset=False, max_decisions=1, on_transition=transitions.append)
    assert result.diagnostic is None and result.terminated and not result.truncated
    assert result.pieces_placed == 1 and result.reward_totals['death'] == -1
    assert result.lines_cleared == int(clear_and_die)
    assert transitions[0].terminated
    from tetris_agent.rl.reward import potential
    assert result.reward_totals['shaping'] == pytest.approx(-potential(transitions[0].state))


@pytest.mark.parametrize('failure', ['exception', 'timeout', 'unknown'])
def test_planner_uncertainty_is_not_agent_failure(env, failure):
    class UncertainPlanner:
        def plan(self, game, target):
            if failure == 'exception':
                raise RuntimeError('injected planner error')
            if failure == 'timeout':
                raise TimeoutError('injected search timeout')
            return replace(TacticalPlanner().plan(game, target), status='UNKNOWN')
    transitions = []
    select_piece(env, 1)
    result = RLRunner(FixedPolicy(TargetPlacement(8, 18, 0)), planner=UncertainPlanner()).run(
        env, reset=False, max_decisions=2, on_transition=transitions.append)
    assert not transitions and result.stored_transitions == result.unreachable_count == 0
    assert result.episode_return == 0 and result.decision_count == 1
    assert result.diagnostic and result.stop_reason in ('PLANNER_ERROR', 'PLANNER_UNKNOWN')


@pytest.mark.parametrize('failure', list(ExecutionFailure))
def test_execution_failure_excluded_even_after_live_mutation(env, failure):
    select_piece(env, 1)
    class FailingExecutor(Executor):
        def execute(self, *args):
            return replace(super().execute(*args), success=False, failure_reason=failure,
                           error_message='injected failure')
    transitions = []
    result = RLRunner(FixedPolicy(TargetPlacement(8, 18, 0)), executor=FailingExecutor()).run(
        env, reset=False, max_decisions=2, on_transition=transitions.append)
    assert not transitions and result.episode_return == 0
    assert result.stop_reason == 'EXECUTION_ERROR' and 'injected failure' in result.diagnostic
    assert result.unreachable_count == result.pieces_placed == 0
    assert np.count_nonzero(env.board[:20, 4:14]) == 4


def test_wrapper_error_after_clear_is_diagnostic_without_reward_or_replay(env):
    select_piece(env, 1)
    env.board[18:20, 4:14] = 2
    env.board[18:20, 8:10] = 0
    class BrokenWrapper(gym.Wrapper):
        def step(self, action):
            self.env.step(action)
            raise RuntimeError('after actual clear')
    transitions = []
    result = RLRunner(FixedPolicy(TargetPlacement(8, 18, 0))).run(
        BrokenWrapper(env), reset=False, max_decisions=1, on_transition=transitions.append)
    assert result.lines_cleared == 2 and result.episode_return == 0
    assert not transitions and 'after actual clear' in result.diagnostic
    assert 'clear_filled_rows' not in vars(env)


@pytest.mark.parametrize('on_lock', [False, True])
def test_primitive_time_limit_only_stores_verified_canonical_transitions(env, on_lock):
    wrapped = gym.wrappers.TimeLimit(env, max_episode_steps=1)
    wrapped.reset(seed=42)
    select_piece(env, 1)
    transitions = []
    result = RLRunner(FixedPolicy(TargetPlacement(8 if on_lock else 10, 18, 0))).run(
        wrapped, reset=False, max_decisions=10, on_transition=transitions.append)
    assert result.truncated and not result.terminated
    assert len(transitions) == int(on_lock)
    assert result.stop_reason == ('TRUNCATED' if on_lock else 'EXECUTION_ERROR')
    assert result.reward_totals['death'] == 0


def test_policy_is_only_called_at_spawn_across_real_placements(env):
    class CheckedRandom(FormalRandomPolicy):
        calls = 0
        def select(self, state, mask):
            assert_canonical(env)
            self.calls += 1
            return super().select(state, mask)
    policy = CheckedRandom(42)
    result = RLRunner(policy).run(env, env_seed=42, max_decisions=12)
    assert result.diagnostic is None and result.pieces_placed > 1
    assert result.decision_count == policy.calls
    assert result.decision_count == result.pieces_placed + result.unreachable_count


def test_noncanonical_environment_never_calls_policy(env):
    env.step(env.actions.move_right)
    policy = FixedPolicy(TargetPlacement(8, 18, 0))
    result = RLRunner(policy).run(env, reset=False, max_decisions=1)
    assert policy.calls == 0 and result.stop_reason == 'STATE_ERROR'


@pytest.mark.parametrize('algorithm', ['vanilla', 'double'])
def test_tiny_real_training_budget_utd_and_episode_resets(env, algorithm):
    trainer = DQNTrainer(TrainingConfig(decision_budget=3, algorithm=algorithm, device='cpu',
                                       learning_starts=2, batch_size=2))
    updates = []
    result = train(env, trainer, episode_decision_limit=2, on_learning=updates.append)
    assert result.decision_count == result.stored_experiences == len(trainer.replay) == 3
    assert result.gradient_updates == len(updates) == 2
    assert result.target_sync_count == 0
    assert len(result.episodes) == 2
    assert all(e.diagnostic is None and e.truncated for e in result.episodes)
    assert all(np.isfinite(m.loss) for m in updates)
