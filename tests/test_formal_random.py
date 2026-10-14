"""Formal Random policy, native decision semantics and durable baseline runs."""
from dataclasses import replace
import importlib.util
import json
from pathlib import Path
from random import Random
import random
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
import torch
from tetris_gymnasium.envs.tetris import Tetris

from tetris_agent import Executor, ExecutionFailure, PlannerStatus, TacticalPlanner, TargetPlacement
from tetris_agent.rl import formal_random as baseline
from tetris_agent.rl.formal_random import FormalRandomConfig, run_formal_random
from tetris_agent.rl.policy import FormalRandomPolicy
from tetris_agent.rl.reward import potential
from tetris_agent.rl.runner import RLRunner
from tetris_agent.rl.state import SPACE, assert_canonical, geometric_mask, read_state
from test_planner import select_piece, state_signature
from test_rl_foundation import env


def records(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def setup_reset(monkeypatch, env, setup):
    original = env.reset
    def reset(*args, **kwargs):
        result = original(*args, **kwargs)
        setup(env)
        return result
    monkeypatch.setattr(env, 'reset', reset)


def roof(env):
    select_piece(env, 1)
    env.board[17, 4:14] = 2
    env.randomizer.index = len(env.randomizer.bag) - 1


def control_choice(monkeypatch, choose):
    """Keep the production policy; control only its owned RNG.choice in tests."""
    policies = []
    class Controlled(FormalRandomPolicy):
        def __init__(self, seed=None):
            super().__init__(seed)
            self._rng = SimpleNamespace(choice=choose)
            policies.append(self)
    monkeypatch.setattr(baseline, 'FormalRandomPolicy', Controlled)
    return policies


def fixed_choice(monkeypatch, target):
    action = SPACE.encode(target)
    calls = []
    def choose(ids):
        calls.append(list(ids))
        assert action in ids
        return action
    control_choice(monkeypatch, choose)
    return calls


def capture_transitions(monkeypatch):
    original = RLRunner.run
    transitions = []
    def run(self, *args, **kwargs):
        # The actual baseline supplies no transition storage callback.
        assert kwargs.get('on_transition') is None
        return original(self, *args, **kwargs, on_transition=transitions.append)
    monkeypatch.setattr(RLRunner, 'run', run)
    return transitions


@pytest.mark.parametrize('piece', range(7))
def test_uniform_candidate_domain_excludes_every_invalid_id(env, piece):
    select_piece(env, piece)
    env.board[17, 4:14] = 2
    state = read_state(env)
    mask = geometric_mask(state)
    before = mask.copy()
    expected = np.flatnonzero(mask).tolist()
    candidates = []
    def choice(ids):
        assert ids == expected
        assert set(ids).isdisjoint(np.flatnonzero(~mask))
        candidates.append(tuple(ids))
        return ids[len(candidates) % len(ids)]
    policy = FormalRandomPolicy(seed=123)
    policy._rng = SimpleNamespace(choice=choice)
    for _ in range(50):
        assert mask[policy.select(state, mask)]
    assert len(candidates) == 50
    np.testing.assert_array_equal(mask, before)


@pytest.mark.parametrize('mode', ['minimal', 'shaped'])
def test_native_unreachable_repeat_no_executor_retry_removal_or_state_change(env, monkeypatch, tmp_path, mode):
    snapshots = []
    def setup(game):
        roof(game)
        snapshots.append(state_signature(game))
    setup_reset(monkeypatch, env, setup)
    calls = fixed_choice(monkeypatch, TargetPlacement(7, 18, 0))
    transitions = capture_transitions(monkeypatch)
    class NeverExecute:
        def execute(self, *args):
            pytest.fail('UNREACHABLE may not execute')
    output = tmp_path / 'unreachable'
    summary = run_formal_random(env, FormalRandomConfig(1, 42, 8, 2, mode, 1),
                                output_dir=output, executor=NeverExecute())
    assert len(calls) == len(transitions) == summary['decision_count'] == 2
    assert calls[0] == calls[1]  # No candidate is removed after failure.
    assert transitions[0].action_id == transitions[1].action_id
    assert all(t.state is t.next_state and not t.terminated for t in transitions)
    assert not transitions[0].truncated and transitions[1].truncated
    assert state_signature(env) == snapshots[0]
    assert summary['unreachable_count'] == summary['self_loop_count'] == 2
    assert summary['unreachable_rate'] == 1 and summary['pieces_placed'] == 0
    assert summary['decisions_per_placement'] is None
    assert summary['max_consecutive_unreachable'] == 2
    assert summary['reward_totals']['unreachable'] == pytest.approx(-.1)
    assert summary['reward_totals']['death'] == 0
    expected_shaping = 2 * (.99 * potential(transitions[0].state) - potential(transitions[0].state))
    assert summary['reward_totals']['shaping'] == pytest.approx(expected_shaping if mode == 'shaped' else 0)
    assert summary['episode_return'] == pytest.approx(-.1 + summary['reward_totals']['shaping'])
    row, = records(output / 'episodes.jsonl')
    assert row['stop_reason'] == 'TRUNCATED' and row['truncated'] and not row['terminated']
    assert row['completed_decisions'] == 2
    assert row['env_seed'] == 42 and row['policy_seed'] == 8


def test_selection_precedes_every_planner_call_and_one_unreachable_is_one_decision(env, monkeypatch):
    events = []
    setup_reset(monkeypatch, env, roof)
    action = SPACE.encode(TargetPlacement(7, 18, 0))
    def choose(ids):
        assert events == []
        events.append('select')
        assert action in ids
        return action
    control_choice(monkeypatch, choose)
    class Planner:
        def plan(self, game, target):
            assert events == ['select']
            events.append('plan')
            return TacticalPlanner().plan(game, target)
    result = run_formal_random(env, FormalRandomConfig(1, 42, 1, 1), planner=Planner())
    assert events == ['select', 'plan']
    assert result['decision_count'] == result['unreachable_count'] == 1
    assert result['episode_return'] == -.05


@pytest.mark.parametrize('failure', ['exception', 'timeout', 'unknown', 'restoration'])
def test_planner_failures_stop_run_without_unreachable_reward(env, monkeypatch, tmp_path, failure):
    def setup(game):
        select_piece(game, 1)
    setup_reset(monkeypatch, env, setup)
    calls = fixed_choice(monkeypatch, TargetPlacement(8, 18, 0))
    class BrokenPlanner:
        def plan(self, game, target):
            if failure == 'unknown':
                return replace(TacticalPlanner().plan(game, target), status='UNKNOWN')
            if failure == 'restoration':
                def fail_restore(*args, **kwargs):
                    raise RuntimeError('injected restoration failure')
                monkeypatch.setattr(game, 'set_state', fail_restore)
                return TacticalPlanner().plan(game, target)
            raise (TimeoutError if failure == 'timeout' else RuntimeError)('injected planner failure')
    result = run_formal_random(env, FormalRandomConfig(3, 42, 9, 5),
                               planner=BrokenPlanner(), output_dir=tmp_path / failure)
    assert result['status'] == 'system_error'
    assert result['attempted_episode_count'] == result['system_failure_count'] == 1
    assert result['completed_episode_count'] == result['completed_decisions'] == 0
    assert result['decision_count'] == result['planner_call_count'] == len(calls) == 1
    assert result['unreachable_count'] == result['episode_return'] == 0
    assert result['reward_totals']['unreachable'] == 0
    assert result['diagnostic']
    if failure == 'restoration':
        assert 'restoration failure' in result['diagnostic']
    assert read_json(tmp_path / failure / 'status.json')['status'] == 'system_error'


@pytest.mark.parametrize('failure', ['raise', *list(ExecutionFailure)])
def test_executor_failures_never_count_as_unreachable(env, monkeypatch, tmp_path, failure):
    setup_reset(monkeypatch, env, lambda game: select_piece(game, 1))
    fixed_choice(monkeypatch, TargetPlacement(8, 18, 0))
    class BrokenExecutor(Executor):
        def execute(self, *args):
            if failure == 'raise':
                raise RuntimeError('injected executor exception')
            return replace(super().execute(*args), success=False, failure_reason=failure,
                           error_message='injected execution failure')
    result = run_formal_random(env, FormalRandomConfig(3, 42, 2, 2),
                               executor=BrokenExecutor(), output_dir=tmp_path / 'error')
    assert result['status'] == 'system_error' and result['system_failure_count'] == 1
    assert result['decision_count'] == 1 and result['completed_decisions'] == 0
    assert result['pieces_placed'] == result['unreachable_count'] == result['episode_return'] == 0
    row, = records(tmp_path / 'error/episodes.jsonl')
    assert row['stop_reason'] == 'EXECUTION_ERROR'


def test_environment_transition_error_preserves_actual_lines_but_no_reward(env, monkeypatch, tmp_path):
    def setup(game):
        select_piece(game, 1)
        game.board[18:20, 4:14] = 2
        game.board[18:20, 8:10] = 0
    setup_reset(monkeypatch, env, setup)
    fixed_choice(monkeypatch, TargetPlacement(8, 18, 0))
    class BrokenWrapper(gym.Wrapper):
        def step(self, action):
            self.env.step(action)
            raise RuntimeError('after actual clear')
    summary = run_formal_random(BrokenWrapper(env), FormalRandomConfig(2, 42, 3, 1),
                                output_dir=tmp_path / 'error')
    assert summary['status'] == 'system_error'
    assert summary['lines_cleared'] == 2
    assert summary['episode_return'] == summary['unreachable_count'] == 0
    assert summary['completed_decisions'] == summary['pieces_placed'] == 0


@pytest.mark.parametrize('rows', [0, 1, 2])
def test_native_line_clear_and_watchdog_minimal_reward(env, monkeypatch, tmp_path, rows):
    def setup(game):
        select_piece(game, 1)
        game.board[20 - rows:20, 4:14] = 2
        game.board[20 - rows:20, 8:10] = 0
        game.score = lambda _: 12345
    setup_reset(monkeypatch, env, setup)
    fixed_choice(monkeypatch, TargetPlacement(8, 18, 0))
    summary = run_formal_random(env, FormalRandomConfig(1, 42, 3, 1), output_dir=tmp_path / 'lines')
    assert summary['status'] == 'completed'
    assert summary['lines_cleared'] == summary['episode_return'] == rows
    assert summary['pieces_placed'] == summary['reachable_decisions'] == 1
    assert summary['decisions_per_placement'] == 1
    assert summary['truncated_count'] == 1 and summary['terminated_count'] == 0
    assert summary['reward_totals']['death'] == 0


@pytest.mark.parametrize('mode', ['minimal', 'shaped'])
def test_native_successful_placement_game_over_is_normal_termination(env, monkeypatch, tmp_path, mode):
    def setup(game):
        select_piece(game, 1)
        game.board[2, 8:10] = 2
    setup_reset(monkeypatch, env, setup)
    fixed_choice(monkeypatch, TargetPlacement(8, 0, 0))
    transitions = capture_transitions(monkeypatch)
    summary = run_formal_random(env, FormalRandomConfig(2, 42, 3, 5, mode),
                                output_dir=tmp_path / 'death')
    assert summary['completed_episode_count'] == summary['terminated_count'] == 2
    assert summary['status'] == 'completed' and summary['truncated_count'] == 0
    assert summary['pieces_placed'] == 2 and summary['reward_totals']['death'] == -2
    assert all(t.terminated and not t.truncated for t in transitions)
    assert all(row['stop_reason'] == 'TERMINATED' for row in records(tmp_path / 'death/episodes.jsonl'))
    shaping = sum(-potential(t.state) for t in transitions) if mode == 'shaped' else 0
    assert summary['episode_return'] == pytest.approx(-2 + shaping)


def test_denominator_streak_and_planner_aggregates(env, monkeypatch, tmp_path):
    outcomes = [PlannerStatus.REACHABLE, PlannerStatus.UNREACHABLE, PlannerStatus.UNREACHABLE,
                PlannerStatus.UNREACHABLE, PlannerStatus.REACHABLE, PlannerStatus.UNREACHABLE]
    class ControlledPlanner:
        def plan(self, game, target):
            index = 6 - len(outcomes)
            return SimpleNamespace(status=outcomes.pop(0), elapsed_seconds=(index + 1) / 10)
    class VerifiedExecutor:
        def execute(self, *args):
            return SimpleNamespace(success=True, target_verified=True, terminated=False, truncated=False)
    summary = run_formal_random(env, FormalRandomConfig(1, 42, 3, 6), planner=ControlledPlanner(),
                                executor=VerifiedExecutor(), output_dir=tmp_path / 'sequence')
    assert summary['decision_count'] == summary['completed_decisions'] == 6
    assert summary['pieces_placed'] == 2 and summary['unreachable_count'] == 4
    assert summary['unreachable_rate'] == pytest.approx(4 / 6)
    assert summary['decisions_per_placement'] == 3
    assert summary['max_consecutive_unreachable'] == summary['mean_max_consecutive_unreachable'] == 3
    assert summary['planner_call_count'] == 6
    assert summary['planner_elapsed_seconds'] == pytest.approx(2.1)
    assert summary['planner_average_seconds'] == pytest.approx(.35)
    assert summary['planner_p95_seconds'] == summary['planner_max_seconds'] == .6
    assert summary['episode_return'] == pytest.approx(-.2)
    row, = records(tmp_path / 'sequence/episodes.jsonl')
    assert row['consecutive_unreachable'] == 1


def test_diagnostic_attempt_stays_in_rate_denominator(env, monkeypatch):
    class Planner:
        calls = 0
        def plan(self, *args):
            self.calls += 1
            if self.calls == 2:
                raise TimeoutError('timeout after unreachable')
            return SimpleNamespace(status=PlannerStatus.UNREACHABLE, elapsed_seconds=.1)
    result = run_formal_random(env, FormalRandomConfig(1, 42, 9, 5), planner=Planner())
    assert result['decision_count'] == 2 and result['completed_decisions'] == 1
    assert result['unreachable_count'] == 1 and result['unreachable_rate'] == .5
    assert result['episode_return'] == -.05


def test_no_learning_replay_optimizer_epsilon_checkpoint_or_global_rng_mutation(env, monkeypatch, tmp_path):
    from tetris_agent.rl.artifacts import RunArtifacts
    from tetris_agent.rl.dqn import DQNTrainer
    from tetris_agent.rl.policy import EpsilonGreedyPolicy
    from tetris_agent.rl.replay import ReplayBuffer
    def forbidden(*args, **kwargs):
        pytest.fail('A formal random baseline cannot learn or checkpoint')
    monkeypatch.setattr(DQNTrainer, '__init__', forbidden)
    monkeypatch.setattr(ReplayBuffer, '__init__', forbidden)
    monkeypatch.setattr(ReplayBuffer, 'add', forbidden)
    monkeypatch.setattr(EpsilonGreedyPolicy, 'select', forbidden)
    monkeypatch.setattr(torch.optim.Adam, '__init__', forbidden)
    monkeypatch.setattr(torch.optim.Adam, 'step', forbidden)
    monkeypatch.setattr(torch.Tensor, 'backward', forbidden)
    monkeypatch.setattr(torch, 'save', forbidden)
    monkeypatch.setattr(RunArtifacts, 'checkpoint', forbidden)
    python_before, numpy_before, torch_before = random.getstate(), np.random.get_state(), torch.get_rng_state()
    result = run_formal_random(env, FormalRandomConfig(2, 42, 13, 2, log_every=1),
                               output_dir=tmp_path / 'no_learning')
    assert result['gradient_updates'] == result['replay_size'] == 0
    assert random.getstate() == python_before
    numpy_after = np.random.get_state()
    assert numpy_before[0] == numpy_after[0] and numpy_before[2:] == numpy_after[2:]
    np.testing.assert_array_equal(numpy_before[1], numpy_after[1])
    torch.testing.assert_close(torch.get_rng_state(), torch_before, rtol=0, atol=0)
    assert not list((tmp_path / 'no_learning').glob('*.pt'))
    assert read_json(tmp_path / 'no_learning/config.json')['learning'] is False


def test_multiple_episode_determinism_seed_progression_and_policy_rng_continuity(env, monkeypatch, tmp_path):
    reset_seeds = []
    original = env.reset
    def reset(*args, **kwargs):
        reset_seeds.append(kwargs['seed'])
        return original(*args, **kwargs)
    monkeypatch.setattr(env, 'reset', reset)
    sequences, policies = [], []
    class Traced(FormalRandomPolicy):
        def __init__(self, seed=None):
            super().__init__(seed)
            self.sequence = []
            policies.append(self)
            sequences.append(self.sequence)
        def select(self, state, mask):
            # Independent oracle checks one continuous RNG over changing masks.
            action = super().select(state, mask)
            self.sequence.append((action, np.flatnonzero(mask).tolist()))
            return action
    monkeypatch.setattr(baseline, 'FormalRandomPolicy', Traced)
    config = FormalRandomConfig(3, 2**32 - 1, 13, 3)
    first = run_formal_random(env, config, output_dir=tmp_path / 'first')
    # Advancing global random state must not affect the second run's owned RNG.
    random.random()
    second = run_formal_random(env, config, output_dir=tmp_path / 'second')
    assert reset_seeds == [2**32 - 1, 0, 1] * 2
    assert len(policies) == 2 and sequences[0] == sequences[1]
    oracle = Random(config.policy_seed)
    assert all(action == oracle.choice(ids) for action, ids in sequences[0])
    ignored = {'wall_seconds', 'decisions_per_second', 'placements_per_second', 'timestamp',
               'planner_elapsed_seconds', 'planner_average_seconds', 'planner_p95_seconds', 'planner_max_seconds'}
    def gameplay(row):
        return {key: value for key, value in row.items() if key not in ignored}
    assert gameplay(first) == gameplay(second)
    first_rows = records(tmp_path / 'first/episodes.jsonl')
    assert [gameplay(row) for row in first_rows] == [gameplay(row) for row in records(tmp_path / 'second/episodes.jsonl')]
    assert [row['env_seed'] for row in first_rows] == [2**32 - 1, 0, 1]
    assert all(row['policy_seed'] == 13 for row in first_rows)
    assert first['mean_episode_return'] == pytest.approx(sum(row['episode_return'] for row in first_rows) / 3)
    assert first['mean_lines'] == sum(row['lines_cleared'] for row in first_rows) / 3
    assert first['mean_pieces'] == sum(row['pieces_placed'] for row in first_rows) / 3


def test_policy_seeds_change_action_sequence_with_fixed_environment(env, monkeypatch):
    sequences = []
    class Traced(FormalRandomPolicy):
        def __init__(self, seed=None):
            super().__init__(seed)
            self.sequence = []
            sequences.append(self.sequence)
        def select(self, state, mask):
            action = super().select(state, mask)
            self.sequence.append(action)
            return action
    monkeypatch.setattr(baseline, 'FormalRandomPolicy', Traced)
    run_formal_random(env, FormalRandomConfig(1, 42, 13, 3))
    run_formal_random(env, FormalRandomConfig(1, 42, 14, 3))
    assert sequences[0] != sequences[1]


def test_episode_is_durable_before_next_reset_and_progress_parses(env, monkeypatch, tmp_path):
    original = env.reset
    seeds, durable = [], []
    output = tmp_path / 'durable'
    def reset(*args, **kwargs):
        if seeds:
            durable.extend(records(output / 'episodes.jsonl'))
        seeds.append(kwargs['seed'])
        return original(*args, **kwargs)
    monkeypatch.setattr(env, 'reset', reset)
    result = run_formal_random(env, FormalRandomConfig(2, 42, 13, 2, log_every=1), output_dir=output)
    assert len(durable) == 1 and durable[0]['episode_index'] == 1
    assert durable[0]['completed_decisions'] == 2
    assert read_json(output / 'summary.json') == result
    assert read_json(output / 'status.json')['status'] == 'completed'
    progress = records(output / 'metrics.jsonl')
    assert progress[0]['event'] == 'start' and progress[0]['status'] == 'running'
    assert progress[-1]['event'] == 'final' and progress[-1]['status'] == 'completed'
    assert [row['decision_count'] for row in progress if row['event'] == 'periodic'] == [1, 2, 3, 4]
    assert len([row for row in progress if row['event'] == 'episode_end']) == 2
    assert result['wall_seconds'] >= progress[-1]['wall_seconds']
    assert result['decisions_per_second'] == pytest.approx(4 / result['wall_seconds'])
    assert set(path.name for path in output.iterdir()) == {
        'config.json', 'episodes.jsonl', 'metrics.jsonl', 'summary.json', 'status.json'}


@pytest.mark.parametrize('exception,status', [(RuntimeError, 'system_error'), (KeyboardInterrupt, 'interrupted')])
def test_reset_failure_or_interrupt_keeps_completed_episodes(env, monkeypatch, tmp_path, exception, status):
    original = env.reset
    resets = []
    def reset(*args, **kwargs):
        resets.append(kwargs['seed'])
        if len(resets) == 2:
            raise exception('injected reset stop')
        return original(*args, **kwargs)
    monkeypatch.setattr(env, 'reset', reset)
    output = tmp_path / status
    with pytest.raises(exception):
        run_formal_random(env, FormalRandomConfig(3, 42, 13, 1), output_dir=output)
    assert read_json(output / 'status.json')['status'] == status
    rows = records(output / 'episodes.jsonl')
    assert len(rows) == 2 and rows[0]['diagnostic'] is None and rows[1]['diagnostic']
    summary = read_json(output / 'summary.json')
    assert summary['completed_episode_count'] == 1 and summary['attempted_episode_count'] == 2
    assert summary['decision_count'] == 1
    assert summary['system_failure_count'] == int(exception is RuntimeError)


def test_final_persistence_failure_never_leaves_success_status(env, monkeypatch, tmp_path):
    original = baseline._Artifacts._json
    def json_write(self, name, payload):
        if name == 'summary.json':
            raise OSError('injected summary disk failure')
        return original(self, name, payload)
    monkeypatch.setattr(baseline._Artifacts, '_json', json_write)
    with pytest.raises(OSError, match='disk failure'):
        run_formal_random(env, FormalRandomConfig(1, 42, 13, 1), output_dir=tmp_path / 'disk')
    assert read_json(tmp_path / 'disk/status.json')['status'] == 'artifact_error'
    assert len(records(tmp_path / 'disk/episodes.jsonl')) == 1


def test_periodic_persistence_failure_is_diagnostic_without_success(env, monkeypatch, tmp_path):
    original = baseline._Artifacts.log
    def log(self, row, event):
        if event == 'periodic':
            raise OSError('injected metrics disk failure')
        return original(self, row, event)
    monkeypatch.setattr(baseline._Artifacts, 'log', log)
    with pytest.raises(OSError, match='disk failure'):
        run_formal_random(env, FormalRandomConfig(2, 42, 13, 2, log_every=1), output_dir=tmp_path / 'disk')
    assert read_json(tmp_path / 'disk/status.json')['status'] == 'artifact_error'
    assert read_json(tmp_path / 'disk/summary.json')['status'] == 'artifact_error'


def test_initial_persistence_failure_records_error_and_never_resets_env(env, monkeypatch, tmp_path):
    original = baseline._Artifacts._json
    def json_write(self, name, payload):
        if name == 'config.json':
            raise OSError('injected config disk failure')
        return original(self, name, payload)
    monkeypatch.setattr(baseline._Artifacts, '_json', json_write)
    monkeypatch.setattr(env, 'reset', lambda **kwargs: pytest.fail('Must not interact after artifact setup failure'))
    with pytest.raises(OSError, match='disk failure'):
        run_formal_random(env, FormalRandomConfig(1, 42, 13), output_dir=tmp_path / 'disk')
    assert read_json(tmp_path / 'disk/status.json')['status'] == 'artifact_error'


@pytest.mark.parametrize('failure', ['state', 'policy'])
def test_state_and_policy_errors_stop_without_planner_or_reward(env, monkeypatch, tmp_path, failure):
    if failure == 'state':
        setup_reset(monkeypatch, env, lambda game: game.step(game.actions.move_right))
    else:
        control_choice(monkeypatch, lambda ids: next(action for action in range(800) if action not in ids))
    class NeverPlan:
        def plan(self, *args):
            pytest.fail('Invalid state/choice must not reach the planner')
    summary = run_formal_random(env, FormalRandomConfig(2, 42, 13, 3), planner=NeverPlan(),
                                output_dir=tmp_path / failure)
    assert summary['status'] == 'system_error'
    assert summary['decision_count'] == int(failure == 'policy')
    assert summary['unreachable_count'] == summary['completed_decisions'] == 0
    assert summary['planner_call_count'] == summary['episode_return'] == 0
    assert summary['planner_average_seconds'] is None


def test_bad_output_parent_file_fails_without_environment_interaction(env, monkeypatch, tmp_path):
    parent = tmp_path / 'file'
    parent.write_text('preserve parent')
    monkeypatch.setattr(env, 'reset', lambda **kwargs: pytest.fail('Must not reset after output rejection'))
    with pytest.raises(OSError):
        run_formal_random(env, FormalRandomConfig(1, 42, 13), output_dir=parent / 'run')
    assert parent.read_text() == 'preserve parent'


@pytest.mark.parametrize('kind', ['empty_directory', 'file', 'nonempty_directory'])
def test_output_existing_path_is_rejected_without_mutation(env, tmp_path, kind):
    output = tmp_path / 'existing'
    if kind == 'file':
        output.write_text('preserve me')
    else:
        output.mkdir()
        if kind == 'nonempty_directory':
            (output / 'marker.txt').write_text('preserve me')
    with pytest.raises(FileExistsError):
        run_formal_random(env, FormalRandomConfig(1, 42, 13, 1), output_dir=output)
    if kind == 'file':
        assert output.read_text() == 'preserve me'
    elif kind == 'nonempty_directory':
        assert (output / 'marker.txt').read_text() == 'preserve me'
    else:
        assert not list(output.iterdir())


@pytest.mark.parametrize('field,value', [
    ('episodes', 0), ('episodes', -1), ('episodes', True), ('episodes', 1.5),
    ('max_decisions_per_episode', 0), ('max_decisions_per_episode', True),
    ('log_every', 0), ('log_every', 1.5), ('env_seed', -1), ('env_seed', 2**32),
    ('env_seed', True), ('env_seed', 1.5), ('policy_seed', -1), ('policy_seed', 2**32),
    ('policy_seed', True), ('reward_mode', 'new_reward'),
])
def test_config_rejects_invalid_values(field, value):
    values = dict(episodes=1, env_seed=42, policy_seed=13)
    values[field] = value
    with pytest.raises(ValueError):
        FormalRandomConfig(**values)


def load_cli():
    spec = importlib.util.spec_from_file_location('formal_random_cli',
            Path(__file__).parents[1] / 'scripts/run_formal_random.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('flag,value', [('--episodes', '0'), ('--episodes', '-1'),
    ('--episodes', 'abc'), ('--env-seed', '-1'), ('--policy-seed', str(2**32)),
    ('--policy-seed', 'abc'), ('--max-decisions-per-episode', '0'), ('--log-every', '0')])
def test_cli_invalid_arguments(flag, value, tmp_path):
    args = ['--episodes', '1', '--env-seed', '42', '--policy-seed', '13', '--output-dir', str(tmp_path / 'new')]
    if flag in args:
        args[args.index(flag) + 1] = value
    else:
        args.extend([flag, value])
    with pytest.raises(SystemExit) as exc:
        load_cli().build_parser().parse_args(args)
    assert exc.value.code == 2


def test_cli_defaults_and_existing_output_rejection(tmp_path):
    cli = load_cli()
    args = ['--episodes', '2', '--env-seed', '42', '--policy-seed', '13', '--output-dir', str(tmp_path)]
    options = cli.build_parser().parse_args(args)
    assert options.reward == 'minimal' and options.max_decisions_per_episode == 10000
    assert options.log_every == 100
    with pytest.raises(SystemExit) as exc:
        cli.main(args)
    assert exc.value.code == 2


@pytest.mark.parametrize('status,code', [('completed', 0), ('system_error', 1)])
def test_cli_uses_native_env_and_closes_it_on_diagnostic(monkeypatch, tmp_path, status, code, capsys):
    cli = load_cli()
    instances = []
    class Env:
        def __init__(self, *, gravity):
            assert gravity
            self.closed = False
            instances.append(self)
        def close(self):
            self.closed = True
    def run(game, config, *, output_dir):
        assert config == FormalRandomConfig(2, 42, 13)
        assert output_dir == tmp_path / 'new'
        return dict(status=status)
    monkeypatch.setattr(cli, 'Tetris', Env)
    monkeypatch.setattr(cli, 'run_formal_random', run)
    assert cli.main(['--episodes', '2', '--env-seed', '42', '--policy-seed', '13',
                     '--output-dir', str(tmp_path / 'new')]) == code
    assert instances[0].closed
    assert json.loads(capsys.readouterr().out)['status'] == status
