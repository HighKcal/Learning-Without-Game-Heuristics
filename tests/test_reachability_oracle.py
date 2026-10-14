"""Physical equivalence, ownership, acting, bootstrap and abort boundaries."""
from dataclasses import replace
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
import torch
from tetris_gymnasium.envs.tetris import Tetris

from tetris_agent import TacticalPlanner, TargetPlacement, TargetSpace, TetrisAdapter
from tetris_agent.planner import EnumerationStatus
from tetris_agent.rl.config import TrainingConfig
from tetris_agent.rl.dqn import DQNTrainer
from tetris_agent.rl.evaluation import load_dataset, evaluate_network
from tetris_agent.rl.oracle import (compute_boundary, restore_semantic_boundary,
                                    OracleInvariantError)
from tetris_agent.rl.policy import EpsilonGreedyPolicy, FormalRandomPolicy
from tetris_agent.rl.replay import Transition, ReplayBuffer
from tetris_agent.rl.runner import RLRunner
from tetris_agent.rl.state import SPACE, geometric_mask, read_state
from test_planner import select_piece, state_signature


@pytest.fixture
def env():
    game = Tetris(gravity=True)
    game.reset(seed=42)
    yield game
    game.close()


@pytest.mark.parametrize('index,count', list(enumerate((17, 9, 34, 17, 17, 34, 34))))
def test_empty_census_exact_paths_and_determinism(env, index, count):
    select_piece(env, index)
    planner = TacticalPlanner()
    before = state_signature(env)
    first, second = compute_boundary(env, planner), compute_boundary(env, planner)
    assert state_signature(env) == before
    assert first.search.status == EnumerationStatus.COMPLETE
    assert first.search.expanded_states == first.search.visited_states > 0
    assert first.geometric_count == first.mask.sum() == count
    assert sha256(first.mask.tobytes()).digest() == sha256(second.mask.tobytes()).digest()
    for target, witness in first.search.placements:
        plan = planner.plan(env, target)
        assert plan.reachable
        assert (plan.actions, plan.needs_hard_drop) == (witness.actions, witness.needs_hard_drop)
    with pytest.raises(ValueError):
        first.mask.setflags(write=True)


@pytest.mark.parametrize('gravity', [True, False])
@pytest.mark.parametrize('case', ['cavity', 'tuck', 'floor', 'clear', 'rotated_i'])
def test_exhaustive_geometry_and_planner_equivalence(env, gravity, case):
    env.gravity_enabled = gravity
    select_piece(env, 1)
    if case == 'cavity':
        env.board[17, 4:14] = 2
    elif case == 'tuck':
        env.board[16, 8:10] = 2
    elif case == 'floor':
        env.x, env.y = 7, 18
    elif case == 'clear':
        env.board[19, 4:14] = 2
        env.board[19, 8:10] = 0
    else:
        select_piece(env, 0, orientation=2)
    planner = TacticalPlanner()
    space = TargetSpace(TetrisAdapter(env).geometry)
    geometric = space.action_mask(env.board, TetrisAdapter(env).read_current_piece().spawn_matrix)
    before = state_signature(env)
    search = planner.enumerate_reachable_targets(env)
    reachable = set(search.targets)
    for action in np.flatnonzero(geometric):
        target = space.decode(int(action))
        assert planner.plan(env, target).reachable == (target in reachable)
    assert all(geometric[space.encode(t)] for t in reachable)
    assert state_signature(env) == before


def cavity(env):
    select_piece(env, 1)
    env.board[17, 4:14] = 2
    state = read_state(env)
    boundary = compute_boundary(env, TacticalPlanner())
    unreachable = int(np.flatnonzero(geometric_mask(state) & ~boundary.mask)[0])
    reachable = int(np.flatnonzero(boundary.mask)[0])
    return state, boundary, reachable, unreachable


def biased_network(trainer, boundary, unreachable):
    with torch.no_grad():
        for net, value in ((trainer.online, 1), (trainer.target, 2)):
            for parameter in net.parameters():
                parameter.zero_()
            net[-1].bias[torch.tensor(boundary.mask.copy())] = value
            net[-1].bias[unreachable] = 100000


def test_acting_random_greedy_and_real_bellman_update(env):
    state, boundary, reachable, unreachable = cavity(env)
    trainer = DQNTrainer(TrainingConfig(decision_budget=1, device='cpu', batch_size=1,
                         learning_starts=1, reachability_mask_mode='oracle'))
    biased_network(trainer, boundary, unreachable)
    policy = EpsilonGreedyPolicy(trainer.online, 3)
    assert boundary.mask[policy.select(state, boundary.mask, 0)]
    assert all(boundary.mask[policy.select(state, boundary.mask, 1)] for _ in range(2000))
    transition = Transition(state, reachable, 0, state, False, True, boundary.mask, boundary.mask)
    metrics = trainer.observe(transition)
    assert metrics.mean_abs_td_error == pytest.approx(.98, abs=1e-5)
    assert metrics.loss == pytest.approx(.5 * .98 ** 2, abs=1e-5)


def test_mask_deep_ownership_and_missing_or_invalid_replay_abort(env):
    state, boundary, reachable, unreachable = cavity(env)
    current, following = boundary.mask.copy(), boundary.mask.copy()
    transition = Transition(state, reachable, 0, state, False, False, current, following)
    current[:] = following[:] = False
    replay = ReplayBuffer(3, 1)
    replay.add(transition)
    sample, = replay.sample(1)
    assert sample.state_reachable_mask[reachable] and sample.next_state_reachable_mask[reachable]
    assert not np.shares_memory(sample.state_reachable_mask, transition.state_reachable_mask)
    with pytest.raises(ValueError):
        sample.next_state_reachable_mask.setflags(write=True)
    trainer = DQNTrainer(TrainingConfig(1, device='cpu', reachability_mask_mode='oracle'))
    for bad in (Transition(state, reachable, 0, state, False, False),
                replace(transition, action_id=unreachable),
                replace(transition, next_state_reachable_mask=None)):
        with pytest.raises(OracleInvariantError):
            trainer.observe(bad)
    assert len(trainer.replay) == trainer.stored_experiences == 0


def test_runner_reuses_next_boundary_no_per_action_search(env):
    class Counted(TacticalPlanner):
        calls = 0
        def enumerate_reachable_targets(self, env):
            self.calls += 1
            return super().enumerate_reachable_targets(env)
        def plan(self, *args):
            pytest.fail('Oracle acting must reuse the one search witness.')
    planner, transitions = Counted(), []
    metrics = RLRunner(FormalRandomPolicy(5), planner=planner, reachability_mask_mode='oracle').run(
        env, reset=False, max_decisions=3, on_transition=transitions.append)
    assert metrics.diagnostic is None and metrics.unreachable_count == metrics.selected_physical_u_count == 0
    assert planner.calls == metrics.oracle_computation_count == 4
    assert len(transitions) == 3 and transitions[-1].truncated
    for previous, following in zip(transitions, transitions[1:]):
        np.testing.assert_array_equal(previous.next_state_reachable_mask, following.state_reachable_mask)
    assert metrics.geometric_valid_action_count >= metrics.reachable_action_count > 0


def test_terminal_no_next_oracle_or_bootstrap(env):
    select_piece(env, 1)
    env.board[2, 4:14] = 2
    action = SPACE.encode(TargetPlacement(8, 0, 0))
    class Fixed:
        def select(self, state, mask):
            assert mask[action]
            return action
    transitions = []
    metrics = RLRunner(Fixed(), reachability_mask_mode='oracle').run(
        env, reset=False, max_decisions=1, on_transition=transitions.append)
    assert metrics.diagnostic is None and metrics.terminated and metrics.oracle_computation_count == 1
    transition, = transitions
    assert transition.next_state_reachable_mask is None
    trainer = DQNTrainer(TrainingConfig(1, device='cpu', batch_size=1, learning_starts=1,
                                      reachability_mask_mode='oracle'))
    with torch.no_grad():
        for p in trainer.online.parameters():
            p.zero_()
        for p in trainer.target.parameters():
            p.fill_(100)
    assert trainer.observe(transition).mean_abs_td_error == pytest.approx(abs(transition.reward))


@pytest.mark.parametrize('failure', ['exception', 'timeout', 'unknown', 'next_exception', 'execution'])
def test_uncertainty_never_enters_replay(env, failure):
    class Failing(TacticalPlanner):
        calls = 0
        def enumerate_reachable_targets(self, game):
            self.calls += 1
            if failure == 'unknown':
                return SimpleNamespace(status='UNKNOWN')
            if failure == 'timeout':
                raise TimeoutError('injected')
            if failure == 'exception' or (failure == 'next_exception' and self.calls == 2):
                raise RuntimeError('injected')
            return super().enumerate_reachable_targets(game)
    class BrokenExecutor:
        def execute(self, *args):
            raise RuntimeError('injected')
    transitions = []
    metrics = RLRunner(FormalRandomPolicy(3), planner=Failing(),
        executor=BrokenExecutor() if failure == 'execution' else None,
        reachability_mask_mode='oracle').run(env, reset=False, max_decisions=1,
                                            on_transition=transitions.append)
    assert metrics.diagnostic and not transitions and metrics.stored_transitions == 0


def test_native_search_exception_restores_original(env, monkeypatch):
    step = env.step
    def broken(action):
        step(action)
        raise RuntimeError('native search failed after mutation')
    monkeypatch.setattr(env, 'step', broken)
    before = state_signature(env)
    with pytest.raises(RuntimeError):
        TacticalPlanner().enumerate_reachable_targets(env)
    assert state_signature(env) == before


def test_fixed_corpus_exhaustive_small_sample_and_oracle_evaluation(env):
    records, _ = load_dataset(Path(__file__).parents[1] / 'evaluation/fixed_eval_final_v1')
    sample = [records[i] for i in (0, 13, 20, 70, 128, 255)]
    planner = TacticalPlanner()
    for record in sample:
        restore_semantic_boundary(env, record.state)
        boundary = compute_boundary(env, planner)
        assert set(np.flatnonzero(boundary.mask)) == set(record.reachable_action_ids)
        for action in record.valid_action_ids:
            assert planner.plan(env, SPACE.decode(action)).reachable == bool(boundary.mask[action])
    trainer = DQNTrainer(TrainingConfig(1, device='cpu'))
    result = evaluate_network(trainer.online, sample, reachability_mask_mode='oracle')
    assert result['greedy_first_choice_unreachable_rate'] == 0


@pytest.mark.parametrize('kwargs', [dict(algorithm='vanilla'), dict(reward_mode='shaped')])
def test_clean_experiment_configuration(kwargs):
    with pytest.raises(ValueError, match='Clean oracle'):
        TrainingConfig(1, reachability_mask_mode='oracle', **kwargs)


def test_primitive_wrapper_interruption_aborts(env):
    wrapped = gym.wrappers.TimeLimit(env, max_episode_steps=1)
    wrapped.reset(seed=42)
    select_piece(env, 1)
    class Fixed:
        def select(self, state, mask):
            return SPACE.encode(TargetPlacement(4, 18, 0))
    transitions = []
    metrics = RLRunner(Fixed(), reachability_mask_mode='oracle').run(
        wrapped, reset=False, max_decisions=1, on_transition=transitions.append)
    assert metrics.stop_reason == 'EXECUTION_ERROR' and not transitions


def test_selected_u_is_visible_invariant_failure(env):
    state, boundary, reachable, unreachable = cavity(env)
    class Disobedient:
        def select(self, state, mask):
            return unreachable
    transitions = []
    metrics = RLRunner(Disobedient(), reachability_mask_mode='oracle').run(
        env, reset=False, max_decisions=1, on_transition=transitions.append)
    assert metrics.stop_reason == 'ORACLE_INVARIANT_VIOLATION'
    assert metrics.selected_physical_u_count == 1 and not transitions


def test_default_mode_never_calls_oracle_and_stores_no_mask(env):
    class GeometricOnly(TacticalPlanner):
        def enumerate_reachable_targets(self, *args):
            pytest.fail('Default production behavior must stay geometric-only.')
    transitions = []
    metrics = RLRunner(FormalRandomPolicy(2), planner=GeometricOnly()).run(
        env, reset=False, max_decisions=2, on_transition=transitions.append)
    assert metrics.diagnostic is None and len(transitions) == 2
    assert metrics.oracle_computation_count == 0
    assert all(t.state_reachable_mask is None and t.next_state_reachable_mask is None for t in transitions)


def test_geometric_invalid_oracle_output_rejected(env):
    class Invalid(TacticalPlanner):
        def enumerate_reachable_targets(self, game):
            search = super().enumerate_reachable_targets(game)
            target, witness = search.placements[0]
            invalid = TargetPlacement(4, 0, 0)
            return replace(search, placements=search.placements + ((invalid, witness),))
    with pytest.raises(OracleInvariantError, match='geometric-invalid'):
        compute_boundary(env, Invalid())
