"""Play fallback contracts plus real native Planner/Executor integration."""
from dataclasses import replace
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from tetris_agent import Executor, PlannerResult, PlannerStatus, TacticalPlanner, TargetPlacement
from tetris_agent.play import FrozenQRankingPolicy, PlayRunner
from tetris_agent.rl.network import QNetwork
from tetris_agent.rl.policy import EpsilonGreedyPolicy
from tetris_agent.rl.runner import RLRunner
from tetris_agent.rl.state import SPACE, geometric_mask, read_state
from test_planner import select_piece, state_signature
from test_rl_foundation import env, semantic


class CountingQ(torch.nn.Module):
    def __init__(self, values):
        super().__init__()
        self.q = torch.nn.Parameter(torch.full((800,), -1000.0))
        with torch.no_grad():
            for action, value in values.items():
                self.q[action] = value
        self.calls = 0

    def forward(self, states):
        self.calls += 1
        assert not torch.is_grad_enabled()
        return self.q[None].expand(len(states), -1)


def policy_for(targets):
    return FrozenQRankingPolicy(CountingQ({SPACE.encode(t): 100 - i for i, t in enumerate(targets)}))


class RejectThenPlan:
    def __init__(self, rejects):
        self.rejects = rejects
        self.calls = []
        self.plans = []

    def plan(self, game, target):
        self.calls.append(SPACE.encode(target))
        result = (PlannerResult(PlannerStatus.UNREACHABLE, (), False, 1, 1, .001)
                  if len(self.calls) <= self.rejects else TacticalPlanner().plan(game, target))
        self.plans.append(result)
        return result


class CheckedExecutor(Executor):
    def __init__(self, planner):
        self.planner = planner
        self.calls = 0

    def execute(self, game, target, plan):
        self.calls += 1
        assert plan is self.planner.plans[-1]  # Identity, not just equal actions.
        return super().execute(game, target, plan)


@pytest.mark.parametrize('rejects', [0, 1, 2])
def test_top1_rank2_rank3_and_exact_plan_reuse(env, rejects):
    select_piece(env, 1)
    targets = [TargetPlacement(x, 18, 0) for x in (7, 10, 5)]
    policy = policy_for(targets)
    planner = RejectThenPlan(rejects)
    executor = CheckedExecutor(planner)
    before = policy.network.q.clone()
    result = PlayRunner(policy, planner=planner, executor=executor).run(
        env, reset=False, max_placements=1)
    record, = result.decisions
    assert result.placements == 1 and result.diagnostic is None
    assert record['selected_rank'] == rejects + 1
    assert record['rejected_action_ids'] == [SPACE.encode(t) for t in targets[:rejects]]
    assert record['selected_action_id'] == SPACE.encode(targets[rejects])
    assert record['planner_attempts'] == len(planner.calls) == rejects + 1
    assert len(set(planner.calls)) == len(planner.calls)
    assert executor.calls == policy.network.calls == 1
    assert record['executor_result'] == 'SUCCESS' and record['target_verified']
    assert np.count_nonzero(env.board[:20, 4:14]) == 4
    assert torch.equal(before, policy.network.q) and policy.network.q.grad is None
    assert result.summary()['mean_reachable_rank'] == rejects + 1


def test_rejected_action_is_candidate_again_in_next_state_and_run(env):
    select_piece(env, 1)
    env.queue.queue[0] = 1  # Next O; the rejected left target stays valid after right lock.
    a, b = TargetPlacement(5, 18, 0), TargetPlacement(11, 18, 0)
    policy = policy_for([a, b])
    planner = RejectThenPlan(1)
    runner = PlayRunner(policy, planner=planner)
    result = runner.run(env, reset=False, max_placements=2)
    first, second = result.decisions
    assert result.diagnostic is None and result.placements == 2
    assert first['rejected_action_ids'] == [SPACE.encode(a)]
    assert second['selected_action_id'] == SPACE.encode(a)
    assert second['selected_rank'] == 1 and second['rejected_action_ids'] == []
    assert first['state_id'] != second['state_id']
    assert planner.calls == [SPACE.encode(a), SPACE.encode(b), SPACE.encode(a)]
    assert policy.network.calls == 2
    env.reset(seed=42)
    select_piece(env, 1)
    again = runner.run(env, reset=False, max_placements=1)
    assert again.decisions[0]['selected_action_id'] == SPACE.encode(a)
    assert again.decisions[0]['rejected_action_ids'] == []


@pytest.mark.parametrize('status', ['SEARCH_TIMEOUT', 'UNKNOWN', 'RESOURCE_ERROR', 'exception', 'timeout'])
def test_uncertainty_is_not_u_and_never_falls_back(env, status):
    select_piece(env, 1)
    class Uncertain:
        calls = 0
        def plan(self, game, target):
            self.calls += 1
            if status == 'exception':
                raise RuntimeError('injected error')
            if status == 'timeout':
                raise TimeoutError('search budget expired')
            return PlannerResult(status, (), False, 0, 0, 0)
    planner = Uncertain()
    policy = policy_for([TargetPlacement(8, 18, 0)])
    before = state_signature(env)
    result = PlayRunner(policy, planner=planner).run(env, reset=False, max_placements=2)
    record, = result.decisions
    assert result.diagnostic and result.placements == 0
    assert result.stop_reason in ('PLANNER_UNKNOWN', 'PLANNER_ERROR', 'SEARCH_TIMEOUT')
    assert record['rejected_action_ids'] == [] and record['rejected_u_count'] == 0
    assert record['executor_result'] == 'NOT_RUN' and planner.calls == 1
    assert state_signature(env) == before


@pytest.mark.parametrize('raising', [False, True])
def test_execution_failure_stops_without_next_candidate_even_after_mutation(env, raising):
    select_piece(env, 1)
    class BrokenExecutor(CheckedExecutor):
        def execute(self, *args):
            result = super().execute(*args)
            if raising:
                raise RuntimeError('after real mutation')
            return replace(result, success=False, error_message='failed verified execution')
    planner = RejectThenPlan(0)
    executor = BrokenExecutor(planner)
    result = PlayRunner(policy_for([TargetPlacement(8, 18, 0)]),
                        planner=planner, executor=executor).run(env, reset=False, max_placements=2)
    record, = result.decisions
    assert result.stop_reason == 'EXECUTION_ERROR' and result.diagnostic
    assert result.placements == 0 and len(planner.calls) == executor.calls == 1
    assert record['rejected_u_count'] == 0 and record['executor_result'] == 'EXECUTION_ERROR'
    assert 'clear_filled_rows' not in vars(env)
    assert np.count_nonzero(env.board[:20, 4:14]) == 4


def test_all_valid_candidates_u_is_finite_and_unbounded_by_arbitrary_depth(env):
    select_piece(env, 2)
    planner = RejectThenPlan(800)
    result = PlayRunner(policy_for([]), planner=planner).run(env, reset=False, max_placements=1)
    record, = result.decisions
    count = int(geometric_mask(read_state(env)).sum())
    assert count > 10
    assert result.stop_reason == 'NO_REACHABLE_TARGET'
    assert record['geometric_valid_count'] == record['tested_count'] == len(planner.calls) == count
    assert len(set(planner.calls)) == count
    assert record['result_histogram'] == {'UNREACHABLE': count}
    assert record['rejected_u_count'] == count and record['executor_result'] == 'NOT_RUN'


def test_real_physical_u_fallback_restores_and_executes(env):
    select_piece(env, 1)
    env.board[17, 4:14] = 2
    u, r = TargetPlacement(7, 18, 0), TargetPlacement(7, 15, 0)
    class AuditedPlanner(TacticalPlanner):
        calls = 0
        def plan(self, game, target):
            before = state_signature(game)
            self.calls += 1
            result = super().plan(game, target)
            assert state_signature(game) == before
            return result
    planner = AuditedPlanner()
    result = PlayRunner(policy_for([u, r]), planner=planner).run(env, reset=False, max_placements=1)
    record, = result.decisions
    assert result.diagnostic is None and result.placements == 1
    assert record['rejected_action_ids'] == [SPACE.encode(u)]
    assert record['selected_action_id'] == SPACE.encode(r) and record['selected_rank'] == 2
    assert planner.calls == 2 and record['target_verified']


def test_research_keeps_top1_u_self_loops_and_geometric_mask(env):
    select_piece(env, 1)
    env.board[17, 4:14] = 2
    u, r = TargetPlacement(7, 18, 0), TargetPlacement(7, 15, 0)
    network = QNetwork()
    with torch.no_grad():
        for parameter in network.parameters():
            parameter.zero_()
        network[-1].bias[SPACE.encode(u)] = 10
        network[-1].bias[SPACE.encode(r)] = 9
    before = state_signature(env)
    transitions = []
    result = RLRunner(EpsilonGreedyPolicy(network)).run(
        env, reset=False, max_decisions=2, on_transition=transitions.append)
    assert result.diagnostic is None and result.unreachable_count == 2
    assert result.pieces_placed == 0 and result.planner_call_count == 2
    assert state_signature(env) == before
    assert [t.action_id for t in transitions] == [SPACE.encode(u)] * 2


@pytest.mark.parametrize('rows', [0, 1, 2])
def test_play_counts_real_lines_and_restores_existing_observer(env, rows):
    select_piece(env, 1)
    env.board[20 - rows:20, 4:14] = 2
    env.board[20 - rows:20, 8:10] = 0
    original = env.clear_filled_rows
    env.clear_filled_rows = original  # An existing instance hook must be retained.
    result = PlayRunner(policy_for([TargetPlacement(8, 18, 0)])).run(
        env, reset=False, max_placements=1)
    assert result.diagnostic is None and result.lines == rows
    assert result.decisions[0]['lines'] == rows and result.placements == 1
    assert env.clear_filled_rows is original


def test_play_verified_lock_can_clear_line_and_die(env):
    select_piece(env, 1)
    env.board[0, 7] = 2
    env.board[19, 4:14] = 2
    env.board[19, 8:10] = 0
    env.queue.queue[0] = 0
    result = PlayRunner(policy_for([TargetPlacement(8, 18, 0)])).run(
        env, reset=False, max_placements=2)
    assert result.diagnostic is None and result.terminated
    assert result.stop_reason == 'TERMINATED' and result.placements == result.lines == 1
    assert result.summary()['deaths'] == 1 and result.decisions[0]['target_verified']


def test_timeout_after_confirmed_u_preserves_only_prior_reject(env):
    select_piece(env, 1)
    class RejectThenTimeout(RejectThenPlan):
        def plan(self, game, target):
            if self.calls:
                self.calls.append(SPACE.encode(target))
                raise TimeoutError('second candidate timeout')
            return super().plan(game, target)
    first, second = TargetPlacement(7, 18, 0), TargetPlacement(10, 18, 0)
    planner = RejectThenTimeout(1)
    result = PlayRunner(policy_for([first, second]), planner=planner).run(
        env, reset=False, max_placements=1)
    record, = result.decisions
    assert result.stop_reason == 'SEARCH_TIMEOUT' and len(planner.calls) == 2
    assert record['rejected_action_ids'] == [SPACE.encode(first)]
    assert record['result_histogram'] == {'UNREACHABLE': 1, 'SEARCH_TIMEOUT': 1}
    assert record['executor_result'] == 'NOT_RUN'


def test_ranking_geometric_only_ties_and_one_forward():
    mask = np.zeros(800, dtype=bool)
    mask[[4, 17, 35]] = True
    network = CountingQ({0: 100, 4: 5, 17: 5, 35: 1})
    policy = FrozenQRankingPolicy(network)
    assert policy.rank(semantic(), mask) == ((4, 5.), (17, 5.), (35, 1.))
    assert network.calls == 1 and not network.training and not network.q.requires_grad


def test_corrupted_planner_state_is_not_confirmed_u(env):
    select_piece(env, 1)
    class Corrupt:
        calls = 0
        def plan(self, game, target):
            self.calls += 1
            game.board[19, 4] = 2
            return PlannerResult(PlannerStatus.UNREACHABLE, (), False, 0, 0, 0)
    planner = Corrupt()
    result = PlayRunner(policy_for([TargetPlacement(8, 18, 0)]), planner=planner).run(
        env, reset=False, max_placements=2)
    assert result.stop_reason == 'PLANNER_ERROR' and planner.calls == 1
    assert result.decisions[0]['rejected_action_ids'] == []


def test_invalid_state_and_nonfinite_q_stop_before_planner(env):
    policy = policy_for([TargetPlacement(8, 18, 0)])
    env.step(env.actions.move_right)
    result = PlayRunner(policy).run(env, reset=False)
    assert result.stop_reason == 'STATE_ERROR' and policy.network.calls == 0
    env.reset(seed=42)
    select_piece(env, 1)
    with torch.no_grad():
        policy.network.q[SPACE.encode(TargetPlacement(8, 18, 0))] = float('nan')
    result = PlayRunner(policy).run(env, reset=False)
    assert result.stop_reason == 'POLICY_ERROR' and result.decisions[0]['planner_attempts'] == 0


@pytest.mark.parametrize('on_lock', [False, True])
def test_wrapper_truncation_obeys_executor_verification(env, on_lock):
    import gymnasium as gym
    wrapped = gym.wrappers.TimeLimit(env, max_episode_steps=1)
    wrapped.reset(seed=42)
    select_piece(env, 1)
    result = PlayRunner(policy_for([TargetPlacement(8 if on_lock else 10, 18, 0)])).run(
        wrapped, reset=False, max_placements=2)
    assert result.truncated
    assert result.placements == int(on_lock)
    assert result.stop_reason == ('TRUNCATED' if on_lock else 'EXECUTION_ERROR')


@pytest.mark.parametrize('mode', ['research', 'play'])
def test_cli_frozen_load_and_mode(tmp_path, mode):
    spec = importlib.util.spec_from_file_location('play_cli',
        Path(__file__).parents[1] / 'scripts/play_agent.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    checkpoint = tmp_path / 'frozen.pt'
    torch.save(dict(schema_version=1,
                    checkpoint_contract=dict(kind='training_snapshot', exact_resume_supported=False),
                    online_state_dict=QNetwork().state_dict()), checkpoint)
    output = tmp_path / 'output'
    assert module.main(['--checkpoint', str(checkpoint), '--mode', mode, '--episodes', '1',
                        '--max-placements', '2', '--max-decisions', '2', '--output', str(output)]) == 0
    report = json.loads((output / 'summary.json').read_text())
    assert report['checkpoint_unchanged'] and report['weights_unchanged'] and report['no_gradients']
    assert report['epsilon'] == 0 and report['optimizer_steps'] == report['replay_updates'] == 0
    assert report['network_forward_count'] == 2
    records = (output / 'decisions.jsonl').read_text().splitlines()
    assert len(records) == (2 if mode == 'play' else 0)
    if mode == 'play':
        assert all(json.loads(line)['target_verified'] for line in records)
