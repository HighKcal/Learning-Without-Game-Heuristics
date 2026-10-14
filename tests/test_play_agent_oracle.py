"""Play CLI routing, exact oracle witness reuse, and frozen-weight integrity."""
import importlib.util
import json
from pathlib import Path

import pytest
import torch
from tetris_gymnasium.envs.tetris import Tetris

from tetris_agent import Executor, TacticalPlanner
from tetris_agent.rl.network import QNetwork


@pytest.fixture
def cli():
    spec = importlib.util.spec_from_file_location(
        'oracle_play_cli', Path(__file__).parents[1] / 'scripts/play_agent.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def checkpoint(tmp_path):
    # New temporary inference fixture; no training or existing checkpoint writes.
    network = QNetwork()
    with torch.no_grad():
        for parameter in network.parameters():
            parameter.zero_()
    path = tmp_path / 'fixture.pt'
    torch.save(dict(schema_version=1,
                    checkpoint_contract=dict(kind='training_snapshot', exact_resume_supported=False),
                    online_state_dict=network.state_dict()), path)
    return path


@pytest.mark.parametrize('mode,limit', [('play', 2), ('research', 1)])
def test_oracle_cli_uses_single_search_and_original_witness(
        cli, checkpoint, tmp_path, monkeypatch, mode, limit):
    searches, executions = [], []
    enumerate_targets = TacticalPlanner.enumerate_reachable_targets
    execute = Executor.execute

    def counted_search(self, env):
        result = enumerate_targets(self, env)
        searches.append(result)
        return result

    def forbidden_plan(*args):
        pytest.fail('Oracle inference must not call target-specific Planner.')

    def checked_execute(self, env, target, plan):
        assert any(target == placement and plan is witness
                   for placement, witness in searches[-1].placements)
        result = execute(self, env, target, plan)
        assert result.success and result.target_verified
        executions.append(result)
        return result

    monkeypatch.setattr(TacticalPlanner, 'enumerate_reachable_targets', counted_search)
    monkeypatch.setattr(TacticalPlanner, 'plan', forbidden_plan)
    monkeypatch.setattr(Executor, 'execute', checked_execute)
    output = tmp_path / 'output'
    assert cli.main(['--checkpoint', str(checkpoint), '--mode', mode,
                     '--reachability-mask', 'oracle', '--episodes', '1',
                     '--max-placements', '2', '--max-decisions', '1',
                     '--output', str(output)]) == 0
    report = json.loads((output / 'summary.json').read_text())
    episode, = report['episodes']
    assert episode['placements'] == len(executions) == report['network_forward_count'] == limit
    assert episode['oracle_computation_count'] == len(searches) == limit + 1
    assert episode['planner_call_count'] == episode['selected_physical_u_count'] == 0
    assert episode['unreachable_count'] == 0
    assert report['inference_semantics'] == 'single_bfs_mask_argmax_witness'
    assert all(report[k] for k in ('checkpoint_unchanged', 'weights_unchanged', 'no_gradients'))
    assert report['optimizer_steps'] == report['replay_updates'] == 0
    assert not report['training'] and not report['trainer_constructed']


def test_default_play_preserves_fallback_and_never_enumerates(
        cli, checkpoint, tmp_path, monkeypatch):
    def forbidden_search(*args):
        pytest.fail('Default Play Mode must keep target-specific fallback.')
    monkeypatch.setattr(TacticalPlanner, 'enumerate_reachable_targets', forbidden_search)
    output = tmp_path / 'output'
    assert cli.main(['--checkpoint', str(checkpoint), '--mode', 'play', '--episodes', '1',
                     '--max-placements', '2', '--output', str(output)]) == 0
    report = json.loads((output / 'summary.json').read_text())
    assert report['reachability_mask_mode'] == 'geometric'
    assert report['inference_semantics'] == 'q_ranking_planner_fallback'
    records = [json.loads(line) for line in (output / 'decisions.jsonl').read_text().splitlines()]
    assert len(records) == 2 and all(record['target_verified'] for record in records)


def test_incomplete_oracle_aborts_without_fallback_or_execution(
        cli, checkpoint, tmp_path, monkeypatch):
    def broken_search(*args):
        raise TimeoutError('injected incomplete search')
    def forbidden(*args):
        pytest.fail('Uncertain oracle must not fall back or execute.')
    monkeypatch.setattr(TacticalPlanner, 'enumerate_reachable_targets', broken_search)
    monkeypatch.setattr(TacticalPlanner, 'plan', forbidden)
    monkeypatch.setattr(Executor, 'execute', forbidden)
    output = tmp_path / 'output'
    assert cli.main(['--checkpoint', str(checkpoint), '--mode', 'play',
                     '--reachability-mask', 'oracle', '--episodes', '2',
                     '--output', str(output)]) == 1
    report = json.loads((output / 'summary.json').read_text())
    episode, = report['episodes']
    assert episode['stop_reason'] == 'ORACLE_ERROR' and episode['placements'] == 0
    assert report['network_forward_count'] == 0 and report['checkpoint_unchanged']


def test_oracle_visual_wrapper_only_renders_live_executor_steps(
        cli, checkpoint, tmp_path, monkeypatch):
    import cv2
    renders, waits, executions = [], [], []
    execute = Executor.execute

    def record_execution(self, *args):
        result = execute(self, *args)
        executions.append(result)
        return result

    monkeypatch.setattr(Tetris, 'render', lambda self: renders.append(True))
    monkeypatch.setattr(cv2, 'waitKey', lambda delay: waits.append(delay))
    monkeypatch.setattr(Executor, 'execute', record_execution)
    assert cli.main(['--checkpoint', str(checkpoint), '--mode', 'play',
                     '--reachability-mask', 'oracle', '--episodes', '1', '--max-placements', '1',
                     '--render', '--step-delay-ms', '7', '--output', str(tmp_path / 'output')]) == 0
    execution, = executions
    assert len(renders) == len(waits) == len(execution.executed_actions) > 0
    assert set(waits) == {7}
