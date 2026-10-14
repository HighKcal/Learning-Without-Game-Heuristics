"""Durable runs and schedule boundaries without long training budgets."""
from dataclasses import asdict
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from tetris_gymnasium.envs.tetris import Tetris

from tetris_agent.rl.artifacts import ArtifactConfig, RunArtifacts
from tetris_agent.rl.config import TrainingConfig
from tetris_agent.rl.dqn import DQNTrainer
from tetris_agent.rl.training import train
from test_rl_learning import experience


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def records(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]


def load_cli():
    spec = importlib.util.spec_from_file_location('train_cli', Path(__file__).parents[1] / 'scripts/train_rl.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('budget,decay,end', [(10000, 2000, 7000), (100000, 38000, 43000),
                                           (300000, 118000, 123000), (5001, 1, 5001)])
def test_exact_budget_schedule_boundaries(budget, decay, end):
    config = TrainingConfig(decision_budget=budget)
    assert config.epsilon_decay_decisions == decay
    assert config.epsilon(4999) == config.epsilon(5000) == 1.
    assert config.epsilon(end - 1) > .05
    assert config.epsilon(end) == pytest.approx(.05)
    assert config.epsilon(budget) == pytest.approx(.05)
    assert config.epsilon(5000 + decay / 2) == pytest.approx(.525)


@pytest.mark.parametrize('budget', [1, 4999, 5000])
def test_budget_ending_during_warmup(budget):
    config = TrainingConfig(decision_budget=budget)
    assert config.epsilon_decay_decisions == 1
    assert config.epsilon(budget) == 1.


@pytest.mark.parametrize('algorithm', ['vanilla', 'double'])
def test_durable_run_periodic_and_final_contents(tmp_path, algorithm):
    output = tmp_path / 'run'
    trainer = DQNTrainer(TrainingConfig(decision_budget=5, algorithm=algorithm, device='cpu',
                                       batch_size=2, learning_starts=2, target_sync_updates=2))
    env = Tetris(gravity=True)
    durable_before_finish = []
    def learning(_):
        # On update 2 (decision 3), decision 2 must already be on disk.
        if trainer.stored_experiences == 3:
            durable_before_finish.extend(records(output / 'metrics.jsonl'))
            assert (output / 'checkpoint_000000002.pt').exists()
    try:
        result = train(env, trainer, episode_decision_limit=3, on_learning=learning,
                       artifacts=ArtifactConfig(output, 2, 2))
    finally:
        env.close()
    assert result.status == 'completed'
    assert result.decision_count == result.stored_experiences == 5
    assert result.gradient_updates == 4 and result.target_sync_count == 2
    assert durable_before_finish[0]['decision_count'] == 2
    rows = records(output / 'metrics.jsonl')
    assert [r['decision_count'] for r in rows] == [2, 4, 5]
    assert [r['event'] for r in rows] == ['periodic', 'periodic', 'final']
    assert rows[-1]['episode'] == 2
    assert rows[-1]['pieces_placed'] + rows[-1]['unreachable_count'] == 5
    assert rows[-1]['planner_call_count'] == 5
    assert rows[-1]['last_learning']['max_selected_q'] >= rows[-1]['last_learning']['mean_selected_q']
    assert rows[-1]['self_loop_ratio'] == rows[-1]['self_loop_count'] / 5
    for key in ('episode_return', 'lines_cleared', 'unreachable_rate', 'epsilon', 'replay_size',
                'wall_clock_seconds', 'decisions_per_second', 'placements_per_second',
                'planner_average_seconds', 'learner_elapsed_seconds', 'learner_average_seconds'):
        assert key in rows[-1]
    config = read_json(output / 'config.json')
    assert config['training'] == asdict(trainer.config)
    assert config['resolved_device'] == 'cpu'
    assert config['epsilon']['counter'] == 'stored_experiences'
    assert config['versions']['tetris_gymnasium']
    status = read_json(output / 'status.json')
    assert status['status'] == 'completed' and status['ended_at'] >= status['started_at']
    assert sorted(p.name for p in output.glob('*.pt')) == [
        'checkpoint_000000002.pt', 'checkpoint_000000004.pt', 'checkpoint_final.pt']
    periodic = torch.load(output / 'checkpoint_000000002.pt', map_location='cpu', weights_only=True)
    assert periodic['decision_count'] == periodic['stored_experiences'] == 2
    assert periodic['gradient_updates'] == 1 and periodic['target_sync_count'] == 0
    assert periodic['epsilon']['next_selection_epsilon'] == 1.
    checkpoint = torch.load(output / 'checkpoint_final.pt', map_location='cpu', weights_only=True)
    assert checkpoint['config'] == asdict(trainer.config)
    assert checkpoint['decision_count'] == checkpoint['stored_experiences'] == 5
    assert checkpoint['gradient_updates'] == 4
    assert checkpoint['target_sync_count'] == 2
    assert checkpoint['epsilon']['next_selection_epsilon'] == trainer.epsilon
    assert checkpoint['optimizer_state_dict']['state']
    for key, value in trainer.online.state_dict().items():
        torch.testing.assert_close(value, checkpoint['online_state_dict'][key], rtol=0, atol=0)
        torch.testing.assert_close(trainer.target.state_dict()[key], checkpoint['target_state_dict'][key], rtol=0, atol=0)
    contract = checkpoint['checkpoint_contract']
    assert contract['kind'] == 'training_snapshot' and contract['exact_resume_supported'] is False
    assert len(contract['missing_for_exact_resume']) >= 9
    summary = read_json(output / 'summary.json')
    assert summary['decisions_per_second'] == pytest.approx(5 / summary['wall_clock_seconds'])
    assert summary['gradient_updates_per_second'] == pytest.approx(4 / summary['wall_clock_seconds'])
    assert summary['planner_average_seconds'] == pytest.approx(summary['planner_elapsed_seconds'] / 5)
    assert summary['learner_average_seconds'] == pytest.approx(summary['learner_elapsed_seconds'] / 4)
    assert summary['wall_clock_seconds'] >= rows[-1]['wall_clock_seconds']
    assert result.throughput['wall_clock_seconds'] == summary['wall_clock_seconds']
    assert not list(output.glob('*.tmp'))


def test_system_error_attempt_not_stored_or_epsilon_progress(tmp_path):
    trainer = DQNTrainer(TrainingConfig(decision_budget=10000, device='cpu'))
    def broken_policy(*args):
        raise RuntimeError('injected policy error')
    trainer.select = broken_policy
    env = Tetris(gravity=True)
    try:
        summary = train(env, trainer, episode_decision_limit=3,
                        artifacts=ArtifactConfig(tmp_path / 'failed', 1, 1))
    finally:
        env.close()
    assert summary.status == 'system_error'
    assert summary.decision_count == 1 and summary.stored_experiences == summary.gradient_updates == 0
    assert trainer.epsilon == 1
    row = records(tmp_path / 'failed/metrics.jsonl')[-1]
    assert row['decision_count'] == 1 and row['stored_experiences'] == 0
    assert row['episode_metrics']['stop_reason'] == 'POLICY_ERROR'
    assert read_json(tmp_path / 'failed/status.json')['status'] == 'system_error'
    assert (tmp_path / 'failed/checkpoint_final.pt').exists()


@pytest.mark.parametrize('exception,status', [(RuntimeError, 'failed'), (KeyboardInterrupt, 'interrupted')])
def test_exception_finalizes_artifacts_without_claiming_success(tmp_path, exception, status):
    class BrokenEnv:
        def reset(self, **kwargs):
            raise exception('injected reset failure')
    trainer = DQNTrainer(TrainingConfig(decision_budget=1, device='cpu'))
    output = tmp_path / status
    with pytest.raises(exception):
        train(BrokenEnv(), trainer, episode_decision_limit=1, artifacts=ArtifactConfig(output))
    assert read_json(output / 'status.json')['status'] == status
    assert read_json(output / 'summary.json')['decision_count'] == 0
    assert (output / 'checkpoint_final.pt').exists()


def test_fresh_run_does_not_offer_partial_resume(tmp_path):
    trainer = DQNTrainer(TrainingConfig(decision_budget=2, device='cpu'))
    trainer.observe(experience())
    with pytest.raises(ValueError, match='fresh experiment'):
        train(None, trainer, episode_decision_limit=1, artifacts=ArtifactConfig(tmp_path / 'resume'))
    assert not (tmp_path / 'resume').exists()


def test_existing_run_is_never_overwritten(tmp_path):
    marker = tmp_path / 'config.json'
    marker.write_text('preserve me')
    trainer = DQNTrainer(TrainingConfig(decision_budget=1, device='cpu'))
    with pytest.raises(FileExistsError):
        train(None, trainer, episode_decision_limit=1, artifacts=ArtifactConfig(tmp_path))
    assert marker.read_text() == 'preserve me'


@pytest.mark.parametrize('field,value', [('log_every', 0), ('checkpoint_every', -1),
                                        ('log_every', True), ('checkpoint_every', 1.5)])
def test_artifact_intervals_validated(tmp_path, field, value):
    with pytest.raises(ValueError, match='positive integer'):
        ArtifactConfig(tmp_path, **{field: value})


@pytest.mark.parametrize('flag,value', [('--decision-budget', '0'), ('--log-every', '-1'),
                                      ('--checkpoint-every', '0'), ('--algorithm', 'ppo'),
                                      ('--reward', 'dense'), ('--device', 'gpu'),
                                      ('--seed', '-1'), ('--seed', str(2**32)),
                                      ('--max-decisions-per-episode', '0')])
def test_cli_argument_validation(tmp_path, flag, value):
    args = ['--algorithm', 'double', '--reward', 'minimal', '--decision-budget', '1',
            '--seed', '42', '--output-dir', str(tmp_path / 'run'), flag, value]
    with pytest.raises(SystemExit) as error:
        load_cli().build_parser().parse_args(args)
    assert error.value.code == 2


@pytest.mark.parametrize('budget', [1, 10000, 100000, 300000, 432109])
def test_cli_always_calls_same_production_trainer(tmp_path, monkeypatch, budget, capsys):
    cli = load_cli()
    assert cli.train is train
    calls = []
    def fake_train(env, trainer, **kwargs):
        calls.append((env, trainer, kwargs))
        from tetris_agent.rl.training import TrainingSummary
        return TrainingSummary([], 0, 0, 0, 0)
    monkeypatch.setattr(cli, 'train', fake_train)
    monkeypatch.setattr(cli, 'DQNTrainer', lambda config: SimpleNamespace(config=config))
    closed = []
    monkeypatch.setattr(cli, 'Tetris', lambda **kwargs: SimpleNamespace(close=lambda: closed.append(True)))
    assert cli.main(['--algorithm', 'vanilla', '--reward', 'shaped', '--decision-budget', str(budget),
                     '--seed', '7', '--device', 'cpu', '--output-dir', str(tmp_path / 'run'),
                     '--log-every', '2', '--checkpoint-every', '3']) == 0
    _, trainer, options = calls[0]
    assert trainer.config.decision_budget == budget
    assert trainer.config.learning_starts == 5000 and trainer.config.batch_size == 128
    assert trainer.config.algorithm == 'vanilla' and trainer.config.reward_mode == 'shaped'
    assert options['artifacts'].log_every == 2 and options['artifacts'].checkpoint_every == 3
    assert closed == [True]


def test_failed_atomic_checkpoint_preserves_previous_snapshot(tmp_path, monkeypatch):
    trainer = DQNTrainer(TrainingConfig(decision_budget=1, device='cpu'))
    writer = RunArtifacts(ArtifactConfig(tmp_path / 'run'), trainer, 1)
    row = dict(decision_count=0, episode=0)
    writer.checkpoint(trainer, row, 'checkpoint_final.pt', 'running')
    previous = (writer.path / 'checkpoint_final.pt').read_bytes()
    def fail(payload, stream):
        stream.write(b'partial')
        raise OSError('injected disk failure')
    monkeypatch.setattr(torch, 'save', fail)
    with pytest.raises(OSError, match='disk failure'):
        writer.finish(trainer, row, 'completed')
    assert (writer.path / 'checkpoint_final.pt').read_bytes() == previous
    assert read_json(writer.path / 'status.json')['status'] == 'artifact_error'
    assert not list(writer.path.glob('*.tmp'))


def test_artifacts_do_not_change_learning_or_action_sequence(tmp_path):
    results = []
    trainers = []
    config = TrainingConfig(decision_budget=4, device='cpu', batch_size=2, learning_starts=2)
    for artifacts in (None, ArtifactConfig(tmp_path / 'logged', 1, 2)):
        trainer = DQNTrainer(config)
        env = Tetris(gravity=True)
        try:
            result = train(env, trainer, episode_decision_limit=2, artifacts=artifacts)
        finally:
            env.close()
        trainers.append(trainer)
        results.append(result)
    first, second = trainers
    assert first.policy._rng.getstate() == second.policy._rng.getstate()
    assert first.replay._rng.getstate() == second.replay._rng.getstate()
    for a, b in zip(first.replay._items, second.replay._items):
        assert (a.action_id, a.reward, a.terminated, a.truncated) == (b.action_id, b.reward, b.terminated, b.truncated)
        assert (a.next_state.board == b.next_state.board).all()
    for name in first.online.state_dict():
        torch.testing.assert_close(first.online.state_dict()[name], second.online.state_dict()[name], rtol=0, atol=0)
        torch.testing.assert_close(first.target.state_dict()[name], second.target.state_dict()[name], rtol=0, atol=0)
    assert results[0].gradient_updates == results[1].gradient_updates == 3


def test_default_small_budget_records_no_learning_and_final_snapshot(tmp_path):
    cli = load_cli()
    output = tmp_path / 'production'
    assert cli.main(['--algorithm', 'double', '--reward', 'minimal', '--decision-budget', '1',
                     '--seed', '42', '--device', 'cpu', '--output-dir', str(output)]) == 0
    row = read_json(output / 'summary.json')
    assert row['decision_count'] == row['stored_experiences'] == 1
    assert row['gradient_updates'] == 0 and row['last_learning'] is None
    assert row['epsilon'] == 1.
    assert [r['event'] for r in records(output / 'metrics.jsonl')] == ['final']
    assert len(list(output.glob('*.pt'))) == 1


def test_cli_cuda_unavailable_fails_without_silent_fallback(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    output = tmp_path / 'cuda'
    with pytest.raises(SystemExit) as error:
        load_cli().main(['--algorithm', 'double', '--reward', 'minimal', '--decision-budget', '1',
                         '--seed', '42', '--device', 'cuda', '--output-dir', str(output)])
    assert error.value.code == 2 and not output.exists()


def test_no_progress_is_an_error_instead_of_infinite_resets(tmp_path):
    class EndedEnv:
        game_over = True
        @property
        def unwrapped(self):
            return self
        def reset(self, **kwargs):
            pass
    trainer = DQNTrainer(TrainingConfig(decision_budget=1, device='cpu'))
    result = train(EndedEnv(), trainer, episode_decision_limit=1,
                   artifacts=ArtifactConfig(tmp_path / 'no_progress'))
    assert result.status == 'system_error' and result.decision_count == 0
    assert len(result.episodes) == 1 and result.episodes[0].stop_reason == 'NO_PROGRESS'


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
def test_cuda_updated_snapshot_loads_on_cpu(tmp_path):
    trainer = DQNTrainer(TrainingConfig(decision_budget=2, device='cuda', batch_size=2, learning_starts=2))
    env = Tetris(gravity=True)
    output = tmp_path / 'cuda'
    try:
        result = train(env, trainer, episode_decision_limit=2, artifacts=ArtifactConfig(output, 1, 1))
    finally:
        env.close()
    assert result.gradient_updates == 1
    snapshot = torch.load(output / 'checkpoint_final.pt', map_location='cpu', weights_only=True)
    assert snapshot['optimizer_state_dict']['state']
    assert snapshot['gradient_updates'] == 1
    assert read_json(output / 'config.json')['resolved_device'] == 'cuda'
    for name, value in trainer.online.state_dict().items():
        torch.testing.assert_close(snapshot['online_state_dict'][name], value.cpu(), rtol=0, atol=0)
    for state in snapshot['optimizer_state_dict']['state'].values():
        assert state['exp_avg'].device.type == 'cpu'


def test_each_episode_is_durable_before_next_reset(tmp_path):
    output = tmp_path / 'episodes'
    visible_between_episodes = []
    class CheckingEnv(Tetris):
        def reset(self, **kwargs):
            if (output / 'episodes.jsonl').exists():
                visible_between_episodes.append(records(output / 'episodes.jsonl'))
            return super().reset(**kwargs)
    trainer = DQNTrainer(TrainingConfig(decision_budget=5, device='cpu'))
    env = CheckingEnv(gravity=True)
    try:
        result = train(env, trainer, episode_decision_limit=2,
                       artifacts=ArtifactConfig(output, 4, 100))
    finally:
        env.close()
    assert [len(rows) for rows in visible_between_episodes] == [1, 2]
    rows = records(output / 'episodes.jsonl')
    assert [row['episode_index'] for row in rows] == [1, 2, 3]
    assert [row['decision_count'] for row in rows] == [2, 2, 1]
    for row, episode in zip(rows, result.episodes):
        assert {key: row[key] for key in asdict(episode)} == asdict(episode)
        assert row['unreachable_rate'] == episode.unreachable_rate
        assert row['truncated'] and not row['terminated']
        assert row['stop_reason'] == 'TRUNCATED'
    # Episode logging must neither add metrics rows nor reset their cadence.
    assert [(r['event'], r['decision_count']) for r in records(output / 'metrics.jsonl')] == [
        ('periodic', 4), ('final', 5)]


def test_diagnostic_episode_is_preserved(tmp_path):
    output = tmp_path / 'diagnostic'
    trainer = DQNTrainer(TrainingConfig(decision_budget=2, device='cpu'))
    def broken_policy(*args):
        raise RuntimeError('injected failure')
    trainer.select = broken_policy
    env = Tetris(gravity=True)
    try:
        result = train(env, trainer, episode_decision_limit=2, artifacts=ArtifactConfig(output))
    finally:
        env.close()
    row, = records(output / 'episodes.jsonl')
    assert result.status == 'system_error'
    assert row['episode_index'] == 1 and row['decision_count'] == 1
    assert row['stop_reason'] == 'POLICY_ERROR'
    assert row['diagnostic'] == result.episodes[0].diagnostic
    assert row['unreachable_rate'] == 0 and row['max_consecutive_unreachable'] == 0


def test_cli_stdout_stays_compact_with_many_episodes(tmp_path, monkeypatch, capsys):
    from tetris_agent.rl.metrics import EpisodeMetrics
    from tetris_agent.rl.training import TrainingSummary
    cli = load_cli()
    # No large run: exercise only reporting using an already-computed summary.
    summary = TrainingSummary([EpisodeMetrics()] * 10000, 300000, 300000, 295001, 59,
                              throughput={'wall_clock_seconds': 123.0})
    monkeypatch.setattr(cli, 'train', lambda *args, **kwargs: summary)
    monkeypatch.setattr(cli, 'DQNTrainer', lambda config: SimpleNamespace(config=config))
    monkeypatch.setattr(cli, 'Tetris', lambda **kwargs: SimpleNamespace(close=lambda: None))
    assert cli.main(['--algorithm', 'double', '--reward', 'minimal', '--decision-budget', '300000',
                     '--seed', '42', '--device', 'cpu', '--output-dir', str(tmp_path / 'run')]) == 0
    stdout = capsys.readouterr().out
    report = json.loads(stdout)
    assert 'episodes' not in report and 'episode_metrics' not in report
    assert report['episode_count'] == 10000 and report['decision_count'] == 300000
    assert report['gradient_updates'] == 295001 and report['status'] == 'completed'
    assert report['throughput'] == summary.throughput
    assert len(stdout) < 2000
