"""Penalty ablation contracts; real integration uses only two decisions."""
from dataclasses import asdict
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from tetris_gymnasium.envs.tetris import Tetris

from tetris_agent import TargetPlacement
from tetris_agent.rl.artifacts import ArtifactConfig
from tetris_agent.rl.config import TrainingConfig
from tetris_agent.rl.dqn import DQNTrainer
from tetris_agent.rl.reward import RewardComponents, reward_components
from tetris_agent.rl.training import TrainingSummary, train
from test_planner import select_piece, state_signature
from test_rl_foundation import semantic
from test_rl_hardening import load_cli, read_json
from test_rl_runner import FixedPolicy


@pytest.mark.parametrize('penalty', [-0.05, -0.10, -0.20])
@pytest.mark.parametrize('mode', ['minimal', 'shaped'])
@pytest.mark.parametrize('lines', [0, 1, 4])
@pytest.mark.parametrize('terminated', [False, True])
@pytest.mark.parametrize('unreachable', [False, True])
def test_only_unreachable_component_changes(penalty, mode, lines, terminated, unreachable):
    board = np.zeros((20, 10), dtype=np.uint8)
    board[17, 0] = 1  # Two holes: phi = -0.25 * 2 / 12.
    state = semantic(board)
    next_board = np.zeros((20, 10), dtype=np.uint8)
    next_board[18, 1] = 1  # One hole: phi' = -0.25 / 11.
    next_state = semantic(next_board)
    shaping = ((0.0 if terminated else .99 * (-0.25 / 11)) - (-0.25 * 2 / 12)
               if mode == 'shaped' else 0.0)
    legacy = RewardComponents(float(lines), -0.05 if unreachable else 0.0,
                              -1.0 if terminated else 0.0, shaping)
    options = dict(lines=lines, unreachable=unreachable, terminated=terminated, mode=mode)
    assert reward_components(state, next_state, **options) == legacy
    actual = reward_components(state, next_state, **options, unreachable_penalty=penalty)
    assert actual.line == legacy.line
    assert actual.death == legacy.death
    assert actual.shaping == legacy.shaping
    assert actual.unreachable == (penalty if unreachable else 0.0)
    if penalty == -0.05 or not unreachable:
        assert actual == legacy


@pytest.mark.parametrize('penalty', [0.0, -0.0, 0.1, float('nan'), float('inf'), -float('inf')])
def test_config_rejects_nonnegative_or_nonfinite_penalty(penalty):
    with pytest.raises(ValueError, match='finite negative'):
        TrainingConfig(decision_budget=1, unreachable_penalty=penalty)


def test_old_config_defaults_and_baseline_contract():
    baseline = dict(decision_budget=100000, algorithm='double', reward_mode='minimal',
                    seed=42, device='cuda', learning_rate=1e-4, gamma=.99,
                    replay_capacity=100000, batch_size=128, learning_starts=5000,
                    target_sync_updates=5000, gradient_clip=10.0)
    config = TrainingConfig(**baseline)
    assert asdict(config) == dict(baseline, unreachable_penalty=-0.05, epsilon_decay_override=None,
                                 reachability_mask_mode='geometric')
    for seed in (42, 123):
        for penalty in (-0.10, -0.20):
            variant = TrainingConfig(**dict(baseline, seed=seed), unreachable_penalty=penalty)
            assert asdict(variant) == dict(baseline, seed=seed, unreachable_penalty=penalty,
                                         epsilon_decay_override=None, reachability_mask_mode='geometric')
            assert variant.target_sync_updates == 5000
            assert variant.epsilon_decay_decisions == config.epsilon_decay_decisions == 38000
            assert [variant.epsilon(n) for n in (0, 4999, 5000, 24000, 43000, 100000)] == [
                config.epsilon(n) for n in (0, 4999, 5000, 24000, 43000, 100000)]


@pytest.mark.parametrize('value', ['0', '0.1', 'nan', 'inf', '-inf', 'invalid'])
def test_cli_rejects_invalid_penalty(tmp_path, value):
    with pytest.raises(SystemExit) as error:
        load_cli().build_parser().parse_args([
            '--algorithm', 'double', '--reward', 'minimal', '--decision-budget', '1',
            '--seed', '42', '--output-dir', str(tmp_path / 'run'),
            '--unreachable-penalty=' + value])
    assert error.value.code == 2


@pytest.mark.parametrize('seed', [42, 123])
@pytest.mark.parametrize('penalty', [None, -0.10, -0.20])
def test_cli_passes_penalty_with_sync5000_without_training(tmp_path, monkeypatch, seed, penalty):
    cli = load_cli()
    configs = []

    def fake_train(env, trainer, **kwargs):
        configs.append(trainer.config)
        assert kwargs['episode_decision_limit'] == 10000
        assert kwargs['artifacts'].log_every == 100
        assert kwargs['artifacts'].checkpoint_every == 1000
        return TrainingSummary([], 0, 0, 0, 0)

    monkeypatch.setattr(cli, 'train', fake_train)
    monkeypatch.setattr(cli, 'DQNTrainer', lambda config: SimpleNamespace(config=config))
    monkeypatch.setattr(cli, 'Tetris', lambda **kwargs: SimpleNamespace(close=lambda: None))
    args = ['--algorithm', 'double', '--reward', 'minimal', '--decision-budget', '100000',
            '--seed', str(seed), '--device', 'cpu', '--output-dir', str(tmp_path / 'run'),
            '--max-decisions-per-episode', '10000', '--log-every', '100',
            '--checkpoint-every', '1000']
    if penalty is not None:
        args += ['--unreachable-penalty', str(penalty)]
    assert cli.main(args) == 0
    config, = configs
    assert config.unreachable_penalty == (-0.05 if penalty is None else penalty)
    assert config.target_sync_updates == 5000
    assert config.decision_budget == 100000 and config.seed == seed
    assert config.algorithm == 'double' and config.reward_mode == 'minimal'
    assert not (tmp_path / 'run').exists()


@pytest.mark.parametrize('penalty', [-0.05, -0.10, -0.20])
def test_two_decision_unreachable_train_and_saved_config(tmp_path, penalty):
    class UnreachableEnv(Tetris):
        def reset(self, **kwargs):
            result = super().reset(**kwargs)
            select_piece(self, 1)
            self.board[17, 4:14] = 2
            self.before = state_signature(self)
            return result

    options = {} if penalty == -0.05 else dict(unreachable_penalty=penalty)
    trainer = DQNTrainer(TrainingConfig(decision_budget=2, device='cpu', **options))
    trainer.select = FixedPolicy(TargetPlacement(7, 18, 0)).select
    output = tmp_path / 'run'
    env = UnreachableEnv(gravity=True)
    try:
        result = train(env, trainer, episode_decision_limit=10000,
                       artifacts=ArtifactConfig(output, log_every=1, checkpoint_every=1))
        assert state_signature(env) == env.before
    finally:
        env.close()
    assert result.status == 'completed'
    assert result.decision_count == result.stored_experiences == 2
    assert result.gradient_updates == result.target_sync_count == 0
    episode, = result.episodes
    assert episode.unreachable_count == episode.self_loop_count == 2
    assert episode.pieces_placed == episode.lines_cleared == 0
    assert episode.reward_totals == dict(line=0.0, unreachable=2 * penalty, death=0.0,
                                        shaping=0.0, total=2 * penalty)
    assert episode.episode_return == 2 * penalty
    assert not episode.terminated and episode.truncated
    for index, transition in enumerate(trainer.replay._items):
        assert transition.reward == penalty and not transition.terminated
        assert transition.truncated == (index == 1)
        np.testing.assert_array_equal(transition.state.board, transition.next_state.board)
        assert transition.state.current_piece == transition.next_state.current_piece
        assert transition.state.next_queue == transition.next_state.next_queue
    saved = read_json(output / 'config.json')['training']
    assert saved == asdict(trainer.config)
    assert saved['unreachable_penalty'] == penalty and saved['target_sync_updates'] == 5000
    for name in ('checkpoint_000000001.pt', 'checkpoint_000000002.pt', 'checkpoint_final.pt'):
        checkpoint = torch.load(output / name, map_location='cpu', weights_only=True)
        assert checkpoint['config'] == saved
        assert checkpoint['config']['unreachable_penalty'] == penalty
        assert checkpoint['config']['target_sync_updates'] == 5000
