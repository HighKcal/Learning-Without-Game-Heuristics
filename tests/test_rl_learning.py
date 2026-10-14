import random

import numpy as np
import pytest
import torch

from tetris_agent.rl.config import TrainingConfig
from tetris_agent.rl.dqn import DQNTrainer
from tetris_agent.rl.network import QNetwork
from tetris_agent.rl.policy import EpsilonGreedyPolicy, FormalRandomPolicy
from tetris_agent.rl.replay import ReplayBuffer, Transition
from tetris_agent.rl.state import geometric_mask
from test_rl_foundation import semantic


def experience(*, terminated=False, truncated=False, reward=1.):
    state = semantic()
    action = int(np.flatnonzero(geometric_mask(state))[0])
    return Transition(state, action, reward, state, terminated, truncated)


def test_network_exact_baseline_and_output():
    network = QNetwork()
    assert [type(m) for m in network] == [torch.nn.Linear, torch.nn.ReLU, torch.nn.Linear,
                                         torch.nn.ReLU, torch.nn.Linear]
    assert [(m.in_features, m.out_features) for m in network if isinstance(m, torch.nn.Linear)] == [
        (235, 256), (256, 256), (256, 800)]
    assert network(torch.zeros(3, 235)).shape == (3, 800)


def test_random_and_greedy_masks_and_seed():
    state = semantic()
    mask = np.zeros(800, dtype=bool)
    mask[[4, 17]] = True
    network = QNetwork()
    with torch.no_grad():
        for p in network.parameters():
            p.zero_()
        network[-1].bias[0] = 1000  # Invalid largest value.
        network[-1].bias[4] = 2
        network[-1].bias[17] = 5
    before = random.getstate()
    policy = EpsilonGreedyPolicy(network, seed=7)
    assert policy.select(state, mask, 0) == 17
    selected = [policy.select(state, mask, 1) for _ in range(1000)]
    assert set(selected) == {4, 17}
    assert 400 < selected.count(4) < 600
    first, second = FormalRandomPolicy(7), FormalRandomPolicy(7)
    a = [first.select(state, mask) for _ in range(1000)]
    assert a == [second.select(state, mask) for _ in range(1000)]
    assert set(a) == {4, 17} and 400 < a.count(4) < 600
    assert random.getstate() == before


def test_replay_ownership_capacity_and_seed():
    board = np.zeros((20, 10), dtype=np.uint8)
    state = semantic(board)
    buffer, twin = ReplayBuffer(3, seed=11), ReplayBuffer(3, seed=11)
    for i in range(5):
        t = Transition(state, i, float(i), state, False, i == 4)
        buffer.add(t)
        twin.add(t)
    board[:] = 1
    assert len(buffer) == 3
    sample = buffer.sample(3)
    assert sorted(t.action_id for t in sample) == [2, 3, 4]
    assert [t.action_id for t in sample] == [t.action_id for t in twin.sample(3)]
    assert all(not t.state.board.any() and not t.next_state.board.any() for t in sample)
    assert all(t.state.board.dtype == np.uint8 for t in sample)
    assert sample[0].state is not state and not np.shares_memory(sample[0].state.board, state.board)
    with pytest.raises(ValueError):
        sample[0].state.board.setflags(write=True)


@pytest.mark.parametrize('algorithm', ['vanilla', 'double'])
def test_initial_sync_updates_and_gradient_counter(algorithm):
    trainer = DQNTrainer(TrainingConfig(decision_budget=12, algorithm=algorithm, device='cpu',
                                       batch_size=2, learning_starts=3, target_sync_updates=2))
    initial = {k: v.clone() for k, v in trainer.online.state_dict().items()}
    assert all(torch.equal(v, trainer.target.state_dict()[k]) for k, v in initial.items())
    assert trainer.target_sync_count == trainer.gradient_updates == 0
    assert trainer.observe(experience()) is None
    assert trainer.observe(experience()) is None
    assert trainer.gradient_updates == 0  # Environment experiences do not sync target.
    metrics = trainer.observe(experience(truncated=True))
    assert trainer.gradient_updates == 1 and trainer.target_sync_count == 0
    assert metrics.loss > 0 and np.isfinite(metrics.mean_abs_td_error)
    assert all(torch.equal(v, trainer.target.state_dict()[k]) for k, v in initial.items())
    assert any(not torch.equal(v, trainer.online.state_dict()[k]) for k, v in initial.items())
    trainer.observe(experience(terminated=True))
    assert trainer.gradient_updates == 2 and trainer.target_sync_count == 1
    assert all(torch.equal(v, trainer.target.state_dict()[k]) for k, v in trainer.online.state_dict().items())
    assert all(p.grad is None for p in trainer.target.parameters())


def test_default_hard_sync_is_5000_optimizer_steps():
    trainer = DQNTrainer(TrainingConfig(decision_budget=6, device='cpu', batch_size=1, learning_starts=1))
    trainer.gradient_updates = 4998  # Exercise the production boundary without a long run.
    trainer.observe(experience())
    assert trainer.gradient_updates == 4999 and trainer.target_sync_count == 0
    trainer.observe(experience())
    assert trainer.gradient_updates == 5000 and trainer.target_sync_count == 1


def test_trainer_terminal_vs_truncated_targets():
    losses = []
    for terminal, truncated in [(True, False), (False, True)]:
        trainer = DQNTrainer(TrainingConfig(decision_budget=1, device='cpu', batch_size=1, learning_starts=1))
        with torch.no_grad():
            for p in trainer.online.parameters():
                p.zero_()
            for p in trainer.target.parameters():
                p.zero_()
            trainer.target[-1].bias.fill_(2)
        metrics = trainer.observe(experience(terminated=terminal, truncated=truncated, reward=0))
        losses.append(metrics.loss)
    assert losses == pytest.approx([0, .99 * 2 - .5])


def test_gradient_clipping_applied_to_actual_gradients():
    trainer = DQNTrainer(TrainingConfig(decision_budget=1, device='cpu', batch_size=1, learning_starts=1))
    with torch.no_grad():
        trainer.online[0].bias.fill_(1000)
        trainer.online[2].weight.fill_(.01)
        trainer.online[2].bias.fill_(1000)
        trainer.online[4].weight.zero_()
        trainer.online[4].bias.zero_()
    metrics = trainer.observe(experience(terminated=True, reward=100))
    assert metrics.gradient_norm_before_clip > 10
    norm = torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(p.grad) for p in trainer.online.parameters()]))
    assert float(norm) <= 10.0001


def test_epsilon_budget_and_warmup_defaults():
    config = TrainingConfig(decision_budget=15_000)
    assert config.epsilon_decay_decisions == 4000
    assert [config.epsilon(n) for n in (0, 4999, 5000, 7000, 9000, 20000)] == pytest.approx(
        [1, 1, 1, .525, .05, .05])
    assert (config.replay_capacity, config.batch_size, config.learning_starts,
            config.target_sync_updates, config.gradient_clip, config.gamma, config.learning_rate) == (
                100_000, 128, 5000, 5000, 10, .99, 1e-4)
    with pytest.raises(TypeError):
        TrainingConfig()


def test_seed_reproduces_cpu_learning():
    configs = TrainingConfig(decision_budget=1, device='cpu', batch_size=1, learning_starts=1)
    first, second = DQNTrainer(configs), DQNTrainer(configs)
    first.observe(experience())
    second.observe(experience())
    assert all(torch.equal(v, second.online.state_dict()[k]) for k, v in first.online.state_dict().items())
