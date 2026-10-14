"""Bit-exact RL geometry against the unchanged general TargetSpace oracle."""
from dataclasses import asdict

import numpy as np
import pytest
import torch

from tetris_agent.rl import dqn
from tetris_agent.rl.config import TrainingConfig
from tetris_agent.rl.replay import Transition
from tetris_agent.rl.state import (
    GEOMETRY, SPACE, SemanticState, _SPAWN_MATRICES, geometric_mask,
)


def original_mask(state):
    board = np.ones(GEOMETRY.board_shape, dtype=np.uint8)
    board[:20, 4:14] = state.board
    return SPACE.action_mask(board, _SPAWN_MATRICES[state.current_piece])


def boards():
    yield np.zeros((20, 10), dtype=np.uint8)
    yield np.ones((20, 10), dtype=np.uint8)
    board = np.ones((20, 10), dtype=np.uint8)
    board[:2] = 0
    yield board
    board = np.zeros((20, 10), dtype=np.uint8)
    board[-1] = 1
    yield board
    board = np.zeros((20, 10), dtype=np.uint8)
    board[:, [0, -1]] = 1
    yield board
    board = np.zeros((20, 10), dtype=np.uint8)
    board[17] = 1  # Sealed cavity remains geometrically legal.
    yield board
    yield (np.indices((20, 10)).sum(axis=0) % 2).astype(np.uint8)
    board = np.zeros((20, 10), dtype=np.uint8)
    board[1] = 1  # Blank matrix border may extend above row zero.
    yield board
    rng = np.random.default_rng(20261004)
    for i in range(288):
        board = np.zeros((20, 10), dtype=np.uint8)
        heights = rng.integers(0, 20, size=10)
        for col, height in enumerate(heights):
            if height:
                board[-height:, col] = 1
        if i % 3 == 0:
            board[rng.random((20, 10)) < .3] = 0  # Holes/overhangs.
        elif i % 3 == 1:
            board = (rng.random((20, 10)) < rng.uniform(.05, .95)).astype(np.uint8)
        yield board


@pytest.mark.parametrize('piece', range(7))
def test_exact_mask_all_slots_structured_and_random_boards(piece):
    for board in boards():
        state = SemanticState(board, piece, (0, 1, 2, 3))
        before = state.board.tobytes()
        expected = original_mask(state)
        actual = geometric_mask(state)
        assert actual.shape == (800,) and actual.dtype == np.bool_
        assert np.array_equal(expected, actual)
        assert state.board.tobytes() == before
        # A returned mask is owned; changing it cannot corrupt a static table.
        actual[:] = False
        assert np.array_equal(expected, geometric_mask(state))


@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason='CUDA unavailable'))])
def test_full_ddqn_updates_bit_exact_with_original_mask(monkeypatch, device):
    config = TrainingConfig(decision_budget=10000, device=device)
    old, new = dqn.DQNTrainer(config), dqn.DQNTrainer(config)
    batch = []
    for i in range(128):
        board = np.zeros((20, 10), dtype=np.uint8)
        board[17:, :i % 8] = 1
        state = SemanticState(board, i % 7, (0, 1, 2, 3))
        action = int(np.flatnonzero(geometric_mask(state))[0])
        batch.append(Transition(state, action, -.05 if i % 2 else 0., state, i % 31 == 0, i % 19 == 0))
    # Cross the real sync boundary in three updates without a 5000-update run.
    old.gradient_updates = new.gradient_updates = 4998
    for _ in range(3):
        with monkeypatch.context() as patch:
            patch.setattr(dqn, 'geometric_mask', original_mask)
            expected = old._update(batch, .525)
        actual = new._update(batch, .525)
        assert asdict(expected) == asdict(actual)
        for name in old.online.state_dict():
            assert torch.equal(old.online.state_dict()[name], new.online.state_dict()[name])
            assert torch.equal(old.target.state_dict()[name], new.target.state_dict()[name])
        for left, right in zip(old.optimizer.state.values(), new.optimizer.state.values()):
            for key in left:
                assert torch.equal(left[key], right[key])
    assert new.target_sync_count == 1
    assert all(p.grad is None for p in new.target.parameters())


def test_formal_random_trajectory_identical_with_original_mask(monkeypatch):
    from tetris_gymnasium.envs.tetris import Tetris
    from tetris_agent.rl import runner
    from tetris_agent.rl.policy import FormalRandomPolicy

    trajectories = []
    summaries = []
    for mask in (original_mask, geometric_mask):
        transitions, episodes = [], []
        policy = FormalRandomPolicy(42)
        env = Tetris(gravity=True)
        try:
            with monkeypatch.context() as patch:
                patch.setattr(runner, 'geometric_mask', mask)
                while len(transitions) < 64:
                    result = runner.RLRunner(policy).run(
                        env, env_seed=42 + len(episodes), max_decisions=64-len(transitions),
                        on_transition=transitions.append)
                    assert result.diagnostic is None and result.decision_count
                    row = asdict(result)
                    row.pop('planner_elapsed_seconds')
                    episodes.append(row)
        finally:
            env.close()
        trajectories.append(transitions)
        summaries.append(episodes)
    assert summaries[0] == summaries[1]
    for a, b in zip(*trajectories):
        assert (a.action_id, a.reward, a.terminated, a.truncated) == (b.action_id, b.reward, b.terminated, b.truncated)
        for name in ('state', 'next_state'):
            left, right = getattr(a, name), getattr(b, name)
            assert left.board.tobytes() == right.board.tobytes()
            assert (left.current_piece, left.next_queue) == (right.current_piece, right.next_queue)
