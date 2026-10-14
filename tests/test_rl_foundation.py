"""P0 contracts, written before wiring the RL runner and learner."""
import copy

import numpy as np
import pytest
import torch
from tetris_gymnasium.envs.tetris import Tetris

from tetris_agent import TacticalPlanner, TargetPlacement, TargetSpace, TetrisAdapter
from tetris_agent.target_space import env_origin_to_target
from test_planner import select_piece, state_signature
from tetris_agent.rl.state import SemanticState, read_state, encode_states, geometric_mask, assert_canonical
from tetris_agent.rl.reward import reward_components, hole_count, potential
from tetris_agent.rl.dqn import bellman_targets


@pytest.fixture
def env():
    game = Tetris(gravity=True)
    game.reset(seed=42)
    yield game
    game.close()


def semantic(board=None, piece=1):
    return SemanticState(np.zeros((20, 10), dtype=np.uint8) if board is None else board,
                         piece, (0, 2, 4, 6))


def test_p0_exact_unreachable_restoration(env):
    select_piece(env, 1)
    env.board[17, 4:14] = 2
    env.randomizer.index = len(env.randomizer.bag) - 1
    before = state_signature(env)
    refs = {name: getattr(env, name) for name in
            ('board', 'active_tetromino', 'queue', 'randomizer', 'holder')}
    assert not TacticalPlanner().plan(env, TargetPlacement(7, 18, 0)).reachable
    assert state_signature(env) == before
    assert all(getattr(env, k) is v for k, v in refs.items())


@pytest.mark.parametrize('seed', [0, 7, 42, 123])
@pytest.mark.parametrize('piece', range(7))
def test_p0_native_drop_has_valid_reachable_target(seed, piece):
    env = Tetris(gravity=True)
    probe = Tetris(gravity=True)
    try:
        env.reset(seed=seed)
        probe.reset(seed=seed)
        select_piece(env, piece)
        for _ in range(8):
            if env.game_over:
                break
            assert_canonical(env)
            state = read_state(env)
            assert geometric_mask(state).any()
            probe.set_state(copy.deepcopy(env.get_state()))
            current = TetrisAdapter(env).read_current_piece()
            targets = []
            place = probe.place_active_tetromino
            def observe():
                targets.append(env_origin_to_target(probe.x, probe.y, 0, current.spawn_matrix))
                place()
            probe.place_active_tetromino = observe
            try:
                probe.step(probe.actions.hard_drop)
            finally:
                del probe.place_active_tetromino
            assert len(targets) == 1
            space = TargetSpace(TetrisAdapter(env).geometry)
            assert geometric_mask(state)[space.encode(targets[0])]
            assert TacticalPlanner().plan(env, targets[0]).reachable
            env.step(env.actions.hard_drop)
    finally:
        env.close()
        probe.close()


def test_state_locked_only_encoding_and_ownership(env):
    state = read_state(env)
    assert state.board.dtype == np.uint8 and state.board.shape == (20, 10)
    assert not state.board.any()  # Observation includes active cells.
    assert state.current_piece == 3  # Explicit I O T S Z J L mapping.
    assert state.next_queue == (2, 6, 4, 1)
    encoded = encode_states([state, state])
    assert encoded.shape == (2, 235) and encoded.dtype == torch.float32
    assert encoded[0, 200:207].tolist() == [0, 0, 0, 1, 0, 0, 0]
    assert encoded[0, 207:].reshape(4, 7).argmax(1).tolist() == [2, 6, 4, 1]
    env.board[19, 4] = 5
    assert not state.board.any()
    with pytest.raises(ValueError):
        state.board[0, 0] = 1


@pytest.mark.parametrize('piece', range(7))
def test_geometric_mask_reuses_existing_space(env, piece):
    select_piece(env, piece)
    env.board[17, 4:14] = 2
    adapter = TetrisAdapter(env)
    state = read_state(env)
    expected = TargetSpace(adapter.geometry).action_mask(env.board, adapter.read_current_piece().spawn_matrix)
    np.testing.assert_array_equal(geometric_mask(state), expected)
    np.testing.assert_array_equal(geometric_mask(state), geometric_mask(read_state(env)))


def test_noncanonical_pose_rejected(env):
    env.step(env.actions.move_right)
    with pytest.raises(ValueError, match='canonical'):
        assert_canonical(env)


@pytest.mark.parametrize('mode', ['minimal', 'shaped'])
def test_reward_terminal_and_truncation(mode):
    board = np.zeros((20, 10), dtype=np.uint8)
    board[17, 0] = 1  # Two holes, including side-open cells.
    state = semantic(board)
    assert hole_count(state) == 2
    phi = -0.25 * 2 / 12
    assert potential(state) == pytest.approx(phi)
    terminal = reward_components(state, state, lines=2, terminated=True, mode=mode)
    assert terminal.line == 2 and terminal.death == -1
    assert terminal.total == pytest.approx(1 + (-phi if mode == 'shaped' else 0))
    limited = reward_components(state, state, lines=2, terminated=False, mode=mode)
    assert limited.death == 0
    assert limited.total == pytest.approx(2 + (-.01 * phi if mode == 'shaped' else 0))


def test_pbrs_self_loop_is_not_zero():
    board = np.zeros((20, 10), dtype=np.uint8)
    board[0, 0] = 1
    state = semantic(board)
    result = reward_components(state, state, unreachable=True, mode='shaped')
    assert result.shaping == pytest.approx((.99 - 1) * potential(state))
    assert result.shaping > 0 and -.05 < result.total < -.0475
    assert reward_components(state, state, unreachable=True).total == -.05


@pytest.mark.parametrize('algorithm,expected', [('vanilla', 7.0), ('double', 3.0)])
def test_handcrafted_masked_bellman(algorithm, expected):
    online = torch.tensor([[2., 9., 1000.]] * 3, requires_grad=True)
    target = torch.tensor([[7., 3., 2000.]] * 3, requires_grad=True)
    mask = torch.tensor([[True, True, False], [False, False, False], [True, True, False]])
    # Rows: normal, terminal with empty mask, truncated (still bootstraps).
    result = bellman_targets(torch.tensor([1., -1., 2.]), torch.tensor([False, True, False]),
                             mask, target, online, algorithm=algorithm)
    torch.testing.assert_close(result, torch.tensor([1 + .99 * expected, -1, 2 + .99 * expected]))
    assert not result.requires_grad


def test_empty_nonterminal_bootstrap_is_diagnostic():
    with pytest.raises(ValueError, match='valid'):
        bellman_targets(torch.zeros(1), torch.tensor([False]), torch.zeros(1, 800, dtype=torch.bool),
                        torch.zeros(1, 800), torch.zeros(1, 800), algorithm='double')


from tetris_agent.rl.config import TrainingConfig

def test_epsilon_decay_override():
    # budget=100000, override=None -> decay 38000
    c1 = TrainingConfig(decision_budget=100000)
    assert c1.epsilon_decay_decisions == 38000

    # budget=500000, override=None -> decay 198000
    c2 = TrainingConfig(decision_budget=500000)
    assert c2.epsilon_decay_decisions == 198000

    # budget=500000, override=38000 -> decay 38000
    c3 = TrainingConfig(decision_budget=500000, epsilon_decay_override=38000)
    assert c3.epsilon_decay_decisions == 38000

    # invalid 0/negative override rejected
    with pytest.raises(ValueError, match='positive integer'):
        TrainingConfig(decision_budget=500000, epsilon_decay_override=0)
    with pytest.raises(ValueError, match='positive integer'):
        TrainingConfig(decision_budget=500000, epsilon_decay_override=-100)

    # epsilon floor timing이 기존 100K schedule과 동일
    assert c1.epsilon(c1.learning_starts + 38000) == pytest.approx(0.05)
    assert c3.epsilon(c3.learning_starts + 38000) == pytest.approx(0.05)
    assert c1.epsilon(c1.learning_starts + 38000) == c3.epsilon(c3.learning_starts + 38000)
