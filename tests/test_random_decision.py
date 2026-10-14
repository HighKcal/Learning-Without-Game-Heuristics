import random

import numpy as np
import pytest
from tetris_gymnasium.envs.tetris import Tetris

from tetris_agent import RandomDecisionAgent, TacticalPlanner, TargetPlacement, TargetSpace, TetrisAdapter


@pytest.fixture
def setup():
    env = Tetris()
    env.reset(seed=42)
    adapter = TetrisAdapter(env)
    yield env, TargetSpace(adapter.geometry), adapter.read()
    env.close()


def test_only_geometric_valid_unmasked_targets_are_selected(setup):
    env, space, state = setup
    mask = space.action_mask(state.board, state.current_piece.spawn_matrix)
    agent = RandomDecisionAgent(17)
    selected = [agent.choose_from_mask(space, mask) for _ in range(200)]
    for decision in selected:
        assert mask[decision.action_id]
        assert space.decode(decision.action_id) == decision.target
        assert space.is_valid(env.board, state.current_piece.spawn_matrix, decision.target)
    assert len({d.action_id for d in selected}) > 1


def test_same_seed_same_sequence_and_global_rng_untouched(setup):
    _, space, state = setup
    ids = np.flatnonzero(space.action_mask(state.board, state.current_piece.spawn_matrix))
    before = random.getstate()
    first, second = RandomDecisionAgent(73), RandomDecisionAgent(73)
    assert [first.choose(space, ids) for _ in range(100)] == [
        second.choose(space, ids) for _ in range(100)
    ]
    assert random.getstate() == before


def test_single_candidate_and_duplicate_ids(setup):
    _, space, _ = setup
    for ids in ([721], [721, 721, 721]):
        decision = RandomDecisionAgent(0).choose(space, ids)
        assert decision.action_id == 721 and decision.target == space.decode(721)


def test_empty_candidates_controlled_failure(setup):
    _, space, _ = setup
    agent = RandomDecisionAgent(0)
    with pytest.raises(ValueError, match="No geometric valid"):
        agent.choose(space, [])
    with pytest.raises(ValueError, match="No geometric valid"):
        agent.choose_from_mask(space, np.zeros(space.n, dtype=bool))


@pytest.mark.parametrize("mask", [np.ones(800), np.ones((20, 40), dtype=bool), np.ones(799, dtype=bool)])
def test_bad_mask_rejected(setup, mask):
    with pytest.raises(ValueError, match="boolean geometric mask"):
        RandomDecisionAgent().choose_from_mask(setup[1], mask)


@pytest.mark.parametrize("ids,error", [([-1], ValueError), ([800], ValueError), ([1.5], TypeError)])
def test_invalid_ids_rejected(setup, ids, error):
    with pytest.raises(error):
        RandomDecisionAgent().choose(setup[1], ids)


def test_valid_unreachable_target_can_be_chosen_without_planner(setup, monkeypatch):
    env, space, _ = setup
    env.active_tetromino = env.tetrominoes[1]
    env.reset_tetromino_position()
    env.board[17, 4:14] = 2
    target = TargetPlacement(7, 18, 0)
    assert not TacticalPlanner().plan(env, target).reachable
    mask = space.action_mask(env.board, env.active_tetromino.matrix)
    assert mask[space.encode(target)]

    def forbidden(*args):
        pytest.fail("Random selector must not plan or step the environment")

    monkeypatch.setattr(TacticalPlanner, "plan", forbidden)
    monkeypatch.setattr(env, "step", forbidden)
    agent = RandomDecisionAgent(0)
    # The full geometric mask retains the cavity, not just a forced singleton.
    choices = [agent.choose_from_mask(space, mask) for _ in range(200)]
    assert any(d.target == target for d in choices)
