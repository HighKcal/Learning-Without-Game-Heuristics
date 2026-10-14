import gymnasium as gym
import numpy as np
import pytest
from tetris_gymnasium.envs.tetris import Tetris

from tetris_agent import CurrentPiece, EnvSnapshot, TetrisAdapter


@pytest.fixture
def env():
    game = Tetris(gravity=False)
    yield game
    game.close()


def test_reset_and_read(env):
    obs, info = env.reset(seed=42)
    adapter = TetrisAdapter(gym.wrappers.TimeLimit(env, max_episode_steps=100))
    state = adapter.read()
    assert isinstance(state, EnvSnapshot)
    assert info == {"lines_cleared": 0}
    assert state.board.shape == adapter.geometry.board_shape == (24, 18)
    assert state.board.dtype == np.uint8
    np.testing.assert_array_equal(state.board, env.board)
    assert not np.array_equal(state.board, obs["board"])
    assert isinstance(state.current_piece, CurrentPiece)
    piece = state.current_piece
    assert piece.piece_type == "S"
    assert piece.env_id == env.active_tetromino.id
    assert isinstance(piece.x, int) and piece.x == env.x
    assert isinstance(piece.y, int) and piece.y == env.y
    assert piece.orientation == 0
    np.testing.assert_array_equal(piece.matrix, env.active_tetromino.matrix)
    np.testing.assert_array_equal(piece.spawn_matrix, piece.matrix)
    assert isinstance(state.visible_queue, tuple)
    assert state.visible_queue == ("T", "L", "Z", "O")
    assert len(state.visible_queue) == env.queue.size
    # Queue order and the complete visible queue agree with observation tiles.
    for i, queue_index in enumerate(env.queue.get_queue()):
        matrix = env.tetrominoes[queue_index].matrix
        tile = obs["queue"][:, i * env.padding:(i + 1) * env.padding]
        np.testing.assert_array_equal(tile[:matrix.shape[0], :matrix.shape[1]], matrix)


def test_read_requires_reset(env):
    with pytest.raises(RuntimeError, match="Reset"):
        TetrisAdapter(env).read_current_piece()


def test_returned_arrays_are_independent(env):
    env.reset(seed=42)
    adapter = TetrisAdapter(env)
    before = adapter.read()
    returned = adapter.read()
    returned.board[:] = 99
    returned.current_piece.matrix[:] = 99
    returned.current_piece.spawn_matrix[:] = 99
    after = adapter.read()
    np.testing.assert_array_equal(before.board, after.board)
    np.testing.assert_array_equal(before.current_piece.matrix, after.current_piece.matrix)
    np.testing.assert_array_equal(before.current_piece.spawn_matrix, after.current_piece.spawn_matrix)
    assert before.visible_queue == after.visible_queue


def test_locked_board_regression_reset(env):
    observation, _ = env.reset(seed=42)
    adapter = TetrisAdapter(env)
    locked = adapter.read().board
    piece = adapter.read_current_piece()
    np.testing.assert_array_equal(locked, env.unwrapped.board)
    assert not np.array_equal(locked, observation["board"])
    assert np.count_nonzero(locked[:env.height, env.padding:env.padding + env.width]) == 0
    # Exactly the occupied active cells, not the whole matrix rectangle, are
    # added to the observation; the adapter never treats them as obstacles.
    expected_observation = locked.copy()
    h, w = piece.matrix.shape
    expected_observation[piece.y:piece.y + h, piece.x:piece.x + w] += piece.matrix
    np.testing.assert_array_equal(observation["board"], expected_observation)


def test_locked_board_regression_move_right(env):
    observation_before, _ = env.reset(seed=42)
    adapter = TetrisAdapter(env)
    before = adapter.read()
    observation_after, _, terminated, truncated, _ = env.step(env.actions.move_right)
    after = adapter.read()
    assert not terminated and not truncated
    assert after.current_piece.x == before.current_piece.x + 1
    assert after.current_piece.y == before.current_piece.y  # fixture disables gravity
    assert not np.array_equal(observation_before["board"], observation_after["board"])
    np.testing.assert_array_equal(after.board, before.board)
    np.testing.assert_array_equal(after.board, env.unwrapped.board)
    assert not np.array_equal(after.board, observation_after["board"])


def test_locked_board_regression_hard_drop(env):
    env.reset(seed=42)
    adapter = TetrisAdapter(env)
    before = adapter.read()
    piece = before.current_piece
    # On an empty board the lowest occupied row rests on the last playable row.
    # This derives the expected lock position without search or state cloning.
    occupied_rows, occupied_cols = np.nonzero(piece.matrix > 0)
    resting_y = env.height - 1 - int(occupied_rows.max())
    expected_locked = before.board.copy()
    expected_locked[resting_y + occupied_rows, piece.x + occupied_cols] = piece.env_id

    observation, _, terminated, truncated, info = env.step(env.actions.hard_drop)
    after = adapter.read()
    assert not terminated and not truncated
    assert info["lines_cleared"] == 0
    assert not np.array_equal(after.board, before.board)
    np.testing.assert_array_equal(after.board, env.unwrapped.board)
    np.testing.assert_array_equal(after.board, expected_locked)
    assert np.count_nonzero(after.board == piece.env_id) == 4
    # The next piece is visible in the observation, but absent from locked board.
    assert after.current_piece.env_id != piece.env_id
    assert not np.array_equal(after.board, observation["board"])


@pytest.mark.parametrize("index", range(7))
def test_absolute_orientation_follows_environment_rotations(env, index):
    env.reset(seed=42)
    env.active_tetromino = env.tetrominoes[index]
    env.reset_tetromino_position()
    adapter = TetrisAdapter(env)
    spawn = adapter.read_current_piece().spawn_matrix
    for k in range(4):
        current = adapter.read_current_piece()
        expected = np.rot90(spawn, k)
        np.testing.assert_array_equal(current.matrix, expected)
        # The O matrix cannot distinguish rotation labels; use the first match.
        assert current.orientation == (0 if index == 1 else k)
        np.testing.assert_array_equal(current.spawn_matrix, spawn)
        env.step(env.actions.rotate_clockwise)
    assert adapter.read_current_piece().orientation == 0


def test_reads_entire_configured_visible_queue(env):
    from tetris_gymnasium.components.tetromino_queue import TetrominoQueue

    # Set up a longer visible queue before reset, without changing package code.
    env.queue = TetrominoQueue(env.randomizer, size=6)
    obs, _ = env.reset(seed=42)
    assert len(TetrisAdapter(env).read_visible_queue()) == 6
    assert obs["queue"].shape[1] == env.padding * 6


def test_custom_board_geometry():
    env = Tetris(width=6, height=12)
    try:
        env.reset(seed=42)
        adapter = TetrisAdapter(env)
        assert adapter.geometry.board_shape == (16, 14)
        assert adapter.read_board().shape == (16, 14)
    finally:
        env.close()
