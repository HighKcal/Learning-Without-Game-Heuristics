"""Read-only access to the object-based Tetris-Gymnasium 0.3.1 environment."""

from dataclasses import dataclass

import gymnasium as gym
import numpy as np
from tetris_gymnasium.envs.tetris import Tetris


@dataclass(frozen=True)
class BoardGeometry:
    width: int
    height: int
    padding: int

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0 or self.padding <= 0:
            raise ValueError("Board dimensions and padding must be positive.")

    @property
    def board_shape(self) -> tuple[int, int]:
        return self.height + self.padding, self.width + 2 * self.padding


@dataclass(frozen=True)
class CurrentPiece:
    piece_type: str
    env_id: int
    x: int
    y: int
    orientation: int
    matrix: np.ndarray
    spawn_matrix: np.ndarray


@dataclass(frozen=True)
class EnvSnapshot:
    # Padded, locked board only; the active piece is stored separately.
    board: np.ndarray
    current_piece: CurrentPiece
    visible_queue: tuple[str, ...]


class TetrisAdapter:
    """Read copies of board/piece data, without stepping or changing the environment.

    orientation is an absolute pose label relative to the spawn matrix:
    matrix = np.rot90(spawn_matrix, k=orientation). This follows the installed
    environment's rotation implementation, not a requested number of actions.
    """

    def __init__(self, env: gym.Env):
        base = env.unwrapped
        if not isinstance(base, Tetris):
            raise TypeError("Expected the object-based Tetris environment.")
        self._env = base
        self.geometry = BoardGeometry(base.width, base.height, base.padding)
        # Identify standard pieces by their actual spawn matrices, not offset IDs.
        names = ("I", "O", "T", "S", "Z", "J", "L")
        self._pieces: dict[int, tuple[str, np.ndarray]] = {}
        self._queue_types: list[str] = []
        for tetromino in base.tetrominoes:
            occupied = tetromino.matrix > 0
            matches = [
                name for name, template in zip(names, Tetris.TETROMINOES)
                if np.array_equal(occupied, template.matrix > 0)
            ]
            if len(matches) != 1:
                raise ValueError("This adapter supports the seven standard spawn matrices.")
            self._pieces[int(tetromino.id)] = (matches[0], tetromino.matrix.copy())
            self._queue_types.append(matches[0])

    def read_board(self) -> np.ndarray:
        """Return the padded locked board, excluding the moving piece."""
        return self._env.board.copy()

    def read_current_piece(self) -> CurrentPiece:
        active = self._env.active_tetromino
        if active is None:
            raise RuntimeError("Reset the environment before reading the active piece.")
        name, spawn_matrix = self._pieces[int(active.id)]
        orientation = next(
            (k for k in range(4)
             if np.array_equal(active.matrix, np.rot90(spawn_matrix, k))),
            None,
        )
        if orientation is None:
            raise ValueError("Active matrix is not a rotation of its spawn matrix.")
        return CurrentPiece(
            name, int(active.id), int(self._env.x), int(self._env.y), orientation,
            active.matrix.copy(), spawn_matrix.copy(),
        )

    def read_visible_queue(self) -> tuple[str, ...]:
        """Return every visible future piece in next-to-spawn order."""
        return tuple(self._queue_types[int(i)] for i in self._env.queue.get_queue())

    def read(self) -> EnvSnapshot:
        return EnvSnapshot(
            self.read_board(), self.read_current_piece(), self.read_visible_queue()
        )
