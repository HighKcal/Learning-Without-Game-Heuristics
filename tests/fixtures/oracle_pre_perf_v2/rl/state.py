"""State v1 and geometric-only masks; no planner or hidden RNG inputs."""
from dataclasses import dataclass
from numbers import Integral
from typing import Sequence

import numpy as np
import torch
from tetris_gymnasium.envs.tetris import Tetris

from ..env_adapter import BoardGeometry, TetrisAdapter
from ..target_space import TargetSpace, orientation_matrices, unique_orientations


PIECE_NAMES = ('I', 'O', 'T', 'S', 'Z', 'J', 'L')
PIECE_IDS = {name: i for i, name in enumerate(PIECE_NAMES)}
GEOMETRY = BoardGeometry(width=10, height=20, padding=4)
SPACE = TargetSpace(GEOMETRY)
# Same standard spawn matrices used to identify pieces in TetrisAdapter.
_SPAWN_MATRICES = tuple((p.matrix > 0).astype(np.uint8) for p in Tetris.TETROMINOES)


def _mask_geometry(spawn):
    """Static occupied cells in action-ID order; no board or reachability data.

    State v1's unpadded board column 0 is TargetPlacement.x == 4. Normalize
    matrix padding to the occupied bbox, exactly as TargetSpace._resting does.
    Duplicate orientation slots remain present, but are never valid.
    """
    y, x = np.divmod(np.arange(SPACE.n) // 4, GEOMETRY.width)
    cells = np.zeros((SPACE.n, 4), dtype=np.intp)
    below = np.zeros_like(cells)
    inside = np.zeros(SPACE.n, dtype=bool)
    below_inside = np.zeros_like(inside)
    matrices = orientation_matrices(spawn)
    for k in unique_orientations(spawn):
        rows, cols = np.nonzero(matrices[k] > 0)
        rows, cols = rows - rows.min(), cols - cols.min()
        ys = y[k::4, None] + rows
        xs = x[k::4, None] + cols
        inside[k::4] = ((ys < GEOMETRY.height) & (xs < GEOMETRY.width)).all(axis=1)
        below_inside[k::4] = ((ys + 1 < GEOMETRY.height) & (xs < GEOMETRY.width)).all(axis=1)
        # Clipping only makes invalid gathers safe; inside flags reject them.
        cells[k::4] = np.clip(ys, 0, 19) * 10 + np.clip(xs, 0, 9)
        below[k::4] = np.clip(ys + 1, 0, 19) * 10 + np.clip(xs, 0, 9)
    for value in (cells, below, inside, below_inside):
        value.setflags(write=False)
    return cells, below, inside, below_inside


_MASK_GEOMETRY = tuple(_mask_geometry(spawn) for spawn in _SPAWN_MATRICES)


@dataclass(frozen=True, eq=False)
class SemanticState:
    board: np.ndarray
    current_piece: int
    next_queue: tuple[int, ...]

    def __post_init__(self):
        board = np.asarray(self.board)
        if board.shape != (20, 10) or not np.all((board == 0) | (board == 1)):
            raise ValueError('State v1 requires a binary locked 20x10 board.')
        ids = (self.current_piece, *self.next_queue)
        if len(self.next_queue) != 4 or any(
                isinstance(i, bool) or not isinstance(i, Integral) or not 0 <= i < 7 for i in ids):
            raise ValueError('Expected current piece and all four queue IDs in 0..6.')
        # Immutable bytes backing prevents even setflags(write=True) aliases.
        owned = np.frombuffer(board.astype(np.uint8).tobytes(), dtype=np.uint8).reshape(20, 10)
        object.__setattr__(self, 'board', owned)
        object.__setattr__(self, 'current_piece', int(self.current_piece))
        object.__setattr__(self, 'next_queue', tuple(map(int, self.next_queue)))

    def copy(self):
        return SemanticState(self.board, self.current_piece, self.next_queue)


def read_state(env) -> SemanticState:
    adapter = TetrisAdapter(env)
    if adapter.geometry != GEOMETRY:
        raise ValueError('State v1 supports only the standard 20x10 board with padding 4.')
    snapshot = adapter.read()
    return SemanticState((snapshot.board[:20, 4:14] != 0).astype(np.uint8),
                         PIECE_IDS[snapshot.current_piece.piece_type],
                         tuple(PIECE_IDS[p] for p in snapshot.visible_queue))


def assert_canonical(env) -> None:
    """Guard the audited native spawn pose; this data is never encoded."""
    base = env.unwrapped
    piece = TetrisAdapter(env).read_current_piece()
    spawn_x = base.width_padded // 2 - piece.spawn_matrix.shape[0] // 2
    if (base.game_over or piece.x != spawn_x or piece.y != 0
            or not np.array_equal(piece.matrix, piece.spawn_matrix)
            or base.has_swapped or base.collision(base.active_tetromino, piece.x, piece.y)):
        raise ValueError('RL decisions require a nonterminal canonical spawn state without Hold.')


def geometric_mask(state: SemanticState) -> np.ndarray:
    cells, below, inside, below_inside = _MASK_GEOMETRY[state.current_piece]
    board = state.board.reshape(-1)
    clear = ~board[cells].any(axis=1)
    supported = ~below_inside | board[below].any(axis=1)
    return inside & clear & supported


def encode_states(states: Sequence[SemanticState], device='cpu') -> torch.Tensor:
    if not states:
        return torch.empty((0, 235), dtype=torch.float32, device=device)
    boards = torch.as_tensor(np.stack([s.board for s in states]), device=device).float().flatten(1)
    ids = torch.tensor([(s.current_piece, *s.next_queue) for s in states], device=device)
    pieces = torch.nn.functional.one_hot(ids, num_classes=7).float().flatten(1)
    return torch.cat((boards, pieces), dim=1)
