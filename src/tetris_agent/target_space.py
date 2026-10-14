"""Fixed target IDs and geometric resting validity. No reachability checks."""

from dataclasses import dataclass
from numbers import Integral

import numpy as np

from .env_adapter import BoardGeometry
from ._piece_geometry import piece_geometry


@dataclass(frozen=True)
class TargetPlacement:
    """Occupied bounding-box top-left in internal board coordinates.

    x is the leftmost occupied column; y is the topmost occupied row.
    orientation k refers to np.rot90(spawn_matrix, k), never a relative action.
    """

    x: int
    y: int
    orientation: int

    def __post_init__(self) -> None:
        if any(isinstance(v, bool) or not isinstance(v, Integral)
               for v in (self.x, self.y, self.orientation)):
            raise TypeError("Target coordinates and orientation must be integers.")


def orientation_matrices(spawn_matrix: np.ndarray) -> tuple[np.ndarray, ...]:
    """Keep the original matrix padding and environment rotation direction."""
    matrix = np.asarray(spawn_matrix)
    geometry = piece_geometry(matrix)
    if geometry is not None:
        return tuple(m.copy() for m in geometry.matrices)
    if matrix.ndim != 2 or not np.any(matrix > 0):
        raise ValueError("Expected a nonempty 2-D piece matrix.")
    return tuple(np.rot90(matrix, k).copy() for k in range(4))


def occupied_offset(matrix: np.ndarray) -> tuple[int, int]:
    """Return (min_col, min_row) of positive piece cells inside the matrix."""
    matrix = np.asarray(matrix)
    geometry = piece_geometry(matrix)
    if geometry is not None:
        return geometry.offsets[0]
    if matrix.ndim != 2 or not np.any(matrix > 0):
        raise ValueError("Expected a nonempty 2-D piece matrix.")
    rows, cols = np.nonzero(matrix > 0)
    return int(cols.min()), int(rows.min())


def _orientation_matrix(spawn_matrix: np.ndarray, orientation: int) -> np.ndarray:
    if isinstance(orientation, bool) or not isinstance(orientation, Integral):
        raise TypeError("orientation must be an integer.")
    if not 0 <= orientation < 4:
        raise ValueError("orientation must be in 0..3.")
    return orientation_matrices(spawn_matrix)[orientation]


def target_to_env_origin(target: TargetPlacement, spawn_matrix: np.ndarray) -> tuple[int, int]:
    """Convert logical bbox coordinates to the original padded matrix origin.

    Conversion does not test validity/reachability and may return a negative
    origin when blank matrix borders extend above the playable board.
    """
    dx, dy = occupied_offset(_orientation_matrix(spawn_matrix, target.orientation))
    return int(target.x - dx), int(target.y - dy)


def env_origin_to_target(env_x: int, env_y: int, orientation: int,
                         spawn_matrix: np.ndarray) -> TargetPlacement:
    """Convert a raw matrix pose to its occupied bbox pose, preserving its slot."""
    origin = TargetPlacement(env_x, env_y, orientation)
    dx, dy = occupied_offset(_orientation_matrix(spawn_matrix, orientation))
    return TargetPlacement(int(origin.x + dx), int(origin.y + dy), int(orientation))


def unique_orientations(spawn_matrix: np.ndarray) -> tuple[int, ...]:
    """Keep the first slot for each occupied shape, ignoring empty borders.

    I/S/Z half-turns have translated occupied cells within the padded matrix.
    Compare cropped shapes to recognize those duplicates without changing the
    actual matrices (or the meaning of x/y) used for placement.
    """
    geometry = piece_geometry(spawn_matrix)
    if geometry is not None:
        return geometry.unique
    seen: set[tuple[tuple[int, int], bytes]] = set()
    unique = []
    for k, matrix in enumerate(orientation_matrices(spawn_matrix)):
        occupied = matrix > 0
        rows, cols = np.nonzero(occupied)
        shape = occupied[rows.min():rows.max() + 1, cols.min():cols.max() + 1]
        signature = (shape.shape, shape.tobytes())
        if signature not in seen:
            unique.append(k)
            seen.add(signature)
    return tuple(unique)


@dataclass(frozen=True)
class TargetSpace:
    """Four slots per playable occupied-bbox coordinate, fixed for one geometry.

    action_id = ((y * width) + (x - padding)) * 4 + orientation.
    Padding bbox coordinates have no IDs. Overflow, overlap, floating placements,
    and duplicate orientations are still masked within the fixed space.
    Raw matrix origins are recovered using each orientation's occupied offset;
    a raw origin in padding does not exclude a geometrically valid placement.
    A different environment size requires a different TargetSpace.
    """

    geometry: BoardGeometry

    @property
    def n(self) -> int:
        return self.geometry.height * self.geometry.width * 4

    def decode(self, action_id: int) -> TargetPlacement:
        if isinstance(action_id, bool) or not isinstance(action_id, Integral):
            raise TypeError("action_id must be an integer.")
        if not 0 <= action_id < self.n:
            raise ValueError("action_id is outside the fixed target space.")
        cell, orientation = divmod(int(action_id), 4)
        y, column = divmod(cell, self.geometry.width)
        return TargetPlacement(column + self.geometry.padding, y, orientation)

    def encode(self, target: TargetPlacement) -> int:
        column = target.x - self.geometry.padding
        if not (0 <= column < self.geometry.width
                and 0 <= target.y < self.geometry.height
                and 0 <= target.orientation < 4):
            raise ValueError("Target is outside the fixed target space.")
        return int((target.y * self.geometry.width + column) * 4 + target.orientation)

    def _check_board(self, board: np.ndarray) -> np.ndarray:
        board = np.asarray(board)
        if board.shape != self.geometry.board_shape:
            raise ValueError("Board shape does not match this target space.")
        return board

    def _collision(self, board: np.ndarray, matrix: np.ndarray, x: int, y: int) -> bool:
        # x/y here are raw matrix origins. Only occupied cells determine
        # geometric collision; blank borders may extend outside the board.
        dy, dx = np.nonzero(matrix > 0)
        xs, ys = x + dx, y + dy
        # Enforce playable cell bounds even if a supplied board omits bedrock.
        if (np.any(xs < self.geometry.padding)
                or np.any(xs >= self.geometry.padding + self.geometry.width)
                or np.any(ys < 0)
                or np.any(ys >= self.geometry.height)):
            return True
        return bool(np.any(board[ys, xs] != 0))

    def _resting(self, board: np.ndarray, matrix: np.ndarray, target: TargetPlacement) -> bool:
        dx, dy = occupied_offset(matrix)
        env_x, env_y = target.x - dx, target.y - dy
        return (not self._collision(board, matrix, env_x, env_y)
                and self._collision(board, matrix, env_x, env_y + 1))

    def is_valid(self, board: np.ndarray, spawn_matrix: np.ndarray,
                 target: TargetPlacement) -> bool:
        """Check only bounds, overlap, unique shape, and support one cell below."""
        board = self._check_board(board)
        try:
            self.encode(target)
        except ValueError:
            return False
        if target.orientation not in unique_orientations(spawn_matrix):
            return False
        matrix = orientation_matrices(spawn_matrix)[target.orientation]
        return self._resting(board, matrix, target)

    def action_mask(self, board: np.ndarray, spawn_matrix: np.ndarray) -> np.ndarray:
        """True means geometrically valid; sealed/unreachable targets may be True."""
        board = self._check_board(board)
        matrices = orientation_matrices(spawn_matrix)
        unique = unique_orientations(spawn_matrix)
        mask = np.zeros(self.n, dtype=np.bool_)
        for action_id in range(self.n):
            target = self.decode(action_id)
            if target.orientation in unique:
                mask[action_id] = self._resting(board, matrices[target.orientation], target)
        return mask
