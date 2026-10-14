"""Shared pre-clear lock-pose verification for Planner and Executor."""

import numpy as np

from .env_adapter import CurrentPiece
from .target_space import TargetPlacement, env_origin_to_target


def target_at_lock(base, piece: CurrentPiece, orientation: int,
                   target_matrix: np.ndarray) -> TargetPlacement | None:
    """Read at native placement, after drop and before clear/next spawn.

    Compare the full padded matrix so absolute orientation is not inferred
    from a cleared board or from the next active piece's type.
    """
    active = base.active_tetromino
    if (active.id == piece.env_id
            and np.array_equal(active.matrix, target_matrix)):
        return env_origin_to_target(
            base.x, base.y, orientation, piece.spawn_matrix
        )
    return None
