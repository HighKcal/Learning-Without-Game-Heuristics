"""Bounded, immutable geometry cache. No board or reachability state."""
from dataclasses import dataclass
from functools import lru_cache

import numpy as np


@dataclass(frozen=True)
class PieceGeometry:
    matrices: tuple[np.ndarray, ...]
    offsets: tuple[tuple[int, int], ...]
    unique: tuple[int, ...]


@lru_cache(maxsize=128)
def _geometry(shape, dtype, data):
    source = np.frombuffer(data, dtype=np.dtype(dtype)).reshape(shape)
    if not np.any(source > 0):
        raise ValueError('Expected a nonempty 2-D piece matrix.')
    matrices, offsets, unique, seen = [], [], [], set()
    for k in range(4):
        rotated = np.rot90(source, k)
        # Bytes backing makes the private cache irreversibly read-only.
        matrix = np.frombuffer(rotated.tobytes(), dtype=source.dtype).reshape(rotated.shape)
        occupied = matrix > 0
        rows, cols = np.nonzero(occupied)
        dx, dy = int(cols.min()), int(rows.min())
        cropped = occupied[dy:rows.max()+1, dx:cols.max()+1]
        key = cropped.shape, cropped.tobytes()
        if key not in seen:
            unique.append(k)
            seen.add(key)
        matrices.append(matrix)
        offsets.append((dx,dy))
    return PieceGeometry(tuple(matrices), tuple(offsets), tuple(unique))


def piece_geometry(matrix):
    """Numeric, small piece matrices only; unusual public inputs use fallback.

    Content keys include raw padding, dtype and values. Mutating a caller's
    source matrix selects a new entry. Public matrix APIs must return copies.
    """
    matrix = np.asarray(matrix)
    if matrix.ndim != 2 or matrix.size > 64 or matrix.dtype.kind not in 'buif':
        return None
    return _geometry(matrix.shape, matrix.dtype.str, matrix.tobytes())
