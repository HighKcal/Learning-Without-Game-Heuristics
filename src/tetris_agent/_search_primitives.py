"""Exact geometry helpers used only inside an audited native search.

Native step/drop/commit/clear/spawn are unchanged. The rotation expression is
the 2-D view operation performed by np.rot90; collision tests exactly the same
positive matrix cells against positive board cells. Unsupported slices retain
native NumPy slicing/error behavior via fallback.
"""
from functools import lru_cache
import numpy as np
from tetris_gymnasium.components.tetromino import Tetromino


@lru_cache(maxsize=128)
def _positive_cells(shape, dtype, data):
    matrix = np.frombuffer(data,dtype=np.dtype(dtype)).reshape(shape)
    rows,cols = np.nonzero(matrix>0)
    return tuple(zip(rows.tolist(),cols.tolist()))


class SearchPrimitives:
    def __init__(self, base):
        self.base = base
        self.native_collision = base.collision

    def collision(self, tetromino, x, y):
        matrix = tetromino.matrix
        h,w = matrix.shape
        board = self.base.board
        if not (0 <= x and 0 <= y and x+w <= board.shape[1] and y+h <= board.shape[0]):
            return self.native_collision(tetromino,x,y)
        cells = _positive_cells(matrix.shape,matrix.dtype.str,matrix.tobytes())
        for dy,dx in cells:
            if board[y+dy,x+dx] > 0:
                return np.bool_(True)
        return np.bool_(False)

    @staticmethod
    def rotate(tetromino, clockwise=True):
        matrix = tetromino.matrix
        rotated = matrix.T[::-1,:] if clockwise else matrix.T[:,::-1]
        return Tetromino(tetromino.id,tetromino.color_rgb,rotated)

    def install(self):
        self.base.collision = self.collision
        self.base.rotate = self.rotate

    def close(self):
        vars(self.base).pop('collision',None)
        vars(self.base).pop('rotate',None)
