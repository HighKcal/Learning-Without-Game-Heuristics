"""Exact batched geometric subset predicate; no state/mask result cache."""
import numpy as np
from .state import _MASK_GEOMETRY


def _immutable_table(index):
    array = np.stack([g[index] for g in _MASK_GEOMETRY])
    return np.frombuffer(array.tobytes(),dtype=array.dtype).reshape(array.shape)


_CELLS, _BELOW, _INSIDE, _BELOW_INSIDE = tuple(_immutable_table(i) for i in range(4))


def geometric_subsets(states, masks):
    """Equivalent to [not any(mask & ~geometric_mask(state))] for each row.

    Only True mask slots can violate subset membership. Evaluate their complete
    bounds/collision/support predicate with the existing static cell tables.
    This is validation of every supplied action, never reachability pruning.
    Inputs are re-read on every call, so replay corruption detection is retained.
    """
    masks = np.asarray(masks)
    if masks.dtype != np.bool_ or masks.shape != (len(states),800):
        raise ValueError('Expected batched boolean masks of shape (N,800).')
    valid = np.ones(len(states),dtype=bool)
    if not states:
        return valid
    rows, actions = np.nonzero(masks)
    if not len(rows):
        return valid
    boards = np.stack([s.board for s in states]).reshape(len(states),200)
    pieces = np.array([s.current_piece for s in states],dtype=np.intp)[rows]
    cells,below = _CELLS[pieces,actions], _BELOW[pieces,actions]
    clear = ~boards[rows[:,None],cells].any(axis=1)
    supported = ~_BELOW_INSIDE[pieces,actions] | boards[rows[:,None],below].any(axis=1)
    good = _INSIDE[pieces,actions] & clear & supported
    valid[rows[~good]] = False
    return valid
