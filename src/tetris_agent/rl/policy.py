"""Policies see semantic state and an allowed-action mask, never a planner."""
from random import Random
import numpy as np
import torch
from .state import encode_states


def valid_ids(mask):
    mask = np.asarray(mask)
    if mask.dtype != np.bool_ or mask.shape != (800,):
        raise ValueError('Expected boolean geometric mask of shape (800,).')
    ids = np.flatnonzero(mask).tolist()
    if not ids:
        raise ValueError('No geometric valid actions in nonterminal state.')
    return ids


class FormalRandomPolicy:
    """Independent uniform draw on every call, including repeated self-loops."""
    def __init__(self, seed=None):
        self._rng = Random(seed)

    def select(self, state, mask) -> int:
        return self._rng.choice(valid_ids(mask))


class EpsilonGreedyPolicy:
    def __init__(self, network, seed=None):
        self.network = network
        self._rng = Random(seed)

    @torch.no_grad()
    def select(self, state, mask, epsilon=0.0) -> int:
        ids = valid_ids(mask)
        if not 0 <= epsilon <= 1:
            raise ValueError('Epsilon must be in [0, 1].')
        if self._rng.random() < epsilon:
            return self._rng.choice(ids)
        device = next(self.network.parameters()).device
        q = self.network(encode_states([state], device))[0]
        # Replay/oracle arrays have immutable bytes backing; avoid tensor aliases.
        array = np.asarray(mask)
        allowed = torch.as_tensor(array if array.flags.writeable else array.copy(),
                                  dtype=torch.bool, device=device)
        return int(q.masked_fill(~allowed, -torch.inf).argmax().item())
