"""Seedable uniform, one-step replay with owned raw semantic states."""
from dataclasses import dataclass
from random import Random
import numpy as np

from .state import SemanticState, SPACE
from .oracle import immutable_mask


@dataclass(frozen=True)
class Transition:
    state: SemanticState
    action_id: int
    reward: float
    next_state: SemanticState
    terminated: bool
    truncated: bool
    state_reachable_mask: np.ndarray | None = None
    next_state_reachable_mask: np.ndarray | None = None

    def __post_init__(self):
        SPACE.decode(self.action_id)
        for name in ('state_reachable_mask', 'next_state_reachable_mask'):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, immutable_mask(value))

    def copy(self):
        return Transition(self.state.copy(), int(self.action_id), float(self.reward),
                          self.next_state.copy(), bool(self.terminated), bool(self.truncated),
                          self.state_reachable_mask, self.next_state_reachable_mask)


class ReplayBuffer:
    def __init__(self, capacity=100_000, seed=None):
        if not isinstance(capacity, int) or isinstance(capacity, bool) or capacity <= 0:
            raise ValueError('Replay capacity must be a positive integer.')
        self.capacity = capacity
        self._items: list[Transition] = []
        self._position = 0
        self._rng = Random(seed)

    def __len__(self):
        return len(self._items)

    def add(self, transition: Transition) -> None:
        owned = transition.copy()
        if len(self._items) < self.capacity:
            self._items.append(owned)
        else:
            self._items[self._position] = owned
        self._position = (self._position + 1) % self.capacity

    def sample(self, batch_size: int) -> list[Transition]:
        if not 0 < batch_size <= len(self):
            raise ValueError('Batch size must be positive and no larger than replay size.')
        return [t.copy() for t in self._rng.sample(self._items, batch_size)]
