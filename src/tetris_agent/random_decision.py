"""Uniform selection from supplied geometric candidates, without planning."""

from dataclasses import dataclass
from random import Random
from typing import Sequence

import numpy as np

from .target_space import TargetPlacement, TargetSpace


@dataclass(frozen=True)
class RandomDecision:
    action_id: int
    target: TargetPlacement


class RandomDecisionAgent:
    """Select only; the caller supplies geometric validity, never reachability.

    This agent has no environment or planner dependency. An owned RNG avoids
    changing Python's global random state. Candidate order must be stable for
    seeded reproduction. Empty candidates raise a controlled ValueError.
    """

    def __init__(self, seed: int | None = None):
        self._rng = Random(seed)

    def choose(self, space: TargetSpace,
               valid_action_ids: Sequence[int]) -> RandomDecision:
        if len(valid_action_ids) == 0:
            raise ValueError("No geometric valid targets to choose from.")
        # Deduplicate without reordering so every unique target is equiprobable.
        candidates = tuple(dict.fromkeys(valid_action_ids))
        # Validate IDs without inspecting the board or physical reachability.
        for action_id in candidates:
            space.decode(action_id)
        action_id = int(self._rng.choice(candidates))
        return RandomDecision(action_id, space.decode(action_id))

    def choose_from_mask(self, space: TargetSpace,
                         action_mask: np.ndarray) -> RandomDecision:
        mask = np.asarray(action_mask)
        if mask.shape != (space.n,) or mask.dtype != np.bool_:
            raise ValueError("Expected a boolean geometric mask of shape (space.n,).")
        return self.choose(space, np.flatnonzero(mask))
