"""Minimal reward and exact potential-based shaping, one discount per decision."""
from dataclasses import dataclass
import numpy as np
from .state import SemanticState


@dataclass(frozen=True)
class RewardComponents:
    line: float
    unreachable: float
    death: float
    shaping: float

    @property
    def total(self) -> float:
        return self.line + self.unreachable + self.death + self.shaping


def hole_count(state: SemanticState) -> int:
    occupied = state.board != 0
    return int(np.count_nonzero(np.maximum.accumulate(occupied, axis=0) & ~occupied))


def potential(state: SemanticState, *, terminated=False) -> float:
    if terminated:
        return 0.0
    h = hole_count(state)
    return -0.25 * h / (h + 10)


def reward_components(state, next_state, *, lines=0, unreachable=False,
                      terminated=False, mode='minimal', unreachable_penalty=-0.05) -> RewardComponents:
    if mode not in ('minimal', 'shaped'):
        raise ValueError('Reward mode must be minimal or shaped.')
    shaping = (.99 * potential(next_state, terminated=terminated) - potential(state)
               if mode == 'shaped' else 0.0)
    return RewardComponents(float(lines), unreachable_penalty if unreachable else 0.0,
                            -1.0 if terminated else 0.0, shaping)
