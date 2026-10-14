"""Frozen defaults; an experiment's total decision budget is always explicit."""
from dataclasses import dataclass
from math import ceil, isfinite
import random
import numpy as np
import torch


@dataclass(frozen=True)
class TrainingConfig:
    decision_budget: int
    algorithm: str = 'double'
    reward_mode: str = 'minimal'
    seed: int = 42
    device: str = 'auto'
    learning_rate: float = 1e-4
    gamma: float = .99
    replay_capacity: int = 100_000
    batch_size: int = 128
    learning_starts: int = 5_000
    target_sync_updates: int = 5_000
    gradient_clip: float = 10.0
    unreachable_penalty: float = -0.05
    epsilon_decay_override: int | None = None
    reachability_mask_mode: str = 'geometric'

    def __post_init__(self):
        if self.reachability_mask_mode not in ('geometric', 'oracle'):
            raise ValueError('Unknown reachability mask mode.')
        if self.reachability_mask_mode == 'oracle' and (self.algorithm != 'double' or self.reward_mode != 'minimal'):
            raise ValueError('Clean oracle condition requires Double DQN and existing minimal reward.')
        if self.algorithm not in ('vanilla', 'double') or self.reward_mode not in ('minimal', 'shaped'):
            raise ValueError('Unknown algorithm or reward mode.')
        for name in ('decision_budget', 'replay_capacity', 'batch_size', 'learning_starts', 'target_sync_updates'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f'{name} must be a positive integer.')
        if self.epsilon_decay_override is not None:
            if isinstance(self.epsilon_decay_override, bool) or not isinstance(self.epsilon_decay_override, int) or self.epsilon_decay_override <= 0:
                raise ValueError('epsilon_decay_override must be a positive integer.')
        if self.batch_size > self.replay_capacity:
            raise ValueError('Batch size exceeds replay capacity.')
        if self.gamma != .99 or self.learning_rate <= 0 or self.gradient_clip <= 0:
            raise ValueError('v0.01 requires gamma=.99 and positive optimizer settings.')
        if not isfinite(self.unreachable_penalty) or self.unreachable_penalty >= 0:
            raise ValueError('unreachable_penalty must be a finite negative number.')

    @property
    def epsilon_decay_decisions(self):
        if self.epsilon_decay_override is not None:
            return self.epsilon_decay_override
        return max(1, ceil(.4 * max(0, self.decision_budget - self.learning_starts)))

    def epsilon(self, stored_experiences: int) -> float:
        """Called before selection: first 5,000 stored experiences use epsilon=1."""
        progress = max(0, stored_experiences - self.learning_starts)
        return 1.0 - .95 * min(1.0, progress / self.epsilon_decay_decisions)

    @property
    def torch_device(self):
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu') if self.device == 'auto' else torch.device(self.device)


def seed_everything(seed: int) -> None:
    """Explicit experiment initialization; policy/replay retain separate RNGs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
