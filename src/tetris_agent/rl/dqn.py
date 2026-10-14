"""Shared Vanilla/Double DQN; only next-action target selection differs."""
from copy import deepcopy
from dataclasses import dataclass
from time import perf_counter
import numpy as np
import torch
from torch.nn import functional as F

from .config import TrainingConfig, seed_everything
from .network import QNetwork
from .policy import EpsilonGreedyPolicy
from .replay import ReplayBuffer, Transition
from .state import encode_states, geometric_mask
from .oracle import OracleInvariantError
from ._oracle_validation import geometric_subsets


@torch.no_grad()
def bellman_targets(rewards, terminated, next_mask, target_q, online_q=None,
                    *, algorithm='double', gamma=.99):
    if algorithm not in ('vanilla', 'double'):
        raise ValueError('Unknown DQN algorithm.')
    active = ~terminated.bool()
    if torch.any(active & ~next_mask.any(dim=1)):
        raise ValueError('Nonterminal bootstrap state has no geometric valid actions.')
    # Do not compute -inf * 0 for terminal rows with empty masks.
    result = rewards.clone()
    if active.any():
        mask = next_mask[active]
        target = target_q[active]
        if algorithm == 'vanilla':
            value = target.masked_fill(~mask, -torch.inf).max(dim=1).values
        else:
            if online_q is None:
                raise ValueError('Double DQN requires online next Q-values.')
            selected = online_q[active].masked_fill(~mask, -torch.inf).argmax(dim=1)
            value = target.gather(1, selected[:, None]).squeeze(1)
        result[active] += gamma * value
    return result


@dataclass(frozen=True)
class LearningMetrics:
    loss: float
    mean_abs_td_error: float
    max_abs_td_error: float
    epsilon: float
    gradient_updates: int
    replay_size: int
    target_sync_count: int
    mean_selected_q: float
    max_q: float
    gradient_norm_before_clip: float
    max_selected_q: float


class DQNTrainer:
    """observe() stores one accepted decision and performs at most one update.

    Learning starts on stored experience #5,000 (also requires a full batch).
    Failed/uncertain decisions must never call observe. The hard-sync counter
    counts optimizer steps only, excluding the initial exact target copy.
    """
    def __init__(self, config: TrainingConfig):
        self.config = config
        seed_everything(config.seed)
        self.device = config.torch_device
        self.online = QNetwork().to(self.device)
        self.target = deepcopy(self.online).requires_grad_(False).eval()
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=config.learning_rate)
        self.replay = ReplayBuffer(config.replay_capacity, config.seed + 1)
        self.policy = EpsilonGreedyPolicy(self.online, config.seed + 2)
        self.stored_experiences = 0
        self.gradient_updates = 0
        self.target_sync_count = 0
        self.last_metrics: LearningMetrics | None = None
        self.learner_elapsed_seconds = 0.0

    @property
    def epsilon(self):
        return self.config.epsilon(self.stored_experiences)

    def select(self, state, mask):
        return self.policy.select(state, mask, self.epsilon)

    def observe(self, transition: Transition) -> LearningMetrics | None:
        if self.config.reachability_mask_mode == 'oracle':
            self._oracle_next_mask(transition)
        epsilon = self.epsilon
        self.replay.add(transition)
        self.stored_experiences += 1
        if (self.stored_experiences < self.config.learning_starts
                or len(self.replay) < self.config.batch_size):
            return None
        started = perf_counter()
        try:
            self.last_metrics = self._update(self.replay.sample(self.config.batch_size), epsilon)
        finally:
            self.learner_elapsed_seconds += perf_counter() - started
        return self.last_metrics

    def _update(self, batch, epsilon):
        states = encode_states([t.state for t in batch], self.device)
        next_states = encode_states([t.next_state for t in batch], self.device)
        terminated = torch.tensor([t.terminated for t in batch], dtype=torch.bool, device=self.device)
        if self.config.reachability_mask_mode == 'oracle':
            # Keep custom validation hooks on their original scalar path.
            masks = (self._oracle_next_masks(batch)
                     if self._oracle_next_mask is _SINGLE_ORACLE_VALIDATOR else
                     np.stack([self._oracle_next_mask(t) for t in batch]))
        else:
            masks = np.stack([np.zeros(800, dtype=bool) if t.terminated else geometric_mask(t.next_state)
                              for t in batch])
        next_mask = torch.as_tensor(masks, device=self.device)
        rewards = torch.tensor([t.reward for t in batch], dtype=torch.float32, device=self.device)
        actions = torch.tensor([t.action_id for t in batch], device=self.device)
        with torch.no_grad():
            targets = bellman_targets(rewards, terminated, next_mask, self.target(next_states),
                                      self.online(next_states) if self.config.algorithm == 'double' else None,
                                      algorithm=self.config.algorithm, gamma=self.config.gamma)
        all_q = self.online(states)
        selected_q = all_q.gather(1, actions[:, None]).squeeze(1)
        td = (targets - selected_q).detach().abs()
        loss = F.smooth_l1_loss(selected_q, targets)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(self.online.parameters(), self.config.gradient_clip)
        self.optimizer.step()
        self.gradient_updates += 1
        if self.gradient_updates % self.config.target_sync_updates == 0:
            self.target.load_state_dict(self.online.state_dict())
            self.target_sync_count += 1
        return LearningMetrics(float(loss.detach()), float(td.mean()), float(td.max()), epsilon,
                               self.gradient_updates, len(self.replay), self.target_sync_count,
                               float(selected_q.detach().mean()), float(all_q.detach().max()), float(norm),
                               float(selected_q.detach().max()))

    @staticmethod
    def _oracle_next_masks(batch):
        """Revalidate all sampled masks; preserve owned replay and error guards."""
        empty = np.zeros(800,dtype=bool)
        current = np.stack([empty if t.state_reachable_mask is None else
                            t.state_reachable_mask for t in batch])
        following = np.stack([empty if t.terminated or t.next_state_reachable_mask is None
                              else t.next_state_reachable_mask for t in batch])
        current_ok = geometric_subsets([t.state for t in batch],current)
        following_ok = geometric_subsets([t.next_state for t in batch],following)
        # Same row/error order as the scalar validator, including terminal and
        # truncated semantics. No validation is skipped based on prior inserts.
        for i,t in enumerate(batch):
            if (t.state_reachable_mask is None or not current[i,t.action_id]
                    or not current_ok[i]):
                raise OracleInvariantError('Replay action lacks a valid current oracle mask.')
            if not t.terminated and (t.next_state_reachable_mask is None
                    or not following[i].any() or not following_ok[i]):
                raise OracleInvariantError('Nonterminal replay lacks a valid next oracle mask.')
        return following

    @staticmethod
    def _oracle_next_mask(transition):
        current = transition.state_reachable_mask
        if (current is None or not current[transition.action_id]
                or np.any(current & ~geometric_mask(transition.state))):
            raise OracleInvariantError('Replay action lacks a valid current oracle mask.')
        if transition.terminated:
            return np.zeros(800, dtype=bool)
        mask = transition.next_state_reachable_mask
        if (mask is None or not mask.any()
                or np.any(mask & ~geometric_mask(transition.next_state))):
            raise OracleInvariantError('Nonterminal replay lacks a valid next oracle mask.')
        return mask


_SINGLE_ORACLE_VALIDATOR = DQNTrainer._oracle_next_mask
