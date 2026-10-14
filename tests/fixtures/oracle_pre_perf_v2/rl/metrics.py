"""Small accumulators; no per-primitive rewards or hidden reachability labels."""
from dataclasses import dataclass, field
from .reward import RewardComponents


@dataclass
class EpisodeMetrics:
    episode_return: float = 0.0
    lines_cleared: int = 0
    pieces_placed: int = 0
    decision_count: int = 0
    stored_transitions: int = 0
    unreachable_count: int = 0
    consecutive_unreachable: int = 0
    max_consecutive_unreachable: int = 0
    self_loop_count: int = 0
    terminated: bool = False
    truncated: bool = False
    planner_elapsed_seconds: float = 0.0
    planner_call_count: int = 0
    oracle_computation_count: int = 0
    oracle_elapsed_seconds: float = 0.0
    oracle_visited_states: int = 0
    geometric_valid_action_count: int = 0
    reachable_action_count: int = 0
    masked_physical_u_count: int = 0
    selected_physical_u_count: int = 0
    reward_totals: dict[str, float] = field(default_factory=lambda: dict(
        line=0.0, unreachable=0.0, death=0.0, shaping=0.0, total=0.0))
    stop_reason: str = ''
    diagnostic: str | None = None

    @property
    def unreachable_rate(self):
        return self.unreachable_count / self.decision_count if self.decision_count else 0.0

    @property
    def self_loop_ratio(self):
        return self.self_loop_count / self.stored_transitions if self.stored_transitions else 0.0

    @property
    def decisions_per_placement(self):
        return self.decision_count / self.pieces_placed if self.pieces_placed else None

    def record(self, reward: RewardComponents, unreachable: bool):
        self.stored_transitions += 1
        self.episode_return += reward.total
        for name in self.reward_totals:
            self.reward_totals[name] += getattr(reward, name)
        if unreachable:
            self.unreachable_count += 1
            self.self_loop_count += 1
            self.consecutive_unreachable += 1
            self.max_consecutive_unreachable = max(self.max_consecutive_unreachable,
                                                   self.consecutive_unreachable)
        else:
            self.consecutive_unreachable = 0

    def fail(self, reason, diagnostic):
        self.stop_reason = reason
        self.diagnostic = diagnostic
