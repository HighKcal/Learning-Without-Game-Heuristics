"""Random end-to-end smoke test orchestration; this retry policy is NOT RL.

UNREACHABLE choices are tried once per turn, removed, and resampled here only
to exercise the architecture. A future RL agent must decide separately how
an unreachable decision contributes a learning signal.
"""

from dataclasses import dataclass
from enum import Enum
from math import ceil
from time import perf_counter

import gymnasium as gym
import numpy as np

from .env_adapter import TetrisAdapter
from .executor import ExecutionFailure, Executor
from .planner import PlannerResult, TacticalPlanner
from .random_decision import RandomDecision, RandomDecisionAgent
from .target_space import TargetSpace


class GameStopReason(str, Enum):
    TERMINATED = "TERMINATED"
    TRUNCATED = "TRUNCATED"
    NO_VALID_TARGETS = "NO_VALID_TARGETS"
    ALL_VALID_TARGETS_UNREACHABLE = "ALL_VALID_TARGETS_UNREACHABLE"
    EXECUTOR_FAILURE = "EXECUTOR_FAILURE"


@dataclass(frozen=True)
class DecisionAttempt:
    turn: int
    attempt: int
    piece_type: str
    decision: RandomDecision
    plan: PlannerResult
    remaining_candidates: int
    execution_success: bool | None
    failure_reason: ExecutionFailure | None
    lines_cleared: int
    terminated: bool
    truncated: bool
    error_message: str | None = None


@dataclass(frozen=True)
class GameSummary:
    pieces_placed: int
    lines_cleared: int
    decision_count: int
    unreachable_count: int
    executor_failure_count: int
    executor_failures: dict[str, int]
    game_duration: float
    stop_reason: GameStopReason
    terminated: bool
    truncated: bool
    attempts: tuple[DecisionAttempt, ...]

    @property
    def unreachable_rate(self) -> float:
        return self.unreachable_count / self.decision_count if self.decision_count else 0.0

    @property
    def planner_times_ms(self) -> tuple[float, ...]:
        # Every attempted plan, including UNREACHABLE; no second timer.
        return tuple(a.plan.elapsed_seconds * 1000 for a in self.attempts)

    @property
    def planner_avg_ms(self) -> float:
        times = self.planner_times_ms
        return sum(times) / len(times) if times else 0.0

    @property
    def planner_max_ms(self) -> float:
        return max(self.planner_times_ms, default=0.0)

    @property
    def planner_p95_ms(self) -> float:
        # Nearest-rank percentile: well-defined even for a single call.
        times = sorted(self.planner_times_ms)
        return times[ceil(0.95 * len(times)) - 1] if times else 0.0

    def format(self) -> str:
        failures = ", ".join(f"{reason}={count}" for reason, count in self.executor_failures.items())
        return (
            "Game Summary\n-------------\n"
            f"pieces placed: {self.pieces_placed}\n"
            f"lines cleared: {self.lines_cleared}\n"
            f"decisions: {self.decision_count}\n"
            f"unreachable: {self.unreachable_count}\n"
            f"unreachable rate: {self.unreachable_rate:.1%}\n"
            f"executor failures: {self.executor_failure_count}"
            f"{(' (' + failures + ')') if failures else ''}\n"
            f"planner avg: {self.planner_avg_ms:.3f} ms\n"
            f"planner p95: {self.planner_p95_ms:.3f} ms\n"
            f"planner max: {self.planner_max_ms:.3f} ms\n"
            f"game duration: {self.game_duration:.3f} s\n"
            f"stop reason: {self.stop_reason.value}"
        )


class GameRunner:
    """Run one random smoke-test game using the existing public APIs.

    run() resets by default. reset=False resumes an already reset environment
    (useful for controlled scenarios); callers must not pass an ended wrapper.
    The caller owns/closes env. Planner/Executor calls stay serialized.
    """

    def __init__(self, agent: RandomDecisionAgent | None = None,
                 planner: TacticalPlanner | None = None,
                 executor: Executor | None = None):
        self.agent = agent if agent is not None else RandomDecisionAgent()
        self.planner = planner if planner is not None else TacticalPlanner()
        self.executor = executor if executor is not None else Executor()

    def run(self, env: gym.Env, *, env_seed: int | None = None,
            reset: bool = True, verbose: bool = False) -> GameSummary:
        started = perf_counter()
        if reset:
            env.reset(seed=env_seed)
        adapter = TetrisAdapter(env)
        space = TargetSpace(adapter.geometry)
        attempts: list[DecisionAttempt] = []
        pieces = lines = unreachable = 0
        failures: dict[str, int] = {}
        terminated = bool(env.unwrapped.game_over)
        truncated = False
        stop = GameStopReason.TERMINATED

        while not (terminated or truncated):
            state = adapter.read()
            # Only geometric validity; no plan is made before random selection.
            mask = space.action_mask(state.board, state.current_piece.spawn_matrix)
            candidates = np.flatnonzero(mask).tolist()
            if not candidates:
                stop = GameStopReason.NO_VALID_TARGETS
                break
            turn, attempt = pieces + 1, 0
            while candidates:
                attempt += 1
                decision = self.agent.choose(space, candidates)
                # Sampling without replacement is a smoke-test runner policy.
                # Exhaustion, rather than any arbitrary retry limit, bounds it.
                candidates.remove(decision.action_id)
                plan = self.planner.plan(env, decision.target)
                execution = None
                cleared = 0
                if not plan.reachable:
                    unreachable += 1
                else:
                    # Count native row-clear results only during real execution.
                    # Normally these are also returned as step.info['lines_cleared'].
                    # Observing the direct source signal also counts a clear if
                    # a wrapper subsequently raises before returning step info.
                    base = env.unwrapped
                    had_clear = "clear_filled_rows" in vars(base)
                    previous_clear = vars(base).get("clear_filled_rows")
                    clear = base.clear_filled_rows

                    def observe_clear(board):
                        nonlocal cleared
                        cleared_board, count = clear(board)
                        cleared += int(count)
                        return cleared_board, count

                    base.clear_filled_rows = observe_clear
                    try:
                        execution = self.executor.execute(env, decision.target, plan)
                    finally:
                        if had_clear:
                            base.clear_filled_rows = previous_clear
                        else:
                            vars(base).pop("clear_filled_rows", None)
                    lines += cleared
                    terminated, truncated = execution.terminated, execution.truncated
                    if execution.success:
                        pieces += 1
                    else:
                        reason = execution.failure_reason.value
                        failures[reason] = failures.get(reason, 0) + 1

                record = DecisionAttempt(
                    turn, attempt, state.current_piece.piece_type, decision, plan,
                    len(candidates), execution.success if execution else None,
                    execution.failure_reason if execution else None, cleared,
                    terminated, truncated, execution.error_message if execution else None,
                )
                attempts.append(record)
                if verbose:
                    self._print_attempt(env, record)
                if execution is not None:
                    if not execution.success:
                        stop = GameStopReason.EXECUTOR_FAILURE
                    elif terminated:
                        stop = GameStopReason.TERMINATED
                    elif truncated:
                        stop = GameStopReason.TRUNCATED
                    break  # Success advances piece; failure never retries.
            else:
                stop = GameStopReason.ALL_VALID_TARGETS_UNREACHABLE
                break
            if failures:
                break

        return GameSummary(
            pieces, lines, len(attempts), unreachable, sum(failures.values()),
            failures, perf_counter() - started, stop, terminated, truncated,
            tuple(attempts),
        )

    @staticmethod
    def _print_attempt(env: gym.Env, record: DecisionAttempt) -> None:
        actions = env.unwrapped.actions
        names = {
            actions.move_left: "LEFT", actions.move_right: "RIGHT",
            actions.move_down: "DOWN", actions.rotate_clockwise: "CW",
            actions.rotate_counterclockwise: "CCW", actions.no_op: "NO_OP",
        }
        target = record.decision.target
        print(f"Turn {record.turn} / Attempt {record.attempt} | piece: {record.piece_type}")
        print(f"target id: {record.decision.action_id} | "
              f"target: ({target.x}, {target.y}, {target.orientation})")
        print(f"planner: {record.plan.status.value} | "
              f"elapsed: {record.plan.elapsed_seconds * 1000:.3f} ms | "
              f"actions: {' '.join(names[a] for a in record.plan.actions) or '[]'} | "
              f"needs hard drop: {record.plan.needs_hard_drop}")
        if record.execution_success is None:
            print(f"remaining candidates: {record.remaining_candidates}")
        else:
            status = "SUCCESS" if record.execution_success else record.failure_reason.value
            print(f"executor: {status} | lines cleared: {record.lines_cleared}")
            if record.error_message:
                print(f"error: {record.error_message}")
