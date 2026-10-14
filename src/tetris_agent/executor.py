"""Executor v0.01: execute one native plan and verify its original piece lock."""

from dataclasses import dataclass
from enum import Enum
from numbers import Integral
from typing import Any

import gymnasium as gym

from ._lock_verification import target_at_lock
from .env_adapter import TetrisAdapter
from .planner import PlannerResult
from .target_space import TargetPlacement, orientation_matrices, target_to_env_origin


class ExecutionFailure(str, Enum):
    UNREACHABLE = "UNREACHABLE"
    EARLY_LOCK = "EARLY_LOCK"
    PLAN_DIVERGED = "PLAN_DIVERGED"
    EPISODE_ENDED = "EPISODE_ENDED"
    TARGET_MISMATCH = "TARGET_MISMATCH"
    TRANSITION_ERROR = "TRANSITION_ERROR"


@dataclass(frozen=True)
class ExecutionStep:
    """One returned env.step result; reward is unmodified."""

    action: int
    observation: Any
    reward: float
    terminated: bool
    truncated: bool
    info: dict


@dataclass(frozen=True)
class ExecutionResult:
    success: bool
    # Dispatched native actions, including a call that raised after mutation.
    executed_actions: tuple[int, ...]
    terminated: bool
    truncated: bool
    raw_reward_sum: float
    target_verified: bool
    failure_reason: ExecutionFailure | None
    # Last returned step's data; None when no step returned.
    final_observation: Any
    final_info: dict | None
    steps: tuple[ExecutionStep, ...]
    error_message: str | None = None


class Executor:
    """Replay PlannerResult against the live environment, without recovery.

    Uses env.step() so TimeLimit/observation wrappers remain active. Native
    action IDs must retain their meaning: ActionWrapper is rejected before
    dispatch. Requires reset and a plan for the current live state. No state
    rollback, replanning, target selection, or reward shaping is performed.
    Calls on the same environment must be serialized, as in TacticalPlanner.
    """

    def execute(self, env: gym.Env, target: TargetPlacement,
                plan: PlannerResult) -> ExecutionResult:
        dispatched: list[int] = []
        steps: list[ExecutionStep] = []
        terminated = truncated = verified = False
        lock_count = 0
        locked_target = None

        def result(failure=None, error=None):
            return ExecutionResult(
                success=failure is None,
                executed_actions=tuple(dispatched),
                terminated=terminated, truncated=truncated,
                raw_reward_sum=sum((step.reward for step in steps), 0.0),
                target_verified=verified, failure_reason=failure,
                final_observation=steps[-1].observation if steps else None,
                final_info=steps[-1].info if steps else None,
                steps=tuple(steps), error_message=error,
            )

        # Do not even read or install a hook on an unreachable environment.
        if not plan.reachable:
            return result(ExecutionFailure.UNREACHABLE)

        try:
            base = env.unwrapped
            piece = TetrisAdapter(env).read_current_piece()
            if base.game_over:
                terminated = True
                return result(ExecutionFailure.EPISODE_ENDED)
            wrapped = env
            while isinstance(wrapped, gym.Wrapper):
                if isinstance(wrapped, gym.ActionWrapper):
                    return result(ExecutionFailure.PLAN_DIVERGED,
                                  "Planner native actions cannot use ActionWrapper.")
                wrapped = wrapped.env
            allowed = {
                base.actions.move_left, base.actions.move_right,
                base.actions.move_down, base.actions.rotate_clockwise,
                base.actions.rotate_counterclockwise, base.actions.no_op,
            }
            if any(isinstance(a, bool) or not isinstance(a, Integral)
                   or a not in allowed for a in plan.actions):
                return result(ExecutionFailure.PLAN_DIVERGED,
                              "Plan prefix must contain only native primitive actions.")
            actions = tuple(int(a) for a in plan.actions)
            if plan.needs_hard_drop:
                actions += (base.actions.hard_drop,)
            if not actions:
                return result(ExecutionFailure.PLAN_DIVERGED,
                              "An auto-lock plan requires a primitive action.")
            # Validate the absolute orientation using the existing converter.
            target_to_env_origin(target, piece.spawn_matrix)
            target_matrix = orientation_matrices(piece.spawn_matrix)[target.orientation]
        except Exception as exc:
            return result(ExecutionFailure.TRANSITION_ERROR,
                          f"{type(exc).__name__}: {exc}")

        had_observer = "place_active_tetromino" in vars(base)
        previous_observer = vars(base).get("place_active_tetromino")
        place = base.place_active_tetromino

        def observe_lock():
            nonlocal lock_count, locked_target
            pose = target_at_lock(base, piece, target.orientation, target_matrix)
            place()
            # A placement that raises is not a confirmed completed lock.
            lock_count += 1
            locked_target = pose

        base.place_active_tetromino = observe_lock
        try:
            for index, action in enumerate(actions):
                lock_count = 0
                locked_target = None
                dispatched.append(action)
                try:
                    obs, reward, ended, limited, info = env.step(action)
                    terminated, truncated = bool(ended), bool(limited)
                    steps.append(ExecutionStep(
                        action, obs, float(reward), terminated, truncated, info
                    ))
                except Exception as exc:
                    verified = lock_count == 1 and locked_target == target
                    # A raising step may already have changed the live game.
                    terminated = terminated or bool(base.game_over)
                    return result(ExecutionFailure.TRANSITION_ERROR,
                                  f"{type(exc).__name__}: {exc}")

                verified = lock_count == 1 and locked_target == target
                last = index == len(actions) - 1
                if lock_count:
                    if not last:
                        return result(ExecutionFailure.EARLY_LOCK)
                    if lock_count != 1:
                        return result(ExecutionFailure.PLAN_DIVERGED)
                    # Correct lock is success even if the next spawn ends play.
                    return result(None if verified else ExecutionFailure.TARGET_MISMATCH)
                if terminated or truncated:
                    return result(ExecutionFailure.EPISODE_ENDED)
            # The final promised drop/auto-lock did not actually place a piece.
            return result(ExecutionFailure.PLAN_DIVERGED)
        finally:
            if had_observer:
                base.place_active_tetromino = previous_observer
            else:
                vars(base).pop("place_active_tetromino", None)
