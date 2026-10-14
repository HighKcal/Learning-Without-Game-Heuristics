"""Foundation for a placement-based Tetris learning agent."""

from .env_adapter import BoardGeometry, CurrentPiece, EnvSnapshot, TetrisAdapter
from .target_space import (
    TargetPlacement, TargetSpace, env_origin_to_target, target_to_env_origin,
)
from .planner import (PlannerResult, PlannerStatus, TacticalPlanner,
                      ReachableTargetsResult, EnumerationStatus)
from .executor import ExecutionFailure, ExecutionResult, ExecutionStep, Executor
from .random_decision import RandomDecision, RandomDecisionAgent
from .game_runner import DecisionAttempt, GameRunner, GameStopReason, GameSummary

__all__ = [
    "BoardGeometry", "CurrentPiece", "EnvSnapshot", "TetrisAdapter",
    "TargetPlacement", "TargetSpace",
    "env_origin_to_target", "target_to_env_origin",
    "PlannerResult", "PlannerStatus", "TacticalPlanner",
    "ReachableTargetsResult", "EnumerationStatus",
    "ExecutionFailure", "ExecutionResult", "ExecutionStep", "Executor",
    "RandomDecision", "RandomDecisionAgent",
    "DecisionAttempt", "GameRunner", "GameStopReason", "GameSummary",
]
