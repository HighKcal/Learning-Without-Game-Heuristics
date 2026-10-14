"""Tactical Planner v0.01: BFS over the installed Tetris transition rules."""

from collections import deque
from dataclasses import dataclass, fields
from enum import Enum
from importlib.metadata import version
from time import perf_counter

import gymnasium as gym
import numpy as np
from tetris_gymnasium.envs.tetris import Tetris, TetrisState
from tetris_gymnasium.components.tetromino import Tetromino
from tetris_gymnasium.components.tetromino_holder import TetrominoHolder
from tetris_gymnasium.components.tetromino_queue import TetrominoQueue

from .env_adapter import TetrisAdapter
from ._lock_verification import target_at_lock
from .target_space import (
    TargetPlacement, TargetSpace, orientation_matrices,
    target_to_env_origin, unique_orientations,
)


# Only the audited 0.3.1 implementation is eligible. Retain identities before
# callers install class/instance hooks; custom methods may consume observations.
_AUDITED_VERSION = version("tetris-gymnasium") == "0.3.1"
_NATIVE_METHODS = tuple((name, method) for name, method in vars(Tetris).items()
                        if callable(method))
_NATIVE_QUEUE_OBS = TetrominoQueue.get_queue
_NATIVE_HOLDER_OBS = TetrominoHolder.get_tetrominoes


def _can_skip_search_observation(base: Tetris) -> bool:
    if not _AUDITED_VERSION or type(base) is not Tetris:
        return False
    if any(name in vars(base) or getattr(Tetris, name) is not method
           for name, method in _NATIVE_METHODS):
        return False
    if (type(base.queue) is not TetrominoQueue
            or type(base.holder) is not TetrominoHolder
            or "get_queue" in vars(base.queue)
            or "get_tetrominoes" in vars(base.holder)
            or TetrominoQueue.get_queue is not _NATIVE_QUEUE_OBS
            or TetrominoHolder.get_tetrominoes is not _NATIVE_HOLDER_OBS):
        return False
    pieces = (*base.tetrominoes, base.active_tetromino, *base.holder.queue)
    return (type(base.board) is np.ndarray
            and all(type(piece) is Tetromino and type(piece.matrix) is np.ndarray
                    for piece in pieces))


def _search_observation() -> None:
    """Planner discards this slot; native step still returns all other values."""
    return None


class PlannerStatus(str, Enum):
    REACHABLE = "REACHABLE"
    UNREACHABLE = "UNREACHABLE"


PLANNER_SEMANTICS_VERSION = 'native-0.3.1-bfs-v1'
ORACLE_IMPLEMENTATION_VERSION = 'single-bfs-v1'


class EnumerationStatus(str, Enum):
    COMPLETE = 'COMPLETE'


@dataclass(frozen=True)
class ReachableTargetsResult:
    status: EnumerationStatus
    # Immutable witnesses, using exactly the same discovery order as plan().
    placements: tuple[tuple[TargetPlacement, 'PlannerResult'], ...]
    visited_states: int
    expanded_states: int
    elapsed_seconds: float

    @property
    def targets(self):
        return tuple(target for target, _ in self.placements)


@dataclass(frozen=True)
class PlannerResult:
    status: PlannerStatus
    # Native env.actions IDs; HARD_DROP is never included in this tuple.
    actions: tuple[int, ...]
    needs_hard_drop: bool
    visited_states: int
    expanded_states: int
    elapsed_seconds: float

    @property
    def reachable(self) -> bool:
        return self.status == PlannerStatus.REACHABLE

    @property
    def path_length(self) -> int:
        return len(self.actions)


def _state_key(state: TetrisState | Tetris) -> tuple:
    """RAW origin and full, immutable actual matrix, including blank borders.

    No post-lock state is enqueued. Board, piece ID, queue/RNG, holder, swap
    flag, gravity configuration and game_over therefore stay fixed throughout
    this single-piece search. There is no gravity timer or lock delay in 0.3.1.
    """
    matrix = state.active_tetromino.matrix
    return int(state.x), int(state.y), matrix.shape, matrix.dtype.str, matrix.tobytes()


def _supported_slice(matrix: np.ndarray, x: int, y: int,
                     board_shape: tuple[int, int]) -> bool:
    """The native collision/projection APIs require the whole raw slice to fit."""
    h, w = matrix.shape
    return 0 <= x and 0 <= y and x + w <= board_shape[1] and y + h <= board_shape[0]


class TacticalPlanner:
    """Find a shortest primitive prefix for a geometrically valid final target.

    plan() uses env.unwrapped.step(), so Gym wrappers (e.g. TimeLimit or action
    remapping) are outside this physical planner's semantics. Results use native
    action IDs. A reachable result either requires one final HARD_DROP or has
    already locked on its last primitive action. A lock remains a success even
    if spawning the next piece ends the game.

    The base game state and the temporary lock observer are restored even on
    exceptions. Calls on the same environment must be serialized.
    """

    def plan(self, env: gym.Env, target: TargetPlacement) -> PlannerResult:
        return self._search(env, target)

    def enumerate_reachable_targets(self, env: gym.Env) -> ReachableTargetsResult:
        """Exhaust one native BFS; exceptions never produce a partial result.

        No Hold/SWAP, wrapper steps, or post-lock states. Full padded matrices
        must match a unique absolute orientation, just as in plan(). Witnesses
        can go straight to the existing Executor without another search.
        """
        return self._search(env, None)

    def _search(self, env: gym.Env, target: TargetPlacement | None):
        started = perf_counter()
        adapter = TetrisAdapter(env)
        base = env.unwrapped
        piece = adapter.read_current_piece()  # Reset is required.
        original = base.get_state()
        # Native get_state() copies queue deques without maxlen, and set_state()
        # assigns an instance-level score attribute. Retain the untouched live
        # objects as well, so restoration preserves their identities/metadata.
        original_refs = TetrisState(**{
            field.name: getattr(base, field.name) for field in fields(TetrisState)
        })
        had_score = "score" in vars(base)
        visited: set[tuple] = set()
        expanded = 0
        # Preserve even a pre-existing instance override of this method.
        had_observer = "place_active_tetromino" in vars(base)
        previous_observer = vars(base).get("place_active_tetromino")
        place = base.place_active_tetromino
        locked_target: TargetPlacement | None = None
        did_lock = False
        skipped_observation = False
        space = TargetSpace(adapter.geometry)
        placements = {}
        matrices = orientation_matrices(piece.spawn_matrix)
        orientations = unique_orientations(piece.spawn_matrix)
        orientation_keys = {(matrices[k].shape, matrices[k].dtype.str,
                             matrices[k].tobytes()): k for k in orientations}

        def enumeration_result():
            return ReachableTargetsResult(EnumerationStatus.COMPLETE,
                tuple(placements.items()), len(visited), expanded, perf_counter() - started)

        def found(path=(), needs_hard_drop=False):
            if locked_target is None:
                return False
            if target is not None:
                return locked_target == target
            if (locked_target not in placements
                    and space.is_valid(original.board, piece.spawn_matrix, locked_target)):
                placements[locked_target] = PlannerResult(PlannerStatus.REACHABLE,
                    path, needs_hard_drop, len(visited), expanded, 0.0)
            return False

        def result(reachable: bool, actions: tuple[int, ...] = (),
                   needs_hard_drop: bool = False) -> PlannerResult:
            return PlannerResult(
                PlannerStatus.REACHABLE if reachable else PlannerStatus.UNREACHABLE,
                actions, needs_hard_drop, len(visited), expanded,
                perf_counter() - started,
            )

        def restore_branch(state: TetrisState) -> None:
            # set_state installs references, not copies. Detach the live branch
            # before step() can mutate a saved board, queue or randomizer.
            base.set_state(state)
            base.set_state(base.get_state())

        try:
            if _can_skip_search_observation(base):
                base._get_obs = _search_observation
                skipped_observation = True
            if base.game_over:
                return result(False) if target is not None else enumeration_result()
            if target is not None:
                if not space.is_valid(original.board, piece.spawn_matrix, target):
                    return result(False)
                target_matrix = matrices[target.orientation]
                target_origin = target_to_env_origin(target, piece.spawn_matrix)
                if not _supported_slice(target_matrix, *target_origin, original.board.shape):
                    return result(False)
            if (not _supported_slice(piece.matrix, piece.x, piece.y, original.board.shape)
                    or base.collision(base.active_tetromino, piece.x, piece.y)):
                return result(False) if target is not None else enumeration_result()

            def observe_lock() -> None:
                nonlocal locked_target, did_lock
                # Called by native commit *after* its native drop and *before*
                # projection, row clearing and spawning the next active piece.
                did_lock = True
                if target is not None:
                    locked_target = target_at_lock(base, piece, target.orientation, target_matrix)
                else:
                    matrix = base.active_tetromino.matrix
                    k = orientation_keys.get((matrix.shape, matrix.dtype.str, matrix.tobytes()))
                    locked_target = (None if k is None else
                        target_at_lock(base, piece, k, matrices[k]))
                place()

            base.place_active_tetromino = observe_lock

            def transition(state: TetrisState, action: int) -> tuple[bool, bool]:
                nonlocal locked_target, did_lock
                restore_branch(state)
                locked_target = None
                did_lock = False
                _, _, terminated, truncated, _ = base.step(action)
                return did_lock, bool(terminated or truncated)

            visited.add(_state_key(original))
            transition(original, base.actions.hard_drop)
            if found(needs_hard_drop=True):
                return result(True, needs_hard_drop=True)

            candidates = (
                base.actions.move_left, base.actions.move_right, base.actions.move_down,
                base.actions.rotate_clockwise, base.actions.rotate_counterclockwise,
                base.actions.no_op,
            )
            frontier = deque([(original, ())])
            while frontier:
                state, path = frontier.popleft()
                expanded += 1
                for action in candidates:
                    did_lock, ended = transition(state, action)
                    next_path = path + (action,)
                    if did_lock:
                        if found(next_path):
                            return result(True, next_path)
                        continue  # Never search the next piece.
                    if ended:
                        continue
                    key = _state_key(base)
                    if key in visited:
                        continue
                    child = base.get_state()
                    visited.add(key)
                    # Test each new node at discovery. All depth-d goals have
                    # been checked before expanding a depth-d node, including
                    # those requiring HARD_DROP; an auto-lock child cannot
                    # preempt a shorter queued prefix.
                    transition(child, base.actions.hard_drop)
                    if found(next_path, needs_hard_drop=True):
                        return result(True, next_path, needs_hard_drop=True)
                    frontier.append((child, next_path))
            return result(False) if target is not None else enumeration_result()
        finally:
            if skipped_observation:
                vars(base).pop("_get_obs", None)
            if had_observer:
                base.place_active_tetromino = previous_observer
            else:
                vars(base).pop("place_active_tetromino", None)
            base.set_state(original_refs)
            if not had_score:
                vars(base).pop("score", None)
