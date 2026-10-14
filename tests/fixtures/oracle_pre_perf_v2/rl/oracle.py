"""Physical masks at live canonical boundaries; never a network feature."""
from dataclasses import dataclass
from time import perf_counter
from importlib.metadata import version
import numpy as np

from ..planner import (EnumerationStatus, ORACLE_IMPLEMENTATION_VERSION,
                       PLANNER_SEMANTICS_VERSION, ReachableTargetsResult)
from .state import SPACE, assert_canonical, geometric_mask, read_state


class OracleInvariantError(RuntimeError):
    pass


def immutable_mask(value):
    array = np.asarray(value)
    if array.dtype != np.bool_ or array.shape != (800,):
        raise ValueError('Expected boolean action mask of shape (800,).')
    return np.frombuffer(array.tobytes(), dtype=np.bool_)


def oracle_metadata(mode):
    return dict(reachability_mask_mode=mode,
                oracle_implementation_version=ORACLE_IMPLEMENTATION_VERSION,
                planner_semantics_version=PLANNER_SEMANTICS_VERSION,
                uncertainty_policy='abort_before_replay',
                replay_mask_format='optional_owned_bool800_v1',
                mask_timing='live_boundary_once_reused_for_next_acting_and_replay')


@dataclass(frozen=True)
class OracleBoundary:
    mask: np.ndarray
    search: ReachableTargetsResult
    geometric_count: int
    elapsed_seconds: float

    def plan_for(self, action_id):
        target = SPACE.decode(action_id)
        for placement, plan in self.search.placements:
            if placement == target:
                return plan
        raise OracleInvariantError('Selected action has no physical reachability witness.')


def compute_boundary(env, planner):
    started = perf_counter()
    if version('tetris-gymnasium') != '0.3.1':
        raise OracleInvariantError('Oracle condition requires the audited tetris-gymnasium 0.3.1.')
    assert_canonical(env)
    geometric = geometric_mask(read_state(env))
    search = planner.enumerate_reachable_targets(env)
    if search.status != EnumerationStatus.COMPLETE:
        raise OracleInvariantError(f'Incomplete oracle search: {search.status}')
    if search.expanded_states != search.visited_states or search.visited_states == 0:
        raise OracleInvariantError('Oracle search did not exhaust the canonical pose graph.')
    physical = np.zeros(800, dtype=bool)
    for target, plan in search.placements:
        if not plan.reachable:
            raise OracleInvariantError('Oracle returned a non-reachable witness.')
        physical[SPACE.encode(target)] = True
    if np.any(physical & ~geometric):
        raise OracleInvariantError('Physical mask contains a geometric-invalid action.')
    final = geometric & physical
    if not final.any():
        raise OracleInvariantError('Nonterminal boundary has no reachable actions.')
    return OracleBoundary(immutable_mask(final), search, int(geometric.sum()), perf_counter() - started)


def restore_semantic_boundary(env, state):
    """Offline canonical reconstruction under the audited native gravity rules.

    Queue/RNG identities do not affect this single-piece graph: search never
    enqueues a post-lock state, and a correct lock succeeds even on game over.
    This is NOT environment restoration for exact training resume.
    """
    base = env.unwrapped
    if not base.gravity_enabled:
        raise ValueError('Offline oracle reconstruction requires native gravity=True.')
    base.reset(seed=0)
    base.board[:20, 4:14] = state.board * 2
    base.active_tetromino = base.tetrominoes[state.current_piece]
    base.reset_tetromino_position()
    for index, piece in enumerate(state.next_queue):
        base.queue.queue[index] = piece
    assert_canonical(env)
