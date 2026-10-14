"""Decision-boundary experience generation, separate from the retry smoke runner."""
from time import perf_counter
from typing import Callable

from ..executor import Executor
from ..planner import TacticalPlanner, PlannerStatus
from .metrics import EpisodeMetrics
from .replay import Transition
from .reward import reward_components
from .state import SPACE, assert_canonical, geometric_mask, read_state
from .oracle import compute_boundary, OracleInvariantError


class RLRunner:
    """Run one episode; the caller owns env, policy and transition storage.

    Watchdogs operate at complete decision boundaries. A wrapper truncating a
    primitive prefix before verified lock is an execution interruption: no
    training transition can represent that noncanonical physical pose in v1.
    Use max_decisions instead of primitive-step TimeLimit for RL experiments.
    Planner exceptions/unknown statuses and execution failures stop the run
    with a diagnostic and never reach on_transition. No retry or auto-reset.
    """
    def __init__(self, policy, *, reward_mode='minimal', unreachable_penalty=-0.05,
                 planner=None, executor=None, reachability_mask_mode='geometric'):
        if reward_mode not in ('minimal', 'shaped'):
            raise ValueError('Unknown reward mode.')
        self.policy = policy
        self.reward_mode = reward_mode
        self.unreachable_penalty = unreachable_penalty
        self.planner = planner if planner is not None else TacticalPlanner()
        self.executor = executor if executor is not None else Executor()
        if reachability_mask_mode not in ('geometric', 'oracle'):
            raise ValueError('Unknown reachability mask mode.')
        if reachability_mask_mode == 'oracle' and reward_mode != 'minimal':
            raise ValueError('Clean oracle condition requires existing minimal reward.')
        self.reachability_mask_mode = reachability_mask_mode

    def run(self, env, *, max_decisions: int, env_seed=None, reset=True,
            on_transition: Callable[[Transition], object] | None = None,
            on_progress: Callable[[EpisodeMetrics], object] | None = None) -> EpisodeMetrics:
        if isinstance(max_decisions, bool) or not isinstance(max_decisions, int) or max_decisions <= 0:
            raise ValueError('max_decisions must be a positive integer.')
        if reset:
            env.reset(seed=env_seed)
        metrics = EpisodeMetrics()
        metrics.terminated = bool(env.unwrapped.game_over)
        # Observability only: expose the live accumulator, then completed decisions.
        # Callers must not mutate it. Failures are returned through the same object.
        if on_progress is not None:
            on_progress(metrics)
        if metrics.terminated:
            metrics.stop_reason = 'TERMINATED'
            return metrics
        boundary = None

        def oracle_boundary():
            value = compute_boundary(env, self.planner)
            metrics.oracle_computation_count += 1
            metrics.oracle_elapsed_seconds += value.elapsed_seconds
            metrics.oracle_visited_states += value.search.visited_states
            return value

        while metrics.decision_count < max_decisions:
            try:
                assert_canonical(env)
                state = read_state(env)
                mask = geometric_mask(state)
                if not mask.any():
                    raise ValueError('Nonterminal canonical state has no geometric valid actions.')
            except Exception as exc:
                metrics.fail('STATE_ERROR', f'{type(exc).__name__}: {exc}')
                break
            if self.reachability_mask_mode == 'oracle':
                try:
                    if boundary is None:
                        boundary = oracle_boundary()
                    mask = boundary.mask
                except Exception as exc:
                    metrics.fail('ORACLE_ERROR', f'{type(exc).__name__}: {exc}')
                    break
                metrics.geometric_valid_action_count += boundary.geometric_count
                metrics.reachable_action_count += int(mask.sum())
                metrics.masked_physical_u_count += boundary.geometric_count - int(mask.sum())
            metrics.decision_count += 1
            try:
                action_id = self.policy.select(state, mask.copy())
                target = SPACE.decode(action_id)
                if not mask[action_id]:
                    if boundary is not None and geometric_mask(state)[action_id]:
                        metrics.selected_physical_u_count += 1
                        raise OracleInvariantError('Policy selected physical U outside the oracle mask.')
                    raise ValueError('Policy selected a geometric-invalid action.')
            except Exception as exc:
                metrics.fail('ORACLE_INVARIANT_VIOLATION' if isinstance(exc, OracleInvariantError)
                             else 'POLICY_ERROR', f'{type(exc).__name__}: {exc}')
                break
            started = perf_counter()
            if boundary is None:
                metrics.planner_call_count += 1
            try:
                plan = (self.planner.plan(env, target) if boundary is None
                        else boundary.plan_for(action_id))
            except Exception as exc:
                metrics.planner_elapsed_seconds += perf_counter() - started
                metrics.fail('PLANNER_ERROR', f'{type(exc).__name__}: {exc}')
                break
            metrics.planner_elapsed_seconds += plan.elapsed_seconds
            if plan.status not in (PlannerStatus.REACHABLE, PlannerStatus.UNREACHABLE):
                metrics.fail('PLANNER_UNKNOWN', f'Uncertain planner status: {plan.status}')
                break
            unreachable = plan.status == PlannerStatus.UNREACHABLE
            if unreachable and boundary is not None:
                metrics.selected_physical_u_count += 1
                metrics.fail('ORACLE_INVARIANT_VIOLATION', 'Oracle selected confirmed physical UNREACHABLE.')
                break
            lines = 0
            if unreachable:
                # Planner restores exact native state. Do not call Executor,
                # dispatch live actions, or remove this ID from the next mask.
                next_state = state
            else:
                base = env.unwrapped
                had_clear = 'clear_filled_rows' in vars(base)
                previous_clear = vars(base).get('clear_filled_rows')
                clear = base.clear_filled_rows

                def observe_clear(board):
                    nonlocal lines
                    cleared_board, count = clear(board)
                    lines += int(count)
                    return cleared_board, count

                base.clear_filled_rows = observe_clear
                try:
                    execution = self.executor.execute(env, target, plan)
                except Exception as exc:
                    metrics.terminated = bool(base.game_over)
                    metrics.fail('EXECUTION_ERROR', f'{type(exc).__name__}: {exc}')
                    break
                finally:
                    metrics.lines_cleared += lines
                    if had_clear:
                        base.clear_filled_rows = previous_clear
                    else:
                        vars(base).pop('clear_filled_rows', None)
                metrics.terminated, metrics.truncated = execution.terminated, execution.truncated
                if not execution.success or not execution.target_verified:
                    metrics.fail('EXECUTION_ERROR',
                                 f'{execution.failure_reason}: {execution.error_message or "Unverified placement"}')
                    break
                metrics.pieces_placed += 1
                try:
                    # A game-over spawn can collide; its semantic state is still
                    # stored, but neither bootstrap nor next potential uses it.
                    if not metrics.terminated:
                        assert_canonical(env)
                    next_state = read_state(env)
                except Exception as exc:
                    metrics.fail('STATE_ERROR', f'{type(exc).__name__}: {exc}')
                    break
            if metrics.decision_count == max_decisions and not metrics.terminated:
                metrics.truncated = True
            next_boundary = None
            if boundary is not None and not metrics.terminated:
                try:
                    # Truncations bootstrap too; compute even at the watchdog.
                    next_boundary = oracle_boundary()
                except Exception as exc:
                    metrics.fail('ORACLE_ERROR', f'{type(exc).__name__}: {exc}')
                    break
            reward = reward_components(state, next_state, lines=lines, unreachable=unreachable,
                                       terminated=metrics.terminated, mode=self.reward_mode,
                                       unreachable_penalty=self.unreachable_penalty)
            transition = Transition(state, action_id, reward.total, next_state,
                                    metrics.terminated, metrics.truncated,
                                    boundary.mask if boundary else None,
                                    next_boundary.mask if next_boundary else None)
            if on_transition is not None:
                on_transition(transition)
            metrics.record(reward, unreachable)
            boundary = next_boundary
            if metrics.terminated or metrics.truncated:
                metrics.stop_reason = 'TERMINATED' if metrics.terminated else 'TRUNCATED'
            if on_progress is not None:
                on_progress(metrics)
            if metrics.terminated or metrics.truncated:
                break
        return metrics
