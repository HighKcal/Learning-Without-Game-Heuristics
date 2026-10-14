"""Frozen deployment: strategic Q ranking -> physical plan -> verified execution.

This module has no learner, reward, replay, or transition callback. Research
continues to use RLRunner unchanged. Calls on one environment are serialized.
"""
from dataclasses import dataclass, field
from math import ceil
from time import perf_counter

import numpy as np
import torch

from .executor import Executor
from .planner import PlannerStatus, TacticalPlanner
from .rl.evaluation import serialize_state, state_hash
from .rl.policy import valid_ids
from .rl.state import SPACE, assert_canonical, encode_states, geometric_mask, read_state


class FrozenQRankingPolicy:
    """One epsilon=0 forward; geometric mask only; ties use ascending action ID."""

    def __init__(self, network):
        self.network = network.eval().requires_grad_(False)

    @torch.inference_mode()
    def rank(self, state, mask):
        ids = valid_ids(mask)
        device = next(self.network.parameters()).device
        q = self.network(encode_states([state], device))[0]
        if q.shape != (800,):
            raise ValueError('Expected 800 Q values.')
        values = q.detach().cpu().numpy()
        if not np.isfinite(values[ids]).all():
            raise ValueError('Non-finite Q on a geometric-valid action.')
        # IDs start ascending; explicit tie key matches existing torch.argmax.
        return tuple((action, float(values[action])) for action in
                     sorted(ids, key=lambda action: (-float(values[action]), action)))


@dataclass
class PlayEpisode:
    decisions: list[dict] = field(default_factory=list)
    placements: int = 0
    lines: int = 0
    terminated: bool = False
    truncated: bool = False
    stop_reason: str | None = None
    diagnostic: str | None = None
    wall_seconds: float = 0.0

    def summary(self):
        successful = [d for d in self.decisions if d['executor_result'] == 'SUCCESS']
        ranks = sorted(d['selected_rank'] for d in successful)
        rejects = sum(d['rejected_u_count'] for d in self.decisions)
        return dict(placements=self.placements, lines=self.lines,
                    deaths=int(self.terminated), terminated=self.terminated,
                    truncated=self.truncated, stop_reason=self.stop_reason,
                    diagnostic=self.diagnostic, decisions=len(self.decisions),
                    total_u_rejects=rejects,
                    rejects_per_placement=rejects / self.placements if self.placements else None,
                    mean_reachable_rank=sum(ranks) / len(ranks) if ranks else None,
                    p95_reachable_rank=ranks[ceil(.95 * len(ranks)) - 1] if ranks else None,
                    # Depth means number of candidates through the selected R (rank).
                    max_fallback_depth=max(ranks, default=0),
                    max_u_rejects_per_decision=max(
                        (d['rejected_u_count'] for d in self.decisions), default=0),
                    planner_call_count=sum(d['planner_attempts'] for d in self.decisions),
                    planner_total_seconds=sum(d['planner_ms'] for d in self.decisions) / 1000,
                    candidate_search_total_seconds=sum(
                        d['candidate_search_ms'] for d in self.decisions) / 1000,
                    wall_seconds=self.wall_seconds)


class PlayRunner:
    """Try each geometric-valid Q candidate at most once at a decision boundary.

    Rejections are local variables of that decision, discarded after execution
    or failure. They cannot leak to another state, episode, or run. The first
    reachable PlannerResult is passed unchanged to Executor, without replanning.
    All uncertain outcomes stop the episode; only normal UNREACHABLE is skipped.
    """

    def __init__(self, policy: FrozenQRankingPolicy, *, planner=None, executor=None):
        self.policy = policy
        self.planner = planner if planner is not None else TacticalPlanner()
        self.executor = executor if executor is not None else Executor()

    def run(self, env, *, max_placements=100, env_seed=None, reset=True,
            on_decision=None) -> PlayEpisode:
        if (isinstance(max_placements, bool) or not isinstance(max_placements, int)
                or max_placements <= 0):
            raise ValueError('max_placements must be a positive integer.')
        started = perf_counter()
        episode = PlayEpisode()

        def fail(reason, message):
            episode.stop_reason = reason
            episode.diagnostic = message
            episode.terminated = bool(env.unwrapped.game_over)

        try:
            if reset:
                env.reset(seed=env_seed)
            episode.terminated = bool(env.unwrapped.game_over)
        except Exception as exc:
            fail('STATE_ERROR', f'{type(exc).__name__}: {exc}')
            episode.wall_seconds = perf_counter() - started
            return episode
        if episode.terminated:
            episode.stop_reason = 'TERMINATED'
        while not episode.stop_reason:
            search_started = perf_counter()
            record = dict(decision_id=len(episode.decisions) + 1, state_id=None,
                          state=None, geometric_valid_count=0, tested_count=0,
                          top_q_action=None, top_1_q=None, selected_action_id=None,
                          selected_rank=None, selected_q=None, rejected_action_ids=[],
                          rejected_u_count=0, planner_attempts=0, planner_ms=0.0,
                          selected_plan_ms=None, candidate_search_ms=0.0,
                          result_histogram={}, executor_result='NOT_RUN',
                          executed_actions=[], target_verified=False,
                          lines=0, terminated=False, truncated=False,
                          outcome=None, diagnostic=None)
            plan = target = None
            try:
                assert_canonical(env)
                state = read_state(env)
                record['state_id'] = state_hash(state)
                record['state'] = serialize_state(state)
                mask = geometric_mask(state)
                record['geometric_valid_count'] = int(mask.sum())
                if not mask.any():
                    raise ValueError('Nonterminal state has no geometric-valid actions.')
            except Exception as exc:
                fail('STATE_ERROR', f'{type(exc).__name__}: {exc}')
            if not episode.stop_reason:
                try:
                    ranking = self.policy.rank(state, mask.copy())
                    ids = [action for action, _ in ranking]
                    if len(ids) != len(set(ids)) or set(ids) != set(valid_ids(mask)):
                        raise ValueError('Ranking must contain every geometric-valid ID exactly once.')
                    if any(not np.isfinite(q) for _, q in ranking):
                        raise ValueError('Non-finite candidate Q.')
                    if tuple(ranking) != tuple(sorted(ranking, key=lambda item: (-item[1], item[0]))):
                        raise ValueError('Ranking must be descending Q with ascending ID ties.')
                    record['top_q_action'], record['top_1_q'] = ranking[0]
                except Exception as exc:
                    fail('POLICY_ERROR', f'{type(exc).__name__}: {exc}')
            if not episode.stop_reason:
                # This ordered list is the CURRENT DECISION ONLY rejected set.
                rejected = record['rejected_action_ids']
                for rank, (action, q) in enumerate(ranking, 1):
                    target = SPACE.decode(action)
                    attempt_started = perf_counter()
                    record['planner_attempts'] += 1
                    record['tested_count'] += 1
                    try:
                        candidate = self.planner.plan(env, target)
                    except Exception as exc:
                        status = 'SEARCH_TIMEOUT' if isinstance(exc, TimeoutError) else 'PLANNER_ERROR'
                        record['result_histogram'][status] = record['result_histogram'].get(status, 0) + 1
                        fail(status, f'action={action}: {type(exc).__name__}: {exc}')
                        break
                    finally:
                        record['planner_ms'] += (perf_counter() - attempt_started) * 1000
                    try:
                        # The audited Planner restores the live state. Never keep
                        # searching if an injected/custom planner violates it.
                        assert_canonical(env)
                        if state_hash(read_state(env)) != record['state_id']:
                            raise ValueError('Planner changed semantic state during search.')
                        status = candidate.status
                        label = str(getattr(status, 'value', status))
                        record['result_histogram'][label] = record['result_histogram'].get(label, 0) + 1
                        if status == PlannerStatus.UNREACHABLE:
                            rejected.append(action)
                            record['rejected_u_count'] += 1
                            continue
                        if status != PlannerStatus.REACHABLE:
                            fail('PLANNER_UNKNOWN', f'action={action}: uncertain planner status {label}')
                            break
                        plan = candidate  # Reuse the exact object, including its primitive path.
                        record['selected_action_id'] = action
                        record['selected_rank'] = rank
                        record['selected_q'] = q
                        record['selected_plan_ms'] = plan.elapsed_seconds * 1000
                        break
                    except Exception as exc:
                        fail('PLANNER_ERROR', f'action={action}: {type(exc).__name__}: {exc}')
                        break
                if not episode.stop_reason and plan is None:
                    fail('NO_REACHABLE_TARGET', 'All geometric-valid candidates are confirmed UNREACHABLE.')
            record['candidate_search_ms'] = (perf_counter() - search_started) * 1000
            if plan is not None and not episode.stop_reason:
                base = env.unwrapped
                had_clear = 'clear_filled_rows' in vars(base)
                previous_clear = vars(base).get('clear_filled_rows')
                clear = base.clear_filled_rows

                def observe_clear(board):
                    cleared, count = clear(board)
                    record['lines'] += int(count)
                    return cleared, count

                base.clear_filled_rows = observe_clear
                try:
                    execution = self.executor.execute(env, target, plan)
                    episode.terminated = bool(execution.terminated)
                    episode.truncated = bool(execution.truncated)
                    record['executed_actions'] = list(execution.executed_actions)
                    record['target_verified'] = bool(execution.target_verified)
                    if not execution.success or not execution.target_verified:
                        record['executor_result'] = 'EXECUTION_ERROR'
                        fail('EXECUTION_ERROR', f'{execution.failure_reason}: '
                             f'{execution.error_message or "Unverified placement"}')
                    else:
                        record['executor_result'] = 'SUCCESS'
                        episode.placements += 1
                        if episode.terminated or episode.truncated:
                            episode.stop_reason = 'TERMINATED' if episode.terminated else 'TRUNCATED'
                        else:
                            assert_canonical(env)
                            if episode.placements == max_placements:
                                episode.truncated = True
                                episode.stop_reason = 'PLACEMENT_LIMIT'
                except Exception as exc:
                    record['executor_result'] = 'EXECUTION_ERROR'
                    fail('EXECUTION_ERROR', f'{type(exc).__name__}: {exc}')
                finally:
                    episode.lines += record['lines']
                    if had_clear:
                        base.clear_filled_rows = previous_clear
                    else:
                        vars(base).pop('clear_filled_rows', None)
            record.update(outcome=episode.stop_reason or 'PLACED', diagnostic=episode.diagnostic,
                          terminated=episode.terminated, truncated=episode.truncated)
            episode.decisions.append(record)
            if on_decision is not None:
                on_decision(record)
        episode.wall_seconds = perf_counter() - started
        return episode
