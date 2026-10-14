"""Seeded natural collection through the existing Planner -> Executor path.

This module is used only by the offline dataset builder. It is not a policy.
"""
from copy import deepcopy
from dataclasses import asdict, dataclass, fields
from hashlib import sha256
from importlib.metadata import version
from math import ceil, isfinite
from pathlib import Path
from random import Random
import subprocess
from time import perf_counter

import numpy as np
from tetris_gymnasium.envs.tetris import Tetris, TetrisState

from ..executor import Executor
from ..planner import PlannerStatus, TacticalPlanner
from .evaluation import (EvaluationRecord, dataset_fingerprint, label_statistics,
                         serialize_state, state_hash)
from .state import SPACE, assert_canonical, geometric_mask, read_state


@dataclass(frozen=True)
class BuildConfig:
    seed: int = 42
    state_count: int = 32
    max_candidates: int = 320
    max_seconds: float = 300.0

    def __post_init__(self):
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise ValueError('seed must be an integer in [0, 2**32 - 1].')
        for name in ('state_count', 'max_candidates'):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f'{name} must be a positive integer.')
        if (isinstance(self.max_seconds, bool) or not isfinite(self.max_seconds)
                or self.max_seconds <= 0):
            raise ValueError('max_seconds must be positive and finite.')


class CandidateRejected(RuntimeError):
    def __init__(self, reason, detail, action_id=None):
        self.reason, self.action_id = reason, action_id
        super().__init__(detail)


def _json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: _json_value(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(v) for v in value]
    return value


def native_context(env):
    """Diagnostic values only; never a SemanticState or a network feature."""
    def piece(value):
        return dict(id=int(value.id), matrix=value.matrix.tolist(),
                    dtype=value.matrix.dtype.str, color=list(value.color_rgb))
    base = env.unwrapped
    rng = base.randomizer
    return _json_value(dict(
        board=base.board.tolist(), board_dtype=base.board.dtype.str,
        active_piece=piece(base.active_tetromino), x=base.x, y=base.y,
        queue=list(base.queue.queue), queue_size=base.queue.size,
        queue_maxlen=base.queue.queue.maxlen, queue_rng_shared=base.queue.randomizer is rng,
        holder=[piece(p) for p in base.holder.queue], holder_size=base.holder.size,
        holder_maxlen=base.holder.queue.maxlen, has_swapped=base.has_swapped,
        game_over=base.game_over, randomizer_type=type(rng).__qualname__,
        randomizer_size=rng.size, randomizer_index=getattr(rng, 'index', None),
        randomizer_bag=getattr(rng, 'bag', None), randomizer_rng=rng.rng.bit_generator.state,
        gym_rng=base.np_random.bit_generator.state, gravity=base.gravity_enabled,
        geometry=[base.width, base.height, base.padding],
        tetrominoes=[piece(p) for p in base.tetrominoes]))


def checked_plan(env, target, planner, timings):
    """Audit native Planner restoration and protect the original live state.

    Native set_state installs references. A deepcopy of the native TetrisState
    detaches the working objects before planning, preserving original objects
    even if a faulty planner mutates its input or raises. No physics is copied.
    The planner must restore this exact working state on both return and error.
    """
    base = env.unwrapped
    original = TetrisState(**{f.name: getattr(base, f.name) for f in fields(TetrisState)})
    hooks = {name: (name in vars(base), vars(base).get(name))
             for name in ('score', 'place_active_tetromino', '_get_obs')}
    gym_rng = deepcopy(base.np_random.bit_generator.state)
    before_original = native_context(env)
    try:
        # Do not deepcopy a bound score method (which would copy its whole env).
        working = TetrisState(**deepcopy({f.name: getattr(original, f.name)
                                          for f in fields(TetrisState) if f.name != 'score'}),
                              score=original.score)
        base.set_state(working)
        if not hooks['score'][0]:
            vars(base).pop('score', None)
        before = native_context(env)
        refs = {name: getattr(base, name) for name in
                ('board', 'active_tetromino', 'queue', 'holder', 'randomizer')}
        error = None
        started = perf_counter()
        try:
            result = planner.plan(env, target)
        except Exception as exc:
            error = exc
        finally:
            timings.append(perf_counter() - started)
        if (native_context(env) != before
                or any(getattr(base, name) is not value for name, value in refs.items())
                or any((name in vars(base), vars(base).get(name)) != saved
                       for name, saved in hooks.items())):
            raise CandidateRejected('RESTORATION_FAILURE', 'Planner did not restore exact native state.')
        if error is not None:
            raise CandidateRejected('PLANNER_TIMEOUT' if isinstance(error, TimeoutError) else 'PLANNER_ERROR',
                                    f'{type(error).__name__}: {error}') from error
        if result.status not in (PlannerStatus.REACHABLE, PlannerStatus.UNREACHABLE):
            raise CandidateRejected('PLANNER_UNKNOWN', f'Non-binary status: {result.status}')
        return result
    finally:
        base.set_state(original)
        base.np_random.bit_generator.state = gym_rng
        for name, (present, value) in hooks.items():
            if present:
                setattr(base, name, value)
            else:
                vars(base).pop(name, None)
        if native_context(env) != before_original:
            raise CandidateRejected('RESTORATION_FAILURE', 'Original environment recovery failed.')


def label_candidate(env, planner=None, *, timings=None, deadline=None, counts=None):
    assert_canonical(env)
    state = read_state(env)
    valid = tuple(np.flatnonzero(geometric_mask(state)).tolist())
    if not valid:
        raise CandidateRejected('NO_VALID_ACTIONS', 'Canonical state has no geometric-valid actions.')
    planner = planner if planner is not None else TacticalPlanner()
    timings = timings if timings is not None else []
    reachable = []
    for action_id in valid:
        # Boundary watchdog: never interrupt/approximate a Planner search.
        if deadline is not None and perf_counter() >= deadline:
            raise CandidateRejected('TIME_BUDGET', 'Build wall-time limit reached between Planner calls.', action_id)
        try:
            plan = checked_plan(env, SPACE.decode(action_id), planner, timings)
        except CandidateRejected as exc:
            exc.action_id = action_id
            raise
        if plan.status == PlannerStatus.REACHABLE:
            reachable.append(action_id)
        if counts is not None:
            key = 'reachable' if plan.status == PlannerStatus.REACHABLE else 'unreachable'
            counts[key] += 1
    if not reachable:
        raise CandidateRejected('ZERO_REACHABLE', 'Invariant violation: no valid action is reachable.')
    return EvaluationRecord(state, valid, tuple(reachable))


def timing_statistics(times):
    ordered = sorted(times)
    return dict(call_count=len(times), total_seconds=sum(times),
                average_seconds=sum(times) / len(times) if times else None,
                p95_seconds=ordered[ceil(.95 * len(times)) - 1] if times else None,
                max_seconds=max(times) if times else None)


def build_dataset(config: BuildConfig, *, planner=None, executor=None, on_progress=None,
                  excluded_hashes=()):
    """Return complete records and diagnostics. Stop at the first system failure.

    Natural termination resets with (seed + episode) modulo 2**32. Collection
    draws uniformly from labeled reachable IDs using an owned random.Random.
    All accepted records are labeled exhaustively, even the final candidate.
    excluded_hashes are previously accepted global states: visit/advance them,
    but keep collecting until this seed fills its own new-state quota.
    """
    planner = planner if planner is not None else TacticalPlanner()
    executor = executor if executor is not None else Executor()
    root = Path(__file__).resolve().parents[3]
    # Source exports are valid projects without live Git metadata. Do not
    # discover an ancestor repository or invent a launch commit for an export.
    commit = (subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
              if (root / '.git').exists() else None)
    planner_source = root / 'src/tetris_agent/planner.py'
    records, diagnostics, labeling_times, advancement_times = [], [], [], []
    label_counts = dict(reachable=0, unreachable=0)
    seen = set(excluded_hashes)
    local_records = {}
    rng = Random(config.seed)
    env = Tetris(gravity=True)
    started = perf_counter()
    deadline = started + config.max_seconds
    visited = duplicates = episode = placements = 0
    status = 'completed'
    pending = None
    try:
        env.reset(seed=config.seed)
        while len(records) < config.state_count:
            if perf_counter() >= deadline or visited >= config.max_candidates:
                status = 'incomplete'
                diagnostics.append(dict(reason='TIME_BUDGET' if perf_counter() >= deadline else 'CANDIDATE_LIMIT'))
                break
            assert_canonical(env)
            state = read_state(env)
            key = state_hash(state)
            visited += 1
            pending = dict(state_hash=key, state=serialize_state(state), episode=episode,
                           episode_seed=(config.seed + episode) % 2**32,
                           candidate_index=visited, native_context=native_context(env))
            record = local_records.get(key)
            if record is None:
                record = label_candidate(env, planner, timings=labeling_times, deadline=deadline,
                                         counts=label_counts)
                record = EvaluationRecord(record.state, record.valid_action_ids, record.reachable_action_ids,
                                          dict(episode=episode, candidate_index=visited,
                                               episode_seed=(config.seed + episode) % 2**32))
                local_records[key] = record
            if key in seen:
                duplicates += 1
            else:
                records.append(record)
                seen.add(key)
            if on_progress:
                on_progress(dict(candidate_states_visited=visited, accepted_unique_states=len(records),
                                 wall_seconds=perf_counter() - started,
                                 labeling=timing_statistics(labeling_times)))
            if len(records) == config.state_count:
                break
            if perf_counter() >= deadline:
                status = 'incomplete'
                diagnostics.append(dict(reason='TIME_BUDGET', **pending))
                break
            action = rng.choice(record.reachable_action_ids)
            target = SPACE.decode(action)
            # Re-plan and verify from the same live state before Executor dispatch.
            plan = checked_plan(env, target, planner, advancement_times)
            if plan.status != PlannerStatus.REACHABLE:
                raise CandidateRejected('LABEL_DISAGREEMENT', 'Selected label changed on replanning.', action)
            result = executor.execute(env, target, plan)
            if not result.success or not result.target_verified:
                raise CandidateRejected('EXECUTION_ERROR', f'{result.failure_reason}: {result.error_message}', action)
            placements += 1
            if result.terminated or result.truncated:
                episode += 1
                env.reset(seed=(config.seed + episode) % 2**32)
            else:
                assert_canonical(env)
            pending = None
    except CandidateRejected as exc:
        status = 'incomplete' if exc.reason == 'TIME_BUDGET' else 'diagnostic_failure'
        diagnostics.append(dict(reason=exc.reason, detail=str(exc), action_id=exc.action_id, **(pending or {})))
    except Exception as exc:
        status = 'diagnostic_failure'
        diagnostics.append(dict(reason='SYSTEM_ERROR', detail=f'{type(exc).__name__}: {exc}', **(pending or {})))
    finally:
        env.close()
    rejected_candidates = visited - len(records) - duplicates
    reasons = {}
    for diagnostic in diagnostics:
        reason = diagnostic['reason']
        reasons[reason] = reasons.get(reason, 0) + 1
    metadata = dict(schema_version=1, git_commit=commit,
        tetris_gymnasium_version=version('tetris-gymnasium'), generation_seed=config.seed,
        requested_state_count=config.state_count, accepted_state_count=len(records),
        rejected_state_count=visited - len(records),
        generation_configuration=dict(asdict(config), gravity=True,
            collection='uniform_labeled_reachable_via_planner_executor',
            episode_seed_rule='(seed + zero_based_episode) modulo 2**32',
            watchdog='checked between complete Planner calls; no per-search timeout'),
        planner_implementation=dict(git_commit=commit, source_sha256=sha256(planner_source.read_bytes()).hexdigest()),
        status=status, aggregate_label_statistics=label_statistics(records),
        build_statistics=dict(wall_seconds=perf_counter() - started, candidate_states_visited=visited,
            accepted_unique_states=len(records), rejected_duplicates=duplicates,
            rejected_candidates=rejected_candidates, rejection_reasons=reasons,
            total_valid_actions_labeled=sum(label_counts.values()),
            reachable_actions_labeled=label_counts['reachable'],
            unreachable_actions_labeled=label_counts['unreachable'],
            zero_reachable_states=reasons.get('ZERO_REACHABLE', 0),
            verified_trajectory_placements=placements, episodes_started=episode + 1,
            planner_labeling=timing_statistics(labeling_times),
            planner_advancement=timing_statistics(advancement_times)), diagnostics=diagnostics)
    return tuple(records), metadata


def build_multiseed_dataset(seeds, *, states_per_seed=64, max_candidates=320,
                            max_seconds=300., on_progress=None):
    """Thin ordered orchestration; watchdogs apply independently to each seed.

    Stop at the first unmet quota/error. No seed borrows another seed's quota.
    The same single-seed collector handles duplicate advancement and labeling.
    """
    seeds = tuple(seeds)
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError('Expected a nonempty list of distinct generation seeds.')
    configs = [BuildConfig(seed, states_per_seed, max_candidates, max_seconds) for seed in seeds]
    started = perf_counter()
    records, per_seed = [], []
    seen = set()
    for config in configs:
        def progress(row):
            if on_progress:
                on_progress(dict(row, generation_seed=config.seed,
                                 total_accepted_states=len(records) + row['accepted_unique_states']))
        contribution, meta = build_dataset(config, excluded_hashes=seen, on_progress=progress)
        for record in contribution:
            records.append(EvaluationRecord(record.state, record.valid_action_ids,
                record.reachable_action_ids, dict(record.metadata, generation_seed=config.seed)))
            seen.add(record.state_hash)
        per_seed.append(meta)
        if meta['status'] != 'completed' or len(contribution) != states_per_seed:
            break
    total_requested = len(seeds) * states_per_seed
    status = ('completed' if len(records) == total_requested
              and all(m['status'] == 'completed' for m in per_seed) else
              'diagnostic_failure' if any(m['status'] == 'diagnostic_failure' for m in per_seed)
              else 'incomplete')
    # Native per-call p95 remains per seed; never average percentiles.
    def combined_timing(name):
        timings = [m['build_statistics'][name] for m in per_seed]
        count = sum(t['call_count'] for t in timings)
        total = sum(t['total_seconds'] for t in timings)
        return dict(call_count=count, total_seconds=total,
                    average_seconds=total / count if count else None,
                    max_seconds=max((t['max_seconds'] for t in timings if t['call_count']), default=None))
    statistics = {key: sum(m['build_statistics'][key] for m in per_seed) for key in (
        'candidate_states_visited', 'accepted_unique_states', 'rejected_duplicates',
        'rejected_candidates', 'total_valid_actions_labeled', 'reachable_actions_labeled',
        'unreachable_actions_labeled', 'zero_reachable_states',
        'verified_trajectory_placements', 'episodes_started')}
    statistics.update(wall_seconds=perf_counter() - started,
                      planner_labeling=combined_timing('planner_labeling'),
                      planner_advancement=combined_timing('planner_advancement'))
    first = per_seed[0]
    metadata = dict(schema_version=first['schema_version'], git_commit=first['git_commit'],
        tetris_gymnasium_version=first['tetris_gymnasium_version'],
        planner_implementation=first['planner_implementation'],
        generation_seed=None, generation_seeds=list(seeds), states_per_seed_requested=states_per_seed,
        total_requested_states=total_requested, requested_state_count=total_requested,
        total_accepted_states=len(records), accepted_state_count=len(records),
        rejected_state_count=sum(m['rejected_state_count'] for m in per_seed),
        global_duplicate_count=statistics['rejected_duplicates'],
        generation_configuration=dict(mode='multi_seed', seeds=list(seeds),
            states_per_seed=states_per_seed, max_candidates_per_seed=max_candidates,
            max_seconds_per_seed=max_seconds, seed_order='as_supplied', global_deduplication=True),
        status=status, per_seed=per_seed, build_statistics=statistics,
        aggregate_label_statistics=label_statistics(records),
        diagnostics=[dict(d, generation_seed=m['generation_seed'])
                     for m in per_seed for d in m['diagnostics']],
        dataset_fingerprint=dataset_fingerprint(records))
    return tuple(records), metadata
