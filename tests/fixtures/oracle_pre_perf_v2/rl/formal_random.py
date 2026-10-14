"""Episode gameplay baseline using the frozen RL decision and reward contracts."""
from dataclasses import asdict, dataclass
from importlib.metadata import version
import json
from math import ceil
import os
from pathlib import Path
import platform
from time import perf_counter

import numpy as np

from ..planner import TacticalPlanner
from .artifacts import RunArtifacts, utc_now
from .metrics import EpisodeMetrics
from .policy import FormalRandomPolicy
from .runner import RLRunner


@dataclass(frozen=True)
class FormalRandomConfig:
    episodes: int
    env_seed: int
    policy_seed: int
    max_decisions_per_episode: int = 10000
    reward_mode: str = 'minimal'
    log_every: int = 100

    def __post_init__(self):
        for name in ('episodes', 'max_decisions_per_episode', 'log_every'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f'{name} must be a positive integer.')
        for name in ('env_seed', 'policy_seed'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < 2**32:
                raise ValueError(f'{name} must be an integer in [0, 2**32 - 1].')
        if self.reward_mode not in ('minimal', 'shaped'):
            raise ValueError('Unknown reward mode.')


class _MeasuredPlanner:
    """Observe post-selection calls only; leave plans and physics untouched."""
    def __init__(self, planner):
        self.planner = planner
        self.times = []

    def plan(self, env, target):
        started = perf_counter()
        try:
            plan = self.planner.plan(env, target)
        except BaseException:
            self.times.append(perf_counter() - started)
            raise
        self.times.append(plan.elapsed_seconds)
        return plan


def _timings(times, total, count):
    ordered = sorted(times)
    return dict(planner_call_count=count, planner_elapsed_seconds=total,
                planner_average_seconds=total / count if count else None,
                planner_p95_seconds=ordered[ceil(.95 * len(ordered)) - 1] if ordered else None,
                planner_max_seconds=max(ordered) if ordered else None)


def _episode_row(metrics, index, seed, config, wall, times):
    row = asdict(metrics)
    # RLRunner's accepted-transition counter is useful without storing transitions.
    row['completed_decisions'] = row.pop('stored_transitions')
    row.update(episode_index=index, env_seed=seed, policy_seed=config.policy_seed,
               policy_rng='run_owned_random_continuous',
               reachable_decisions=metrics.pieces_placed,
               unreachable_rate=metrics.unreachable_rate,
               decisions_per_placement=metrics.decisions_per_placement,
               wall_seconds=wall,
               decisions_per_second=metrics.decision_count / wall if wall > 0 else 0.,
               placements_per_second=metrics.pieces_placed / wall if wall > 0 else 0.)
    row.update(_timings(times, metrics.planner_elapsed_seconds, metrics.planner_call_count))
    return row


def _summary(rows, config, wall, times, status, diagnostic):
    decisions = sum(row['decision_count'] for row in rows)
    placements = sum(row['pieces_placed'] for row in rows)
    unreachable = sum(row['unreachable_count'] for row in rows)
    count = len(rows)
    result = dict(
        schema_version=1, kind='formal_random_episode_baseline', status=status,
        diagnostic=diagnostic, requested_episodes=config.episodes,
        episode_count=count, attempted_episode_count=count,
        completed_episode_count=sum(row['diagnostic'] is None and
                                    (row['terminated'] or row['truncated']) for row in rows),
        decision_count=decisions, completed_decisions=sum(row['completed_decisions'] for row in rows),
        pieces_placed=placements, reachable_decisions=placements,
        lines_cleared=sum(row['lines_cleared'] for row in rows), unreachable_count=unreachable,
        unreachable_rate=unreachable / decisions if decisions else 0.,
        episode_return=sum(row['episode_return'] for row in rows),
        mean_episode_return=sum(row['episode_return'] for row in rows) / count if count else None,
        mean_lines=sum(row['lines_cleared'] for row in rows) / count if count else None,
        mean_pieces=placements / count if count else None,
        terminated_count=sum(row['terminated'] for row in rows),
        truncated_count=sum(row['truncated'] for row in rows),
        system_failure_count=sum(row['diagnostic'] is not None and
                                 row['stop_reason'] != 'INTERRUPTED' for row in rows),
        self_loop_count=sum(row['self_loop_count'] for row in rows),
        mean_max_consecutive_unreachable=(sum(row['max_consecutive_unreachable'] for row in rows)
                                          / count if count else None),
        max_consecutive_unreachable=max((row['max_consecutive_unreachable'] for row in rows), default=0),
        decisions_per_placement=decisions / placements if placements else None,
        reward_totals={key: sum(row['reward_totals'][key] for row in rows)
                       for key in EpisodeMetrics().reward_totals},
        wall_seconds=wall, decisions_per_second=decisions / wall if wall > 0 else 0.,
        placements_per_second=placements / wall if wall > 0 else 0.,
        gradient_updates=0, replay_size=0,
    )
    result.update(_timings(times, sum(row['planner_elapsed_seconds'] for row in rows),
                           sum(row['planner_call_count'] for row in rows)))
    return result


class _Artifacts:
    # Reuse only the existing fsync/atomic JSON/status primitives, not the trainer
    # constructor, periodic checkpoints, or checkpoint finalization.
    _atomic = RunArtifacts._atomic
    _json = RunArtifacts._json
    _status = RunArtifacts._status

    def __init__(self, path, config, env):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=False)
        self.started_at = utc_now()
        self._stream = None
        try:
            self._json('config.json', dict(
                schema_version=1, kind='formal_random_episode_baseline', learning=False,
                config=asdict(config), output_dir=str(self.path.resolve()), started_at=self.started_at,
                seed_contract=dict(environment='(env_seed + zero_based_episode) % 2**32',
                                   policy='one owned random.Random(policy_seed), continuous across episodes',
                                   global_rng_reseeded=False),
                environment=dict(type=f'{type(env).__module__}.{type(env).__qualname__}',
                                 note='CLI constructs native Tetris(gravity=True); API callers own environment setup'),
                versions=dict(python=platform.python_version(), numpy=np.__version__,
                              tetris_gymnasium=version('tetris-gymnasium')),
            ))
            self._status('running', None)
            self._stream = (self.path / 'metrics.jsonl').open('x', encoding='utf-8')
            (self.path / 'episodes.jsonl').touch(exist_ok=False)
        except BaseException as exc:
            self.error(exc)
            self.close()
            raise

    def error(self, exc):
        try:
            self._status('artifact_error', f'{type(exc).__name__}: {exc}')
        except BaseException:
            pass  # An unavailable disk can also prevent the error marker.

    def log(self, row, event):
        RunArtifacts.log(self, row, event)

    def episode(self, row):
        with (self.path / 'episodes.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(dict(row, timestamp=utc_now()), allow_nan=False) + '\n')
            stream.flush()
            os.fsync(stream.fileno())

    def close(self):
        if self._stream is not None:
            self._stream.close()


def run_formal_random(env, config: FormalRandomConfig, *, output_dir=None,
                      planner=None, executor=None):
    """Run independent episodes with one owned uniform policy RNG, without learning.

    Diagnostic decisions are attempted decisions, but earn no physical-unreachable
    penalty. Stop the entire run on the first diagnostic. No retry/resume protocol.
    Output is optional for embedded callers; the production CLI always supplies it.
    """
    measured = _MeasuredPlanner(planner if planner is not None else TacticalPlanner())
    runner = RLRunner(FormalRandomPolicy(seed=config.policy_seed), reward_mode=config.reward_mode,
                      planner=measured, executor=executor)
    artifacts = _Artifacts(output_dir, config, env) if output_dir is not None else None
    started = perf_counter()
    rows = []
    current = None
    status, diagnostic = 'running', None
    last_log = 0
    previous_decisions = 0
    writing = False

    def snapshot():
        active = [] if current is None else [_episode_row(
            current, index, seed, config, perf_counter() - episode_started,
            measured.times[time_offset:])]
        return _summary(rows + active, config, perf_counter() - started, measured.times,
                        status, diagnostic)

    def persist(function, *args):
        nonlocal writing
        writing = True
        function(*args)
        writing = False

    def progress(metrics):
        nonlocal current, last_log
        current = metrics
        count = previous_decisions + metrics.decision_count
        if artifacts is not None and count - last_log >= config.log_every:
            row = snapshot()
            persist(artifacts.log, row, 'periodic')
            last_log = count

    try:
        if artifacts is not None:
            persist(artifacts.log, snapshot(), 'start')
        for index in range(1, config.episodes + 1):
            seed = (config.env_seed + index - 1) % 2**32
            episode_started, time_offset = perf_counter(), len(measured.times)
            current = EpisodeMetrics()
            metrics = runner.run(env, max_decisions=config.max_decisions_per_episode,
                                 env_seed=seed, on_progress=progress)
            rows.append(_episode_row(metrics, index, seed, config,
                                     perf_counter() - episode_started, measured.times[time_offset:]))
            current = None
            previous_decisions += metrics.decision_count
            if artifacts is not None:
                persist(artifacts.episode, rows[-1])  # Durable before the next reset.
                persist(artifacts.log, snapshot(), 'episode_end')
                last_log = previous_decisions
            if metrics.diagnostic is not None:
                status, diagnostic = 'system_error', metrics.diagnostic
                break
        if status == 'running':
            status = 'completed'
    except BaseException as exc:
        diagnostic = f'{type(exc).__name__}: {exc}'
        status = 'artifact_error' if writing else ('interrupted' if isinstance(exc, KeyboardInterrupt)
                                                 else 'system_error')
        if current is not None:
            current.fail('INTERRUPTED' if status == 'interrupted' else status.upper(), diagnostic)
            rows.append(_episode_row(current, index, seed, config,
                                     perf_counter() - episode_started, measured.times[time_offset:]))
            current = None
            if artifacts is not None and not writing:
                try:
                    artifacts.episode(rows[-1])
                except BaseException as persistence_error:
                    status = 'artifact_error'
                    diagnostic = f'{type(persistence_error).__name__}: {persistence_error}'
        raise
    finally:
        if artifacts is not None:
            try:
                artifacts.log(snapshot(), 'final')
                final = snapshot()  # Include the final JSONL flush in throughput.
                artifacts._json('summary.json', final)
                artifacts._status(status, diagnostic)
            except BaseException as exc:
                artifacts.error(exc)
                raise
            finally:
                artifacts.close()
    return final if artifacts is not None else snapshot()
