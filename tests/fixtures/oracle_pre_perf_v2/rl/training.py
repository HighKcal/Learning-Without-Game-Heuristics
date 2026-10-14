"""Explicit-budget orchestration shared by both DQN modes; no implicit run."""
from dataclasses import asdict, dataclass, field
from time import perf_counter
from typing import Callable
from .artifacts import ArtifactConfig, RunArtifacts
from .dqn import DQNTrainer, LearningMetrics
from .metrics import EpisodeMetrics
from .runner import RLRunner
from .oracle import oracle_metadata


@dataclass
class TrainingSummary:
    episodes: list[EpisodeMetrics]
    decision_count: int
    stored_experiences: int
    gradient_updates: int
    target_sync_count: int
    status: str = 'completed'
    throughput: dict[str, float | int] = field(default_factory=dict)


def train(env, trainer: DQNTrainer, *, episode_decision_limit: int,
          on_learning: Callable[[LearningMetrics], object] | None = None,
          artifacts: ArtifactConfig | None = None) -> TrainingSummary:
    """Fresh experiment, config.decision_budget selections, <=1 update/decision.

    A system error stops the experiment instead of silently resetting/retrying.
    Final budget exhaustion truncates at the last real decision next-state.
    Environment seeds advance by episode, policy/replay RNGs persist.
    """
    if trainer.stored_experiences or trainer.gradient_updates or len(trainer.replay):
        raise ValueError('train() starts a fresh experiment; use RLRunner for manual continuation.')
    if isinstance(episode_decision_limit, bool) or not isinstance(episode_decision_limit, int) or episode_decision_limit <= 0:
        raise ValueError('episode_decision_limit must be a positive integer.')
    runner = RLRunner(trainer, reward_mode=trainer.config.reward_mode,
                      unreachable_penalty=trainer.config.unreachable_penalty,
                      reachability_mask_mode=trainer.config.reachability_mask_mode)
    episodes = []
    decisions = 0
    current = None
    started = perf_counter()
    writer = RunArtifacts(artifacts, trainer, episode_decision_limit) if artifacts else None
    total_fields = ('episode_return', 'lines_cleared', 'pieces_placed', 'unreachable_count',
                    'self_loop_count', 'planner_elapsed_seconds', 'planner_call_count',
                    'oracle_computation_count', 'oracle_elapsed_seconds', 'oracle_visited_states',
                    'geometric_valid_action_count', 'reachable_action_count',
                    'masked_physical_u_count', 'selected_physical_u_count')
    totals = dict.fromkeys(total_fields, 0)

    def snapshot():
        combined = {key: value + (getattr(current, key) if current else 0)
                    for key, value in totals.items()}
        count = decisions + (current.decision_count if current else 0)
        wall = perf_counter() - started
        planner_calls = combined['planner_call_count']
        learner = trainer.learner_elapsed_seconds
        return dict(
            **oracle_metadata(trainer.config.reachability_mask_mode),
            decision_count=count, stored_experiences=trainer.stored_experiences,
            gradient_updates=trainer.gradient_updates, target_sync_count=trainer.target_sync_count,
            episode=len(episodes) + int(current is not None),
            episode_metrics=asdict(current) if current else (asdict(episodes[-1]) if episodes else None),
            **combined,
            reachable_fraction=(combined['reachable_action_count'] / combined['geometric_valid_action_count']
                                if combined['geometric_valid_action_count'] else None),
            oracle_average_seconds=(combined['oracle_elapsed_seconds'] / combined['oracle_computation_count']
                                    if combined['oracle_computation_count'] else 0.),
            unreachable_rate=combined['unreachable_count'] / count if count else 0.,
            self_loop_ratio=combined['self_loop_count'] / trainer.stored_experiences
                            if trainer.stored_experiences else 0.,
            epsilon=trainer.epsilon, replay_size=len(trainer.replay),
            last_learning=asdict(trainer.last_metrics) if trainer.last_metrics else None,
            wall_clock_seconds=wall,
            decisions_per_second=count / wall if wall else 0.,
            placements_per_second=combined['pieces_placed'] / wall if wall else 0.,
            gradient_updates_per_second=trainer.gradient_updates / wall if wall else 0.,
            planner_average_seconds=combined['planner_elapsed_seconds'] / planner_calls if planner_calls else 0.,
            learner_elapsed_seconds=learner,
            learner_average_seconds=learner / trainer.gradient_updates if trainer.gradient_updates else 0.,
        )

    def progress(metrics):
        nonlocal current
        current = metrics
        count = decisions + metrics.decision_count
        # Serialize only at intervals; the hook itself does no policy/learner work.
        if writer and metrics.decision_count and (
                count % artifacts.log_every == 0 or count % artifacts.checkpoint_every == 0):
            writer.periodic(trainer, snapshot())

    def observe(transition):
        metrics = trainer.observe(transition)
        if metrics is not None and on_learning is not None:
            on_learning(metrics)

    status = 'completed'
    diagnostic = None
    try:
        while decisions < trainer.config.decision_budget:
            episode = runner.run(env, env_seed=trainer.config.seed + len(episodes),
                                 max_decisions=min(episode_decision_limit, trainer.config.decision_budget - decisions),
                                 on_transition=observe, on_progress=progress)
            if episode.decision_count == 0 and episode.diagnostic is None:
                episode.fail('NO_PROGRESS', 'Episode ended without a decision; refusing an endless reset loop.')
            episodes.append(episode)
            decisions += episode.decision_count
            for key in totals:
                totals[key] += getattr(episode, key)
            current = None
            if writer:
                writer.log_episode(len(episodes), episode)
            if episode.diagnostic is not None:
                status, diagnostic = 'system_error', episode.diagnostic
                break
    except BaseException as exc:
        if writer:
            try:
                writer.finish(trainer, snapshot(),
                              'interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed',
                              f'{type(exc).__name__}: {exc}')
            except BaseException as persistence_error:
                exc.add_note(f'Artifact finalization also failed: {persistence_error}')
        raise
    row = snapshot()
    if writer:
        row = writer.finish(trainer, row, status, diagnostic, refresh=snapshot)
    throughput_keys = ('wall_clock_seconds', 'decisions_per_second', 'placements_per_second',
                       'gradient_updates_per_second', 'planner_elapsed_seconds', 'planner_call_count',
                       'planner_average_seconds', 'learner_elapsed_seconds', 'learner_average_seconds')
    return TrainingSummary(episodes, decisions, trainer.stored_experiences,
                           trainer.gradient_updates, trainer.target_sync_count, status,
                           {key: row[key] for key in throughput_keys})
