"""Periodic durable artifacts. Checkpoints are snapshots, NOT resumable runs."""
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform

import numpy as np
import torch
from .oracle import oracle_metadata


SNAPSHOT_CONTRACT = {
    'kind': 'training_snapshot',
    'exact_resume_supported': False,
    'missing_for_exact_resume': [
        'replay contents and ring position', 'replay sampling RNG', 'policy RNG',
        'Python RNG', 'NumPy RNG', 'Torch CPU RNG', 'Torch CUDA RNGs',
        'environment and wrapper state, randomizer/queue and RNGs',
        'episode orchestration state and tested restoration protocol',
    ],
    'note': 'For weights/optimizer inspection only; no resume API. train() starts fresh.',
}


def utc_now():
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class ArtifactConfig:
    output_dir: str | Path
    log_every: int = 100
    checkpoint_every: int = 5000

    def __post_init__(self):
        for name in ('log_every', 'checkpoint_every'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f'{name} must be a positive integer.')
        if not str(self.output_dir).strip():
            raise ValueError('output_dir must not be empty.')


def epsilon_state(trainer):
    config = trainer.config
    return dict(counter='stored_experiences', start=1.0, final=0.05,
                warmup=config.learning_starts, post_warmup_budget_fraction=0.4,
                decay_stored_experiences=config.epsilon_decay_decisions,
                stored_experiences=trainer.stored_experiences,
                progress=max(0, trainer.stored_experiences - config.learning_starts),
                next_selection_epsilon=trainer.epsilon)


class RunArtifacts:
    def __init__(self, options, trainer, episode_decision_limit):
        self.options = options
        self.path = Path(options.output_dir)
        # Never append to or overwrite another run, including an empty old directory.
        self.path.mkdir(parents=True, exist_ok=False)
        self.started_at = utc_now()
        self._stream = None
        self._last_log = self._last_checkpoint = 0
        self._json('config.json', dict(
            schema_version=1, training=asdict(trainer.config),
            **oracle_metadata(trainer.config.reachability_mask_mode),
            artifacts={**asdict(options), 'output_dir': str(self.path.resolve())},
            episode_decision_limit=episode_decision_limit,
            epsilon=epsilon_state(trainer), checkpoint_contract=SNAPSHOT_CONTRACT,
            started_at=self.started_at, resolved_device=str(trainer.device),
            device_name=(torch.cuda.get_device_name(trainer.device)
                         if trainer.device.type == 'cuda' else platform.processor()),
            versions=dict(python=platform.python_version(), torch=str(torch.__version__),
                          numpy=np.__version__, tetris_gymnasium=version('tetris-gymnasium')),
            determinism=dict(torch_deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
                             cudnn_deterministic=torch.backends.cudnn.deterministic,
                             cudnn_benchmark=torch.backends.cudnn.benchmark),
        ))
        self._status('running', None)
        self._stream = (self.path / 'metrics.jsonl').open('x', encoding='utf-8')

    def _atomic(self, name, writer):
        destination = self.path / name
        temporary = self.path / ('.' + name + '.tmp')
        try:
            with temporary.open('wb') as stream:
                writer(stream)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)

    def _json(self, name, payload):
        data = (json.dumps(payload, indent=2, allow_nan=False) + '\n').encode('utf-8')
        self._atomic(name, lambda stream: stream.write(data))

    def _status(self, status, diagnostic):
        self._json('status.json', dict(status=status, diagnostic=diagnostic,
                                      started_at=self.started_at,
                                      ended_at=None if status == 'running' else utc_now()))

    def log(self, row, event):
        self._stream.write(json.dumps(dict(row, event=event, timestamp=utc_now()), allow_nan=False) + '\n')
        self._stream.flush()
        os.fsync(self._stream.fileno())
        self._last_log = row['decision_count']

    def log_episode(self, episode_index, episode):
        row = dict(asdict(episode), episode_index=episode_index,
                   unreachable_rate=episode.unreachable_rate, timestamp=utc_now())
        with (self.path / 'episodes.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(row, allow_nan=False) + '\n')
            stream.flush()
            os.fsync(stream.fileno())

    def checkpoint(self, trainer, row, name, status):
        payload = dict(
            schema_version=1, checkpoint_contract=SNAPSHOT_CONTRACT,
            saved_at=utc_now(), started_at=self.started_at, status=status,
            config=asdict(trainer.config), epsilon=epsilon_state(trainer),
            **oracle_metadata(trainer.config.reachability_mask_mode),
            online_state_dict=trainer.online.state_dict(),
            target_state_dict=trainer.target.state_dict(),
            optimizer_state_dict=trainer.optimizer.state_dict(),
            gradient_updates=trainer.gradient_updates,
            stored_experiences=trainer.stored_experiences,
            decision_count=row['decision_count'], target_sync_count=trainer.target_sync_count,
            episode=row['episode'], metrics=row,
        )
        self._atomic(name, lambda stream: torch.save(payload, stream))
        self._last_checkpoint = row['decision_count']

    def periodic(self, trainer, row):
        count = row['decision_count']
        if count - self._last_log >= self.options.log_every:
            self.log(row, 'periodic')
        if count - self._last_checkpoint >= self.options.checkpoint_every:
            self.checkpoint(trainer, row, f'checkpoint_{count:09d}.pt', 'running')

    def finish(self, trainer, row, status, diagnostic=None, *, refresh=None):
        try:
            self.log(row, 'final')
            self.checkpoint(trainer, row, 'checkpoint_final.pt', status)
            if refresh is not None:
                row = refresh()  # Include final log/checkpoint I/O in summary throughput.
            self._json('summary.json', dict(row, status=status, diagnostic=diagnostic))
            self._status(status, diagnostic)
            return row
        except BaseException as exc:
            # Never leave a success marker if final persistence failed.
            try:
                self._status('artifact_error', f'{type(exc).__name__}: {exc}')
            except BaseException:
                pass  # A full/unavailable disk may also prevent the status write.
            raise
        finally:
            if self._stream is not None:
                self._stream.close()
