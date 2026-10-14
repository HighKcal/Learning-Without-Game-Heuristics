"""Tiny infrastructure verification only; never starts a production experiment.

Run: .\\.venv\\Scripts\\python.exe scripts/smoke_rl.py
Uses the existing PyTorch installation. CPU real-environment run: three
decisions/mode and two updates/mode. CUDA, when available: one synthetic
terminal update. Test-only small warmup/batch overrides are reported below.
"""
from dataclasses import asdict
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

import numpy as np
import torch
from tetris_gymnasium.envs.tetris import Tetris

from tetris_agent.rl.config import TrainingConfig
from tetris_agent.rl.dqn import DQNTrainer
from tetris_agent.rl.policy import FormalRandomPolicy
from tetris_agent.rl.replay import Transition
from tetris_agent.rl.runner import RLRunner
from tetris_agent.rl.state import read_state, geometric_mask
from tetris_agent.rl.training import train


def main():
    env = Tetris(gravity=True)
    report = {'torch': torch.__version__, 'smoke_overrides': {'batch_size': 2, 'learning_starts': 2}}
    try:
        report['formal_random'] = asdict(RLRunner(FormalRandomPolicy(42)).run(
            env, env_seed=42, max_decisions=3))
        for algorithm in ('vanilla', 'double'):
            trainer = DQNTrainer(TrainingConfig(decision_budget=3, algorithm=algorithm,
                                               device='cpu', batch_size=2, learning_starts=2))
            summary = train(env, trainer, episode_decision_limit=3)
            assert summary.stored_experiences == 3 and summary.gradient_updates == 2
            assert all(e.diagnostic is None for e in summary.episodes)
            report[algorithm] = {'summary': asdict(summary), 'last_update': asdict(trainer.last_metrics)}
        report['cuda_available'] = torch.cuda.is_available()
        if torch.cuda.is_available():
            env.reset(seed=42)
            state = read_state(env)
            action = int(np.flatnonzero(geometric_mask(state))[0])
            trainer = DQNTrainer(TrainingConfig(decision_budget=1, device='cuda',
                                               batch_size=1, learning_starts=1))
            before = trainer.online[-1].weight.detach().clone()
            metrics = trainer.observe(Transition(state, action, 1., state, True, False))
            torch.cuda.synchronize()
            assert trainer.gradient_updates == 1 and np.isfinite(metrics.loss)
            assert not torch.equal(before, trainer.online[-1].weight)
            report['cuda'] = {'device': torch.cuda.get_device_name(), 'update': asdict(metrics)}
        print(json.dumps(report, indent=2))
    finally:
        env.close()


if __name__ == '__main__':
    main()
