"""Production RL entrypoint; tiny smoke and long runs use exactly the same train()."""
import argparse
import json
from math import isfinite
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from tetris_gymnasium.envs.tetris import Tetris

from tetris_agent.rl.artifacts import ArtifactConfig
from tetris_agent.rl.config import TrainingConfig
from tetris_agent.rl.dqn import DQNTrainer
from tetris_agent.rl.training import train


def positive_int(value):
    try:
        number = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError('must be a positive integer') from None
    if number <= 0:
        raise argparse.ArgumentTypeError('must be a positive integer')
    return number


def seed_int(value):
    try:
        number = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError('seed must be an integer in [0, 2**32 - 1]') from None
    if not 0 <= number < 2**32:
        raise argparse.ArgumentTypeError('seed must be in [0, 2**32 - 1]')
    return number


def finite_negative_float(value):
    try:
        number = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError('must be a finite negative number') from None
    if not isfinite(number) or number >= 0:
        raise argparse.ArgumentTypeError('must be a finite negative number')
    return number


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--algorithm', choices=('vanilla', 'double'), required=True)
    parser.add_argument('--reward', choices=('minimal', 'shaped'), required=True)
    parser.add_argument('--decision-budget', type=positive_int, required=True)
    parser.add_argument('--seed', type=seed_int, required=True)
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    parser.add_argument('--reachability-mask', choices=('geometric', 'oracle'), default='geometric',
                        help='Explicit clean physical oracle condition; default is geometric-only.')
    parser.add_argument('--unreachable-penalty', type=finite_negative_float, default=-0.05)
    parser.add_argument('--epsilon-decay-decisions', type=positive_int, default=None,
                        help='Override the number of decisions over which epsilon decays to its final value.')
    parser.add_argument('--output-dir', type=Path, required=True,
                        help='New run directory; an existing directory is rejected.')
    parser.add_argument('--log-every', type=positive_int, default=100)
    parser.add_argument('--checkpoint-every', type=positive_int, default=5000)
    parser.add_argument('--max-decisions-per-episode', type=positive_int, default=10000,
                        help='Decision-boundary watchdog; truncations still bootstrap.')
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        parser.error('--output-dir must be a new directory for each run')
    if args.reachability_mask == 'oracle' and (args.algorithm != 'double' or args.reward != 'minimal'):
        parser.error('Clean oracle condition requires --algorithm double --reward minimal')
    config = TrainingConfig(decision_budget=args.decision_budget, algorithm=args.algorithm,
                            reward_mode=args.reward, seed=args.seed, device=args.device,
                            unreachable_penalty=args.unreachable_penalty,
                            epsilon_decay_override=args.epsilon_decay_decisions,
                            reachability_mask_mode=args.reachability_mask)
    if args.device == 'cuda':
        import torch
        if not torch.cuda.is_available():
            parser.error('--device cuda requested but CUDA is unavailable')
    from tetris_agent.native_preflight import native_preflight
    native_preflight()
    trainer = DQNTrainer(config)
    env = Tetris(gravity=True)
    try:
        summary = train(env, trainer, episode_decision_limit=args.max_decisions_per_episode,
                        artifacts=ArtifactConfig(args.output_dir, args.log_every, args.checkpoint_every))
        print(json.dumps(dict(
            status=summary.status, episode_count=len(summary.episodes),
            decision_count=summary.decision_count, stored_experiences=summary.stored_experiences,
            gradient_updates=summary.gradient_updates, target_sync_count=summary.target_sync_count,
            throughput=summary.throughput, output_dir=str(args.output_dir),
        ), indent=2))
        return 0 if summary.status == 'completed' else 1
    finally:
        env.close()


if __name__ == '__main__':
    raise SystemExit(main())
