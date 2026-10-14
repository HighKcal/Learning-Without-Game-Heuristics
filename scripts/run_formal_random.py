"""Production Formal Random episode baseline; no learner or checkpoints."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from tetris_gymnasium.envs.tetris import Tetris
from tetris_agent.rl.formal_random import FormalRandomConfig, run_formal_random


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


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episodes', type=positive_int, required=True)
    parser.add_argument('--env-seed', type=seed_int, required=True)
    parser.add_argument('--policy-seed', type=seed_int, required=True)
    parser.add_argument('--reward', choices=('minimal', 'shaped'), default='minimal')
    parser.add_argument('--max-decisions-per-episode', type=positive_int, default=10000)
    parser.add_argument('--log-every', type=positive_int, default=100)
    parser.add_argument('--output-dir', type=Path, required=True,
                        help='New Formal Random directory; existing paths are rejected.')
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        parser.error('--output-dir must be a new directory for each run')
    config = FormalRandomConfig(args.episodes, args.env_seed, args.policy_seed,
                                args.max_decisions_per_episode, args.reward, args.log_every)
    env = Tetris(gravity=True)
    try:
        summary = run_formal_random(env, config, output_dir=args.output_dir)
        print(json.dumps(dict(summary, output_dir=str(args.output_dir.resolve())), allow_nan=False))
        return 0 if summary['status'] == 'completed' else 1
    finally:
        env.close()


if __name__ == '__main__':
    raise SystemExit(main())
