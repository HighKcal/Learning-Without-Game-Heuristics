"""Run one complete headless random smoke-test game, without RL or heuristics."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tetris_gymnasium.envs.tetris import Tetris

from tetris_agent import GameRunner, GameStopReason, RandomDecisionAgent


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-seed", type=int, default=42)
    parser.add_argument("--decision-seed", type=int, default=42)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    env = Tetris(gravity=True)
    try:
        runner = GameRunner(RandomDecisionAgent(args.decision_seed))
        summary = runner.run(env, env_seed=args.env_seed, verbose=args.verbose)
        print(f"env seed: {args.env_seed} | decision seed: {args.decision_seed}")
        print(summary.format())
        return 0 if summary.stop_reason in {
            GameStopReason.TERMINATED, GameStopReason.TRUNCATED,
        } else 1
    finally:
        env.close()


if __name__ == "__main__":
    raise SystemExit(main())
