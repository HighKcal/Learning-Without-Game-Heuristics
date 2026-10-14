"""Build a small offline fixed exam; never trains or runs an episode baseline."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from tetris_agent.rl.evaluation import save_dataset, json_text
from tetris_agent.rl.evaluation_build import BuildConfig, build_dataset, build_multiseed_dataset


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument('--seed', type=int)
    source.add_argument('--seeds', type=int, nargs='+', help='Process seeds in supplied order.')
    parser.add_argument('--states', type=int)
    parser.add_argument('--states-per-seed', type=int)
    parser.add_argument('--max-candidates', type=int, default=320)
    parser.add_argument('--max-seconds', type=float, default=300.)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.seeds is not None and args.states is not None:
        parser.error('--states is single-seed only; use --states-per-seed')
    if args.seeds is None and args.states_per_seed is not None:
        parser.error('--states-per-seed requires --seeds')
    if args.output_dir.exists():
        parser.error('--output-dir must be a new directory')
    progress = lambda row: print(json.dumps(row), file=sys.stderr, flush=True)
    try:
        if args.seeds is not None:
            records, metadata = build_multiseed_dataset(args.seeds,
                states_per_seed=args.states_per_seed if args.states_per_seed is not None else 64,
                max_candidates=args.max_candidates, max_seconds=args.max_seconds, on_progress=progress)
        else:
            config = BuildConfig(args.seed if args.seed is not None else 42,
                args.states if args.states is not None else 32, args.max_candidates, args.max_seconds)
            records, metadata = build_dataset(config, on_progress=progress)
    except ValueError as exc:
        parser.error(str(exc))
    save_dataset(args.output_dir, records, metadata)
    print(json_text(metadata))
    return 0 if metadata['status'] == 'completed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
