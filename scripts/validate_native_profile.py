"""Read-only preflight for the preserved Native source contract."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from tetris_agent.experiments import Registry, validate_environment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', default='native_tetris_031_oracle_perf_v2')
    args = parser.parse_args()
    profile = Registry(ROOT).profile(args.profile)
    validate_environment(profile)
    print(json.dumps(dict(profile=profile['id'],
                          source_files=len(profile['native_environment']['source_hashes']),
                          native_source_verified=True, training_executed=False)))


if __name__ == '__main__':
    main()
