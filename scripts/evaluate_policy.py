"""Greedy fixed-state evaluation of existing online weights or an untrained MLP."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

import torch
from tetris_agent.rl.evaluation import evaluate_network, load_dataset, load_checkpoint_network, json_text
from tetris_agent.rl.network import QNetwork


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--checkpoint', type=Path)
    source.add_argument('--untrained-seed', type=int)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--reachability-mask', choices=('geometric', 'oracle'), default='geometric')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    if args.output and args.output.exists():
        parser.error('--output must be a new file')
    if args.device == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA requested but unavailable')
    from tetris_agent.native_preflight import native_preflight
    native_preflight()
    records, metadata = load_dataset(args.dataset)
    if metadata.get('status') != 'completed':
        parser.error('Dataset build did not complete; inspect its diagnostics')
    if args.checkpoint:
        network = load_checkpoint_network(args.checkpoint, args.device)
    else:
        if not 0 <= args.untrained_seed < 2**32:
            parser.error('--untrained-seed must be in [0, 2**32 - 1]')
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(args.untrained_seed)
            network = QNetwork().to(args.device)
    report = evaluate_network(network, records, reachability_mask_mode=args.reachability_mask)
    report['source'] = dict(dataset=str(args.dataset.resolve()),
        checkpoint=str(args.checkpoint.resolve()) if args.checkpoint else None,
        untrained_seed=args.untrained_seed, device=args.device,
        dataset_git_commit=metadata['git_commit'])
    if args.output:
        with args.output.open('x', encoding='utf-8') as stream:
            stream.write(json_text(report))
    print(json_text({k: v for k, v in report.items() if k != 'states'}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
