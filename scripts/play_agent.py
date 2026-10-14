"""Run existing frozen checkpoint weights with explicit research/play semantics."""
import argparse
from contextlib import ExitStack
from hashlib import sha256
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

import gymnasium as gym
import torch
from tetris_gymnasium.envs.tetris import Tetris

from tetris_agent.play import FrozenQRankingPolicy, PlayRunner
from tetris_agent.rl.evaluation import load_checkpoint_network
from tetris_agent.rl.policy import EpsilonGreedyPolicy
from tetris_agent.rl.runner import RLRunner


def file_hash(path):
    digest = sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


class RenderSteps(gym.Wrapper):
    """Render live Executor steps only; Planner calls the unwrapped base."""

    def __init__(self, env, delay_ms):
        super().__init__(env)
        self.delay_ms = delay_ms

    def step(self, action):
        result = self.env.step(action)
        self.env.render()
        import cv2
        cv2.waitKey(self.delay_ms)
        return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--mode', choices=('research', 'play'), default='research',
                        help='research preserves top-1 self-loops; play enables Planner fallback')
    parser.add_argument('--reachability-mask', choices=('geometric', 'oracle'), default='geometric',
                        help='oracle reuses RLRunner single-BFS mask/argmax/witness execution; '
                             'geometric preserves the existing mode behavior')
    parser.add_argument('--episodes', type=int, default=3)
    parser.add_argument('--env-seed', type=int, default=100000)
    parser.add_argument('--max-placements', type=int, default=100)
    parser.add_argument('--max-decisions', type=int, default=100,
                        help='research-only decision watchdog, including U self-loops')
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--output', type=Path, required=True, help='new directory for JSONL and summary')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--render', action='store_true', help='show live primitive actions in a game window')
    parser.add_argument('--step-delay-ms', type=int, default=80)
    args = parser.parse_args(argv)
    if min(args.episodes, args.max_placements, args.max_decisions, args.step_delay_ms) <= 0:
        parser.error('episode/placement/decision limits and step delay must be positive')
    if args.output.exists():
        parser.error('--output must be a new directory')
    if args.device == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA requested but unavailable')
    checkpoint_hash = file_hash(args.checkpoint)
    network = load_checkpoint_network(args.checkpoint, args.device).eval().requires_grad_(False)
    initial_weights = {name: value.clone() for name, value in network.state_dict().items()}
    forward_count = 0

    def count_forward(module, inputs, output):
        nonlocal forward_count
        forward_count += 1

    forward_hook = network.register_forward_hook(count_forward)
    args.output.mkdir(parents=True)
    report = dict(mode=args.mode, epsilon=0, checkpoint=str(args.checkpoint.resolve()),
                  reachability_mask_mode=args.reachability_mask,
                  inference_semantics=('single_bfs_mask_argmax_witness' if args.reachability_mask == 'oracle'
                                       else 'q_ranking_planner_fallback' if args.mode == 'play'
                                       else 'geometric_top1_self_loop'),
                  checkpoint_sha256_before=checkpoint_hash, device=args.device,
                  requested_episodes=args.episodes, max_placements=args.max_placements,
                  max_decisions=args.max_decisions, seed_start=args.env_seed,
                  training=False, trainer_constructed=False, optimizer_steps=0,
                  replay_updates=0, episodes=[])
    failed = False
    with ExitStack() as stack:
        stack.callback(forward_hook.remove)
        stream = stack.enter_context((args.output / 'decisions.jsonl').open('x', encoding='utf-8'))
        env = Tetris(gravity=True, render_mode='human' if args.render else None)
        if args.render:
            env = RenderSteps(env, args.step_delay_ms)
        stack.callback(env.close)
        for index in range(args.episodes):
            seed = args.env_seed + index

            def log(record):
                value = dict(episode=index + 1, env_seed=seed, **record)
                stream.write(json.dumps(value, allow_nan=False) + '\n')
                stream.flush()
                if args.verbose:
                    print(f"state={record['state_id']} top={record['top_q_action']} "
                          f"rejected={record['rejected_action_ids']} "
                          f"selected={record['selected_action_id']} rank={record['selected_rank']} "
                          f"planner_attempts={record['planner_attempts']} "
                          f"planner_ms={record['planner_ms']:.2f} "
                          f"executor={record['executor_result']} outcome={record['outcome']}")

            if args.mode == 'play' and args.reachability_mask == 'geometric':
                result = PlayRunner(FrozenQRankingPolicy(network)).run(
                    env, env_seed=seed, max_placements=args.max_placements, on_decision=log)
                summary = result.summary()
            else:
                # Oracle decisions always execute a reachable witness, so one
                # decision is one placement. Reuse the training runner unchanged;
                # no trainer, transition callback, or target-specific replanning.
                result = RLRunner(EpsilonGreedyPolicy(network),
                                  reachability_mask_mode=args.reachability_mask).run(
                    env, env_seed=seed, max_decisions=(args.max_placements
                                                     if args.mode == 'play' else args.max_decisions))
                summary = dict(placements=result.pieces_placed, lines=result.lines_cleared,
                               decisions=result.decision_count, unreachable_count=result.unreachable_count,
                               planner_call_count=result.planner_call_count,
                               oracle_computation_count=result.oracle_computation_count,
                               oracle_elapsed_seconds=result.oracle_elapsed_seconds,
                               reachable_action_count=result.reachable_action_count,
                               geometric_valid_action_count=result.geometric_valid_action_count,
                               masked_physical_u_count=result.masked_physical_u_count,
                               selected_physical_u_count=result.selected_physical_u_count,
                               terminated=result.terminated, truncated=result.truncated,
                               stop_reason=result.stop_reason, diagnostic=result.diagnostic)
            report['episodes'].append(dict(episode=index + 1, env_seed=seed, **summary))
            print(json.dumps(report['episodes'][-1], allow_nan=False))
            failed = bool(summary['diagnostic'])
            if failed:
                break  # No silent recovery or episode reset after any failure.
    report['checkpoint_sha256_after'] = file_hash(args.checkpoint)
    report['checkpoint_unchanged'] = checkpoint_hash == report['checkpoint_sha256_after']
    report['weights_unchanged'] = all(torch.equal(initial_weights[k], value)
                                      for k, value in network.state_dict().items())
    report['no_gradients'] = all(parameter.grad is None for parameter in network.parameters())
    report['network_forward_count'] = forward_count
    failed |= not all(report[k] for k in ('checkpoint_unchanged', 'weights_unchanged', 'no_gradients'))
    report['status'] = 'FAILED' if failed else 'COMPLETED'
    (args.output / 'summary.json').write_text(
        json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    return int(failed)


if __name__ == '__main__':
    raise SystemExit(main())
