"""Offline learner audit/benchmark. Never resumes or writes the baseline run.

Explicit subcommands write new files under --output only. Corpus comes from
RLRunner interaction, never fixed-evaluation labels. Timers/profilers stay here.
"""
import argparse
import cProfile
from contextlib import contextmanager
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import pstats
import subprocess
import sys
from time import perf_counter
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

import numpy as np
import torch
from torch.nn import functional as F
from tetris_gymnasium.envs.tetris import Tetris

from tetris_agent.rl.config import TrainingConfig
from tetris_agent.rl.dqn import DQNTrainer, LearningMetrics, bellman_targets
from tetris_agent.rl.evaluation import load_dataset, dataset_fingerprint, evaluate_network
from tetris_agent.rl.policy import FormalRandomPolicy
from tetris_agent.rl.replay import Transition
from tetris_agent.rl.runner import RLRunner
from tetris_agent.rl.state import SemanticState, geometric_mask
from tetris_agent.rl import dqn, state as state_module


ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / 'runs/geometric/baseline/seed_42/10k'
FIXED = ROOT / 'evaluation/fixed_eval_final_v1'


def write(path, data):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
        stream.write('\n')


def hashes():
    paths = list(BASELINE.rglob('*')) + list(FIXED.rglob('*'))
    paths += [Path(p) for p in subprocess.check_output(
        ['git', 'ls-files'], text=True).splitlines() if 'formal_random' in p]
    return {p.as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in paths if p.is_file()}


def finite(value):
    if isinstance(value, torch.Tensor):
        assert torch.isfinite(value).all(), 'Nonfinite checkpoint tensor'
    elif isinstance(value, dict):
        for v in value.values():
            finite(v)
    elif isinstance(value, (tuple, list)):
        for v in value:
            finite(v)
    elif isinstance(value, float):
        assert np.isfinite(value), 'Nonfinite JSON value'


def audit(output):
    output.mkdir(parents=True, exist_ok=False)
    write(output / 'preservation_before.json', hashes())
    write(output / 'preflight.json', {
        'head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'status': subprocess.check_output(['git', 'status', '--short'], text=True),
        'log': subprocess.check_output(['git', 'log', '--oneline', '--decorate', '-n', '20'], text=True),
        'baseline_tests': 'Must be run separately before audit; this script does not run pytest.',
        'torch': str(torch.__version__), 'numpy': np.__version__,
        'cuda': torch.version.cuda, 'device': torch.cuda.get_device_name(),
        'threads': torch.get_num_threads(),
    })
    config = json.loads((BASELINE / 'config.json').read_text())
    cfg = TrainingConfig(**config['training'])
    status = json.loads((BASELINE / 'status.json').read_text())
    summary = json.loads((BASELINE / 'summary.json').read_text())
    rows = [json.loads(line) for line in (BASELINE / 'metrics.jsonl').read_text().splitlines()]
    episodes = [json.loads(line) for line in (BASELINE / 'episodes.jsonl').read_text().splitlines()]
    finite([config, status, summary, rows, episodes])
    assert status['status'] == summary['status'] == 'completed'
    assert summary['decision_count'] == summary['stored_experiences'] == summary['replay_size'] == 10000
    assert summary['gradient_updates'] == 5001 and summary['target_sync_count'] == 1
    assert [r['decision_count'] for r in rows if r['event'] == 'periodic'] == list(range(100, 10001, 100))
    for r in rows:
        n = r['decision_count']
        assert r['stored_experiences'] == r['replay_size'] == n
        assert r['gradient_updates'] == max(0, n - 4999)
        assert r['target_sync_count'] == r['gradient_updates'] // 5000
        assert abs(r['epsilon'] - cfg.epsilon(n)) < 1e-12
        assert r['self_loop_count'] == r['unreachable_count']
        if r['last_learning']:
            assert abs(r['last_learning']['epsilon'] - cfg.epsilon(n - 1)) < 1e-12
    assert len(episodes) == summary['episode'] == 205
    for name in ('decision_count', 'unreachable_count', 'self_loop_count', 'pieces_placed', 'lines_cleared'):
        assert sum(e[name] for e in episodes) == summary[name]
    for e in episodes:
        assert e['diagnostic'] is None
        assert e['stored_transitions'] == e['decision_count']
        assert e['decision_count'] == e['pieces_placed'] + e['unreachable_count']
        assert e['unreachable_count'] == e['self_loop_count']
        assert abs(e['episode_return'] - (e['lines_cleared'] - .05 * e['unreachable_count'] - int(e['terminated']))) < 1e-8
    assert all(e['terminated'] and not e['truncated'] for e in episodes[:-1])
    assert episodes[-1]['truncated'] and not episodes[-1]['terminated']
    checkpoint_rows = []
    for path in sorted(BASELINE.glob('*.pt')):
        cp = torch.load(path, map_location='cpu', weights_only=True)
        finite(cp)
        n = cp['decision_count']
        assert cp['stored_experiences'] == n
        assert cp['gradient_updates'] == max(0, n - 4999)
        assert cp['target_sync_count'] == cp['gradient_updates'] // 5000
        assert cp['config'] == config['training']
        assert not cp['checkpoint_contract']['exact_resume_supported']
        assert abs(cp['epsilon']['next_selection_epsilon'] - cfg.epsilon(n)) < 1e-12
        trainer = DQNTrainer(cfg)
        trainer.online.load_state_dict(cp['online_state_dict'], strict=True)
        trainer.target.load_state_dict(cp['target_state_dict'], strict=True)
        trainer.optimizer.load_state_dict(cp['optimizer_state_dict'])
        steps = [int(s['step']) for s in cp['optimizer_state_dict']['state'].values()]
        assert not steps or set(steps) == {cp['gradient_updates']}
        checkpoint_rows.append(dict(file=path.name, decisions=n, updates=cp['gradient_updates'], syncs=cp['target_sync_count']))
    assert len(checkpoint_rows) == 11
    final = torch.load(BASELINE / 'checkpoint_final.pt', map_location='cpu', weights_only=True)
    last = torch.load(BASELINE / 'checkpoint_000010000.pt', map_location='cpu', weights_only=True)
    for field in ('online_state_dict', 'target_state_dict'):
        assert all(torch.equal(v, last[field][k]) for k, v in final[field].items())
    assert final['status'] == 'completed'
    metrics = [r['last_learning'] for r in rows if r['last_learning']]
    ranges = {key: dict(min=min(m[key] for m in metrics), max=max(m[key] for m in metrics), final=metrics[-1][key])
              for key in ('loss', 'mean_abs_td_error', 'max_abs_td_error', 'mean_selected_q', 'max_q', 'gradient_norm_before_clip', 'max_selected_q')}
    records, _ = load_dataset(FIXED)
    fingerprint = dataset_fingerprint(records)
    assert fingerprint == 'faa32dfae194863b05bb2e2e49ba224f9d0f3869564a28aa2ba0a50ad185e994'
    trainer.online.load_state_dict(final['online_state_dict'])
    evaluation = evaluate_network(trainer.online, records)
    write(output / 'fixed_eval_before.json', evaluation)
    result = dict(verdict='PASS', summary=summary, metric_ranges=ranges, logged_rows=len(rows),
                  checkpoints=checkpoint_rows, terminated=204, truncated=1, finite='PASS',
                  fixed_count=len(records), fingerprint=fingerprint,
                  limits='Snapshots omit replay and per-update metrics. Self-loop state identity and sync at update 5000 are supported by source/tests and counters, not recoverable per-transition evidence.')
    write(output / 'audit.json', result)
    print(json.dumps(result, indent=2), flush=True)


def corpus(output):
    transitions = []
    episodes = []
    runner = RLRunner(FormalRandomPolicy(20261004))
    env = Tetris(gravity=False)
    started = perf_counter()
    try:
        while len(transitions) < 512:
            result = runner.run(env, env_seed=1000 + len(episodes), max_decisions=512 - len(transitions), on_transition=transitions.append)
            assert result.diagnostic is None and result.decision_count
            episodes.append(asdict(result))
    finally:
        env.close()
    data = {}
    for name in ('state', 'next_state'):
        states = [getattr(t, name) for t in transitions]
        data[name + '_boards'] = np.stack([s.board for s in states])
        data[name + '_pieces'] = np.array([s.current_piece for s in states])
        data[name + '_queues'] = np.array([s.next_queue for s in states])
    for name in ('action_id', 'reward', 'terminated', 'truncated'):
        data[name] = np.array([getattr(t, name) for t in transitions])
    path = output / 'corpus.npz'
    assert not path.exists()
    np.savez_compressed(path, **data)
    write(output / 'corpus.json', dict(count=len(transitions), wall_seconds=perf_counter()-started, episodes=episodes,
                                     source='512 actual RLRunner decisions, FormalRandomPolicy seed 20261004, environment seeds 1000 onward; no evaluation data'))
    print('Corpus complete:', len(transitions), flush=True)


def make_trainer(output):
    trainer = DQNTrainer(TrainingConfig(decision_budget=10000, device='cuda'))
    with np.load(output / 'corpus.npz') as data:
        for i in range(len(data['action_id'])):
            states = [SemanticState(data[name+'_boards'][i], int(data[name+'_pieces'][i]), tuple(data[name+'_queues'][i]))
                      for name in ('state', 'next_state')]
            trainer.replay.add(Transition(states[0], int(data['action_id'][i]), float(data['reward'][i]), states[1],
                                          bool(data['terminated'][i]), bool(data['truncated'][i])))
    return trainer


class Timer:
    def __init__(self):
        self.values = {}

    @contextmanager
    def stage(self, name, cuda=False):
        if cuda:
            torch.cuda.synchronize()
        start = perf_counter()
        with torch.profiler.record_function(name):
            yield
        if cuda:
            torch.cuda.synchronize()
        self.values[name] = self.values.get(name, 0) + (perf_counter()-start)*1000


def timed_encode(states, timer, name, device):
    # Match production placement/dtypes/order, separating host prep and H2D.
    with timer.stage(name + ' host preparation'):
        boards = np.stack([s.board for s in states])
        ids = [(s.current_piece, *s.next_queue) for s in states]
    with timer.stage('CPU to GPU transfer', cuda=True):
        boards = torch.as_tensor(boards, device=device)
    with timer.stage(name + ' encoding', cuda=True):
        boards = boards.float().flatten(1)
    with timer.stage('CPU to GPU transfer', cuda=True):
        ids = torch.tensor(ids, device=device)
    with timer.stage(name + ' encoding', cuda=True):
        pieces = F.one_hot(ids, num_classes=7).float().flatten(1)
        result = torch.cat((boards, pieces), dim=1)
    return result


def timed_update(trainer, timer):
    torch.cuda.synchronize()
    started = perf_counter()
    with timer.stage('Replay sample'):
        batch = trainer.replay.sample(128)
    with timer.stage('Current-state extraction'):
        current = [t.state for t in batch]
    states = timed_encode(current, timer, 'Current-state', trainer.device)
    with timer.stage('Next-state extraction'):
        following = [t.next_state for t in batch]
    next_states = timed_encode(following, timer, 'Next-state', trainer.device)
    with timer.stage('Other tensor preparation'):
        terminal_values = [t.terminated for t in batch]
    with timer.stage('CPU to GPU transfer', cuda=True):
        terminated = torch.tensor(terminal_values, dtype=torch.bool, device=trainer.device)
    with timer.stage('Geometric next-action masks'):
        masks = np.stack([np.zeros(800, dtype=bool) if t.terminated else geometric_mask(t.next_state) for t in batch])
    with timer.stage('Other tensor preparation'):
        reward_values = [t.reward for t in batch]
        action_values = [t.action_id for t in batch]
    with timer.stage('CPU to GPU transfer', cuda=True):
        next_mask = torch.as_tensor(masks, device=trainer.device)
        rewards = torch.tensor(reward_values, dtype=torch.float32, device=trainer.device)
        actions = torch.tensor(action_values, device=trainer.device)
    with torch.no_grad():
        with timer.stage('Target forward next states', cuda=True):
            target_q = trainer.target(next_states)
        with timer.stage('Online forward next states', cuda=True):
            online_q = trainer.online(next_states)
        with timer.stage('DDQN target construction', cuda=True):
            targets = bellman_targets(rewards, terminated, next_mask, target_q, online_q, algorithm='double', gamma=.99)
    with timer.stage('Online forward current states', cuda=True):
        all_q = trainer.online(states)
    with timer.stage('Loss and TD', cuda=True):
        selected_q = all_q.gather(1, actions[:, None]).squeeze(1)
        td = (targets - selected_q).detach().abs()
        loss = F.smooth_l1_loss(selected_q, targets)
    with timer.stage('zero_grad', cuda=True):
        trainer.optimizer.zero_grad(set_to_none=True)
    with timer.stage('backward', cuda=True):
        loss.backward()
    with timer.stage('gradient clipping', cuda=True):
        norm = torch.nn.utils.clip_grad_norm_(trainer.online.parameters(), trainer.config.gradient_clip)
    with timer.stage('optimizer step', cuda=True):
        trainer.optimizer.step()
    with timer.stage('Counters and target sync', cuda=True):
        trainer.gradient_updates += 1
        if trainer.gradient_updates % trainer.config.target_sync_updates == 0:
            trainer.target.load_state_dict(trainer.online.state_dict())
            trainer.target_sync_count += 1
    # Production interleaves reductions and scalar copies. Split only in this
    # synchronized attribution pass; uninstrumented runs retain production order.
    with timer.stage('metrics reductions', cuda=True):
        values = (loss.detach(), td.mean(), td.max(), selected_q.detach().mean(), all_q.detach().max(), norm, selected_q.detach().max())
    with timer.stage('CUDA to CPU metrics extraction', cuda=True):
        loss_v, mean_td, max_td, mean_q, max_q, norm_v, max_selected = map(float, values)
        metrics = LearningMetrics(loss_v, mean_td, max_td, 1., trainer.gradient_updates, len(trainer.replay),
                                  trainer.target_sync_count, mean_q, max_q, norm_v, max_selected)
    torch.cuda.synchronize()
    timer.values['Total'] = (perf_counter()-started)*1000
    timer.values['Unattributed timer gaps'] = timer.values['Total'] - sum(v for k, v in timer.values.items() if k != 'Total')
    return metrics


def statistics(values):
    return dict(mean_ms=float(np.mean(values)), median_ms=float(np.median(values)),
                p95_ms=float(np.percentile(values, 95)), min_ms=float(min(values)), max_ms=float(max(values)))


def reference_mask(state):
    board = np.ones(state_module.GEOMETRY.board_shape, dtype=np.uint8)
    board[:20, 4:14] = state.board
    return state_module.SPACE.action_mask(board, state_module._SPAWN_MATRICES[state.current_piece])


def compare(output):
    """Same-process, alternating old/new, identical inputs and optimizer history."""
    old, new = make_trainer(output), make_trainer(output)
    mask_times = {'before': [], 'after': []}
    update_times = {'before': [], 'after': []}
    batch = old.replay.sample(128)
    corpus_mismatches = sum(not np.array_equal(reference_mask(t.next_state), geometric_mask(t.next_state))
                            for t in old.replay._items)
    assert corpus_mismatches == 0
    for i in range(13):
        for name, fn in [('before', reference_mask), ('after', geometric_mask)]:
            started = perf_counter()
            masks = np.stack([np.zeros(800, dtype=bool) if t.terminated else fn(t.next_state) for t in batch])
            elapsed = (perf_counter() - started) * 1000
            if i >= 3:
                mask_times[name].append(elapsed)
            if name == 'before':
                expected = masks
            else:
                assert np.array_equal(expected, masks)
    old, new = make_trainer(output), make_trainer(output)
    for i in range(13):
        metrics = []
        for name, trainer, fn in [('before', old, reference_mask), ('after', new, geometric_mask)]:
            with patch.object(dqn, 'geometric_mask', fn):
                torch.cuda.synchronize()
                started = perf_counter()
                metrics.append(asdict(trainer._update(trainer.replay.sample(128), 1.)))
                torch.cuda.synchronize()
                elapsed = (perf_counter() - started)*1000
            if i >= 3:
                update_times[name].append(elapsed)
        assert metrics[0] == metrics[1]
        assert all(torch.equal(v, new.online.state_dict()[k]) for k,v in old.online.state_dict().items())
        for a, b in zip(old.optimizer.state.values(), new.optimizer.state.values()):
            assert all(torch.equal(a[k], b[k]) for k in a)
        print('paired update', i+1, 'bit exact', flush=True)
    result = dict(warmup=3, measured=10, actual_corpus_states=512, corpus_mismatches=corpus_mismatches,
                  metrics_weights_adam_bit_exact=True,
                  mask={k: statistics(v) for k,v in mask_times.items()},
                  update={k: statistics(v) for k,v in update_times.items()},
                  mask_samples=mask_times, update_samples=update_times)
    write(output / 'paired_comparison.json', result)
    print(json.dumps(result, indent=2), flush=True)


def smoke_audit(output):
    path = ROOT / 'archive/baseline_preparation/evidence/runs/ddqn_postopt_smoke_5002_seed42'
    summary = json.loads((path / 'summary.json').read_text())
    config = TrainingConfig(**json.loads((path / 'config.json').read_text())['training'])
    rows = [json.loads(line) for line in (path / 'metrics.jsonl').read_text().splitlines()]
    old_rows = {r['decision_count']: r for r in map(json.loads, (BASELINE / 'metrics.jsonl').read_text().splitlines())}
    episodes = [json.loads(line) for line in (path / 'episodes.jsonl').read_text().splitlines()]
    finite([summary, rows, episodes])
    assert summary['status'] == 'completed' and summary['diagnostic'] is None
    assert summary['decision_count'] == summary['stored_experiences'] == summary['replay_size'] == 5002
    assert summary['gradient_updates'] == 3 and summary['target_sync_count'] == 0
    assert [r['decision_count'] for r in rows] == [*range(100, 5001, 100), 5002]
    for r in rows:
        n = r['decision_count']
        assert r['gradient_updates'] == max(0, n-4999)
        assert abs(r['epsilon'] - config.epsilon(n)) < 1e-12
        if n <= 5000:
            for field in ('decision_count', 'stored_experiences', 'replay_size', 'gradient_updates',
                          'target_sync_count', 'episode', 'episode_return', 'pieces_placed',
                          'lines_cleared', 'unreachable_count', 'self_loop_count', 'epsilon', 'last_learning'):
                assert r[field] == old_rows[n][field], (n, field)
    assert all(e['diagnostic'] is None and e['self_loop_count'] == e['unreachable_count'] for e in episodes)
    assert episodes[-1]['truncated'] and not episodes[-1]['terminated']
    assert sorted(p.name for p in path.glob('*.pt')) == [
        *[f'checkpoint_{i:09d}.pt' for i in range(1000, 5001, 1000)], 'checkpoint_final.pt']
    old = torch.load(BASELINE / 'checkpoint_000005000.pt', weights_only=True, map_location='cpu')
    new = torch.load(path / 'checkpoint_000005000.pt', weights_only=True, map_location='cpu')
    for field in ('online_state_dict', 'target_state_dict'):
        assert all(torch.equal(value, new[field][key]) for key, value in old[field].items())
    for key, left in old['optimizer_state_dict']['state'].items():
        right = new['optimizer_state_dict']['state'][key]
        assert all(torch.equal(left[k], right[k]) for k in left)
    final = torch.load(path / 'checkpoint_final.pt', weights_only=True, map_location='cpu')
    finite(final)
    assert final['gradient_updates'] == 3 and final['decision_count'] == 5002
    result = dict(summary=summary, checkpoints=6, original_first_5000_logged_values_identical=True,
                  original_first_update_weights_target_adam_bit_exact=True,
                  terminated=sum(e['terminated'] for e in episodes), truncated=sum(e['truncated'] for e in episodes),
                  notes='Budget 5002 has epsilon decay length 1. Compare original 10000-budget run only through experience 5000. Target-sync boundary 4999/5000/5001 covered by bit-exact CPU/CUDA tests.')
    write(output / 'smoke_audit.json', result)
    print(json.dumps(result, indent=2), flush=True)


def profile(output, label):
    trainer = make_trainer(output)
    instrumented = make_trainer(output)
    batch = trainer.replay.sample(128)
    ordinary = trainer._update(batch, 1.)
    timed = timed_update(instrumented, Timer())
    assert asdict(ordinary) == asdict(timed), 'Instrumented update metrics differ'
    assert all(torch.equal(v, instrumented.online.state_dict()[k]) for k, v in trainer.online.state_dict().items())
    for name, fn in [('raw', None), ('stages', timed_update)]:
        trainer = make_trainer(output)  # Identical seed, weights, Adam state, replay RNG.
        for _ in range(3):
            trainer._update(trainer.replay.sample(128), 1.)
        torch.cuda.synchronize()
        samples = []
        for i in range(20):
            if fn:
                timer = Timer()
                fn(trainer, timer)
                samples.append(timer.values)
            else:
                torch.cuda.synchronize()
                started = perf_counter()
                trainer._update(trainer.replay.sample(128), 1.)
                torch.cuda.synchronize()
                samples.append({'Total': (perf_counter()-started)*1000})
            if i % 5 == 4:
                print(label, name, i+1, 'updates', flush=True)
        summary = {k: statistics([row[k] for row in samples]) for k in samples[0]}
        for row in summary.values():
            row['percent_total'] = row['mean_ms'] / summary['Total']['mean_ms'] * 100
        write(output / f'{label}_{name}.json', dict(warmup=3, measured=20, samples=samples, summary=summary))
        if fn:
            lines = ['| Stage | Mean ms | Median ms | p95 ms | % total |', '|---|---:|---:|---:|---:|']
            lines += [f"| {k} | {v['mean_ms']:.3f} | {v['median_ms']:.3f} | {v['p95_ms']:.3f} | {v['percent_total']:.2f} |" for k,v in summary.items()]
            with (output / f'{label}_stages.md').open('x', encoding='utf-8') as stream:
                stream.write('\n'.join(lines)+'\n')
            print('\n'.join(lines), flush=True)
    trainer = make_trainer(output)
    for _ in range(3):
        trainer._update(trainer.replay.sample(128), 1.)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA], record_shapes=True) as prof:
        with torch.profiler.record_function('production learner update including sample'):
            trainer._update(trainer.replay.sample(128), 1.)
            torch.cuda.synchronize()
    prof.export_chrome_trace(str(output / f'{label}_torch_trace.json'))
    (output / f'{label}_torch.txt').write_text(prof.key_averages().table(sort_by='self_cpu_time_total', row_limit=35), encoding='utf-8')
    batch = trainer.replay.sample(128)
    profiler = cProfile.Profile()
    profiler.runcall(lambda: [geometric_mask(t.next_state) for t in batch if not t.terminated])
    profiler.dump_stats(str(output / f'{label}_mask.prof'))
    with (output / f'{label}_mask_calls.txt').open('x', encoding='utf-8') as stream:
        pstats.Stats(profiler, stream=stream).strip_dirs().sort_stats('cumulative').print_stats(40)
    print(label, 'profile complete', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=['audit', 'corpus', 'profile', 'compare', 'smoke-audit', 'integrity'])
    parser.add_argument('--output', type=Path, default=ROOT / '_validation/learner_profile')
    parser.add_argument('--label', default='before')
    args = parser.parse_args()
    if args.phase == 'audit':
        audit(args.output)
    elif args.phase == 'corpus':
        corpus(args.output)
    elif args.phase == 'profile':
        profile(args.output, args.label)
    elif args.phase == 'compare':
        compare(args.output)
    elif args.phase == 'smoke-audit':
        smoke_audit(args.output)
    else:
        before = json.loads((args.output / 'preservation_before.json').read_text())
        after = hashes()
        assert before == after
        records, _ = load_dataset(FIXED)
        trainer = make_trainer(args.output)
        cp = torch.load(BASELINE / 'checkpoint_final.pt', map_location='cpu', weights_only=True)
        trainer.online.load_state_dict(cp['online_state_dict'])
        result = evaluate_network(trainer.online, records)
        assert result == json.loads((args.output / 'fixed_eval_before.json').read_text())
        write(args.output / 'integrity_after.json', dict(unchanged=True, files=len(after), count=len(records), fingerprint=dataset_fingerprint(records), evaluation_identical=True))
        print('Preservation and deterministic fixed evaluation: PASS', flush=True)


if __name__ == '__main__':
    main()
