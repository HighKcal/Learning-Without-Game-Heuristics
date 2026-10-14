"""Controlled reconstructed-replay diagnostic; never runs an environment.

Production DQNTrainer._update is reused unchanged. Labels only reach ranking().
Run: python -B scripts/diagnose_u_feedback_timing.py
The output directory must not exist; validation gates both actual comparisons.
"""
from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
import random
import subprocess
import sys
import time
from collections import Counter
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import numpy as np
import torch
from tetris_agent.rl.config import TrainingConfig
from tetris_agent.rl.dqn import DQNTrainer, bellman_targets
from tetris_agent.rl.evaluation import EvaluationRecord, deserialize_state, state_hash, validate_records
from tetris_agent.rl.replay import Transition
from tetris_agent.rl.reward import reward_components
from tetris_agent.rl.state import encode_states, geometric_mask

OUT = ROOT / '_validation/u_feedback_timing'
GAMEPLAY = ROOT / 'analysis/geometric/frozen_gameplay'
D, H, BATCH, K, TOL = 128, 1024, 128, 128, 1e-6
ARMS = ('immediate', 'delayed')


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def fingerprint(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def tensor_tree_fingerprint(value):
    """Stable hash of values, including tensor dtype/shape and raw CPU bytes."""
    h = hashlib.sha256()
    def visit(v):
        if isinstance(v, torch.Tensor):
            h.update(canonical([str(v.dtype), list(v.shape)]).encode())
            h.update(v.detach().cpu().contiguous().numpy().tobytes())
        elif isinstance(v, dict):
            for key in sorted(v, key=str):
                h.update(canonical(key).encode())
                visit(v[key])
        elif isinstance(v, (tuple, list)):
            h.update(str(len(v)).encode())
            for item in v:
                visit(item)
        else:
            h.update(canonical(v).encode())
    visit(value)
    return h.hexdigest()


def transition_identity(t):
    return (state_hash(t.state), t.action_id, t.reward, state_hash(t.next_state), t.terminated, t.truncated)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT, text=True).strip()


def deterministic():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)


def rng_state(seed):
    return dict(python=random.Random(seed).getstate(), numpy=np.random.RandomState(seed).get_state(),
                torch=torch.Generator().manual_seed(seed).get_state())


def reset_rng(value):
    random.setstate(value['python'])
    np.random.set_state(value['numpy'])
    torch.set_rng_state(value['torch'])


def json_rng(value):
    n = value['numpy']
    return dict(python=value['python'], numpy=[n[0], n[1].tolist(), *n[2:]], torch=value['torch'].tolist())


def reconstruct(row, states, penalty):
    """Infer line/death components from the recorded common minimal reward."""
    s, ns = states[row['state_hash']], states[row['next_state_hash']]
    unreachable, terminal = row['unreachable'], row['terminated']
    if unreachable:
        assert state_hash(s) == state_hash(ns) and not terminal
        assert math.isclose(row['reward'], -.05, abs_tol=1e-9)
    lines = row['reward'] + (.05 if unreachable else 0.) + int(terminal)
    assert lines >= 0 and math.isclose(lines, round(lines), abs_tol=1e-8)
    reward = reward_components(s, ns, lines=round(lines), unreachable=unreachable,
                               terminated=terminal, mode='minimal', unreachable_penalty=penalty).total
    return Transition(s, row['action_id'], reward, ns, terminal, row['truncated'])


@dataclass
class Inputs:
    seed: int
    checkpoint: dict
    record: EvaluationRecord
    current: Transition
    background: list[Transition]
    occurrence_ids: list[str]
    metadata: dict
    paths: list[Path]


def read_trace(path):
    with gzip.open(path, 'rt', encoding='utf-8') as f:
        return [json.loads(line) for line in f]


def read_states(path):
    result = {}
    for item in read_json(path):
        state = deserialize_state(item['state'])
        assert state_hash(state) == item['state_hash']
        assert item['state_hash'] not in result
        result[item['state_hash']] = state
    return result


def load_inputs(seed):
    run = f'ddqn_100k_seed{seed}_penalty020'
    cp_path = ROOT / 'runs' / run / 'checkpoint_final.pt'
    cp = torch.load(cp_path, map_location='cpu', weights_only=True)
    cfg = cp['config']
    assert cfg['seed'] == seed and cfg['algorithm'] == 'double' and cfg['reward_mode'] == 'minimal'
    assert cfg['unreachable_penalty'] == -.20 and cfg['batch_size'] == BATCH
    assert cfg['target_sync_updates'] == 5000
    assert cp['gradient_updates'] % 5000 + H < 5000, 'Unexpected target sync in primary horizon'
    assert cp['target_sync_count'] == cp['gradient_updates'] // 5000
    trap_path = GAMEPLAY / 'traps' / f'{run}_1000.json'
    trap = read_json(trap_path)
    assert trap['run'] == run and trap['env_seed'] == 1000
    state = deserialize_state(trap['state'])
    assert state_hash(state) == trap['state_hash']
    record = EvaluationRecord(state, tuple(trap['valid_action_ids']), tuple(trap['reachable_action_ids']))
    validate_records([record])
    assert trap['action_id'] in set(record.valid_action_ids) - set(record.reachable_action_ids)
    cap_trace = GAMEPLAY / 'traces' / f'{run}_1000.jsonl.gz'
    cap_states = GAMEPLAY / 'states' / f'{run}_1000.json'
    cap_episode = GAMEPLAY / 'episodes' / f'{run}_1000.json'
    rows = read_trace(cap_trace)
    first = next(row for row in rows if row['unreachable'])
    assert first['decision'] == trap['first_unreachable_decision']
    assert first['action_id'] == trap['action_id'] and first['state_hash'] == record.state_hash
    current = reconstruct(first, read_states(cap_states), cfg['unreachable_penalty'])
    paths = [cp_path, trap_path, cap_trace, cap_states, cap_episode]
    background, ids, excluded, counts = [], [], [], []
    for env_seed in range(1001, 1010):
        name = f'{run}_{env_seed}'
        ep_path = GAMEPLAY / 'episodes' / (name + '.json')
        state_path = GAMEPLAY / 'states' / (name + '.json')
        trace_path = GAMEPLAY / 'traces' / (name + '.jsonl.gz')
        paths.extend([ep_path, state_path, trace_path])
        ep, states, trace = read_json(ep_path), read_states(state_path), read_trace(trace_path)
        assert ep['diagnostic'] is None and ep['run'] == run and ep['env_seed'] == env_seed
        assert len(trace) == ep['decisions']
        assert [r['decision'] for r in trace] == list(range(1, len(trace) + 1))
        accepted = 0
        for row in trace:
            oid = f'{name}:decision:{row["decision"]}'
            assert row['state_hash'] != record.state_hash, 'Background contains probe state'
            if row['state_hash'] not in states or row['next_state_hash'] not in states:
                assert seed == 42 and row['terminated'] and row['state_hash'] in states
                excluded.append(dict(occurrence_id=oid, reason='terminal next semantic state not saved', trace_row=row))
                continue
            background.append(reconstruct(row, states, cfg['unreachable_penalty']))
            ids.append(oid)
            accepted += 1
        counts.append(dict(env_seed=env_seed, recorded=len(trace), accepted=accepted))
    assert len(excluded) == (1 if seed == 42 else 0)
    assert len(background) >= BATCH and len(set(ids)) == len(ids)
    meta = dict(run=run, captured_decision=first['decision'], selected_u=first['action_id'],
                state_hash=record.state_hash, saved_start_margin=-trap['unreachable_advantage'],
                background_counts=counts, background_occurrences=len(background), excluded=excluded,
                reward_reconstruction='recorded common -.05 U -> checkpoint -.20; preserve line/death/truncation',
                original_replay_restored=False)
    return Inputs(seed, cp, record, current, background, ids, meta, paths)


@dataclass
class Plan:
    seed: int
    x: int
    current: int
    admission_sources: list[int]
    batches: list[list[int]]
    occurrence_ids: list[str]
    delay: int = D

    def batch(self, update, arm):
        assert arm in ARMS
        result = self.batches[update - 1].copy()
        if update in (1, self.delay + 1):
            u_now = (update == 1) == (arm == 'immediate')
            result[-1] = self.current if u_now else self.x
        return result

    def payload(self):
        return asdict(self)


def make_plan(occurrence_ids, seed, *, horizon=H, delay=D, batch_size=BATCH):
    assert 0 < delay < horizon and len(occurrence_ids) >= batch_size
    rng = random.Random(seed)
    n = len(occurrence_ids)
    x = rng.randrange(n)  # Uniform occurrence, before any outcomes.
    admissions = [rng.randrange(n) for _ in range(horizon - 1)]
    ids = list(occurrence_ids) + ['current_U'] + [f'admit:{i + 2}:{occurrence_ids[a]}'
                                               for i, a in enumerate(admissions)]
    source_ids = list(range(n)) + [-1] + admissions
    batches = []
    for update in range(1, horizon + 1):
        swap = update in (1, delay + 1)
        eligible = [i for i in range(n + update) if i != n and (not swap or source_ids[i] != x)]
        batch = rng.sample(eligible, batch_size - int(swap))
        batches.append(batch + [-1] if swap else batch)
    plan = Plan(seed, x, n, admissions, batches, ids, delay)
    validate_plan(plan, batch_size)
    return plan


def validate_plan(plan, batch_size=BATCH):
    exposure = {}
    horizon = len(plan.batches)
    for arm in ARMS:
        rows = [plan.batch(k, arm) for k in range(1, horizon + 1)]
        assert all(len(row) == batch_size and len(set(row)) == batch_size for row in rows)
        assert all(all(0 <= i < plan.current + k for i in row) for k, row in enumerate(rows, 1))
        assert [k for k, row in enumerate(rows, 1) if plan.current in row] == [1 if arm == 'immediate' else plan.delay + 1]
        exposure[arm] = Counter(i for row in rows for i in row)
    assert exposure['immediate'] == exposure['delayed']
    for k in range(1, horizon + 1):
        a, b = (plan.batch(k, arm) for arm in ARMS)
        changed = [i for i in range(batch_size) if a[i] != b[i]]
        assert changed == ([batch_size - 1] if k in (1, plan.delay + 1) else [])
        if changed:
            assert plan.x not in a[:-1] and plan.current not in a[:-1]
            origins = list(range(plan.current)) + [-1] + plan.admission_sources
            assert all(origins[i] != plan.x for i in a[:-1])
    return dict(total_exposures=sum(exposure['immediate'].values()),
                current_u_exposures=exposure['immediate'][plan.current],
                x_exposures=exposure['immediate'][plan.x], multiset_fingerprint=fingerprint(sorted(exposure['immediate'].items())))


def assert_equal(a, b):
    if isinstance(a, torch.Tensor):
        assert a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b)
    elif isinstance(a, np.ndarray):
        assert np.array_equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            assert_equal(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            assert_equal(x, y)
    else:
        assert a == b


def restore(cp, background, diagnostic_seed):
    cfg = TrainingConfig(**{**cp['config'], 'device': 'cpu'})
    trainer = DQNTrainer(cfg)
    trainer.online.load_state_dict(deepcopy(cp['online_state_dict']), strict=True)
    trainer.target.load_state_dict(deepcopy(cp['target_state_dict']), strict=True)
    trainer.optimizer.load_state_dict(deepcopy(cp['optimizer_state_dict']))
    for name in ('gradient_updates', 'stored_experiences', 'target_sync_count'):
        setattr(trainer, name, cp[name])
    for transition in background:
        trainer.replay.add(transition)
    trainer.replay._rng.seed(diagnostic_seed + 1)
    trainer.policy._rng.seed(diagnostic_seed + 2)
    assert_equal(trainer.online.state_dict(), cp['online_state_dict'])
    assert_equal(trainer.target.state_dict(), cp['target_state_dict'])
    assert_equal(trainer.optimizer.state_dict(), cp['optimizer_state_dict'])
    params = list(trainer.online.parameters())
    assert [id(p) for group in trainer.optimizer.param_groups for p in group['params']] == [id(p) for p in params]
    assert set(trainer.optimizer.state) <= set(params)
    return trainer


def trainer_equal(a, b):
    assert_equal(a.online.state_dict(), b.online.state_dict())
    assert_equal(a.target.state_dict(), b.target.state_dict())
    assert_equal(a.optimizer.state_dict(), b.optimizer.state_dict())
    assert_equal(a.config, b.config)
    for key in ('gradient_updates', 'stored_experiences', 'target_sync_count'):
        assert getattr(a, key) == getattr(b, key)
    assert len(a.replay) == len(b.replay) and a.replay._position == b.replay._position
    assert a.replay._rng.getstate() == b.replay._rng.getstate()
    assert a.policy._rng.getstate() == b.policy._rng.getstate()


def independent(a, b):
    def tensors(t):
        result = list(t.online.parameters()) + list(t.target.parameters())
        result += [v for state in t.optimizer.state.values() for v in state.values() if isinstance(v, torch.Tensor)]
        return result
    assert {t.data_ptr() for t in tensors(a)}.isdisjoint(t.data_ptr() for t in tensors(b))
    assert a.optimizer is not b.optimizer and a.replay is not b.replay
    for x, y in zip(a.replay._items, b.replay._items):
        assert x is not y and not np.shares_memory(x.state.board, y.state.board)
        assert not np.shares_memory(x.next_state.board, y.next_state.board)
        assert (state_hash(x.state), x.action_id, x.reward, state_hash(x.next_state), x.terminated, x.truncated) == (
            state_hash(y.state), y.action_id, y.reward, state_hash(y.next_state), y.terminated, y.truncated)


def ranking(q, record):
    valid, reachable = sorted(record.valid_action_ids), set(record.reachable_action_ids)
    assert np.isfinite(q).all()
    ranked = sorted(valid, key=lambda a: (-float(q[a]), a))
    top_r = next(a for a in ranked if a in reachable)
    top_u = next(a for a in ranked if a not in reachable)
    max_r, max_u = float(q[top_r]), float(q[top_u])
    margin = max_r - max_u
    return dict(maxR=max_r, maxU=max_u, margin=margin, topR_id=top_r, topU_id=top_u,
                greedy_id=ranked[0], greedy_class='R' if ranked[0] in reachable else 'U',
                corrected=margin > TOL)


@torch.no_grad()
def current_probe(trainer, transition):
    """No R/U labels: bootstrap mask is exclusively geometric."""
    q = trainer.online(encode_states([transition.state], trainer.device))[0]
    mask = torch.as_tensor(geometric_mask(transition.next_state), device=trainer.device)
    nq = trainer.online(encode_states([transition.next_state], trainer.device))[0]
    selected = int(nq.masked_fill(~mask, -torch.inf).argmax()) if not transition.terminated else None
    tq = trainer.target(encode_states([transition.next_state], trainer.device))[0]
    target = bellman_targets(torch.tensor([transition.reward], dtype=torch.float32),
                             torch.tensor([transition.terminated]), mask[None], tq[None], nq[None],
                             algorithm=trainer.config.algorithm, gamma=trainer.config.gamma)[0]
    chosen = float(q[transition.action_id])
    greedy = int(q.masked_fill(~torch.as_tensor(geometric_mask(transition.state)), -torch.inf).argmax())
    result = dict(online_q_u=chosen, pre_greedy_valid_id=greedy, bootstrap_selected_id=selected,
                  target_q_bootstrap=float(tq[selected]) if selected is not None else None,
                  bellman_target_u=float(target), signed_td_u=float(target) - chosen)
    assert all(math.isfinite(v) for v in result.values() if isinstance(v, float))
    return result, q.cpu().numpy()


def step(trainer, data, plan, update, arm):
    admitted = data.current if update == 1 else data.background[plan.admission_sources[update - 2]]
    trainer.replay.add(admitted)
    trainer.stored_experiences += 1
    assert len(trainer.replay) == plan.current + update
    ids = plan.batch(update, arm)
    batch = [trainer.replay._items[i].copy() for i in ids]
    probe, before_q = current_probe(trainer, data.current)
    before = ranking(before_q, data.record)
    extra = {}
    if update in (1, plan.delay + 1):
        xp, _ = current_probe(trainer, data.background[plan.x])
        extra = {'x_' + key: value for key, value in xp.items()}
    previous_sync = trainer.target_sync_count
    metrics = trainer._update(batch, trainer.epsilon)
    assert all(math.isfinite(v) for v in asdict(metrics).values())
    assert trainer.gradient_updates == data.checkpoint['gradient_updates'] + update
    assert trainer.target_sync_count == trainer.gradient_updates // trainer.config.target_sync_updates
    assert all(torch.isfinite(p).all() for p in trainer.online.parameters())
    assert all(int(state['step']) == trainer.gradient_updates for state in trainer.optimizer.state.values())
    assert all(torch.isfinite(v).all() for state in trainer.optimizer.state.values()
               for v in state.values() if isinstance(v, torch.Tensor))
    with torch.no_grad():
        after_q = trainer.online(encode_states([data.record.state]))[0].numpy()
    after = ranking(after_q, data.record)
    same_state = sum(state_hash(t.state) == data.record.state_hash for t in batch)
    assert same_state == int(plan.current in ids)
    return dict(diagnostic_id=f'u_feedback_seed{data.seed}', training_seed=data.seed, branch=arm,
                local_update=update, absolute_gradient_update=trainer.gradient_updates,
                sync_phase=trainer.gradient_updates % trainer.config.target_sync_updates,
                target_sync_happened=trainer.target_sync_count != previous_sync,
                batch_schedule_id=fingerprint(ids), current_U_in_batch=plan.current in ids,
                X_in_batch=plan.x in ids, same_probe_state_occurrences_in_batch=same_state,
                **after, delta_margin=after['margin'] - before['margin'],
                pre_maxR=before['maxR'], pre_maxU=before['maxU'],
                post_online_q_current_u=float(after_q[data.current.action_id]), **probe, **extra)


def summarize(rows, *, consecutive=K):
    flags = [r['corrected'] for r in rows]
    first = next((i + 1 for i, value in enumerate(flags) if value), None)
    onset = next((i + 1 for i in range(len(flags) - consecutive + 1)
                  if all(flags[i:i + consecutive])), None)
    return dict(first_robust_correction_update=first, sustained_correction_onset=onset,
                sustained_correction_confirmation_update=onset + consecutive - 1 if onset is not None else None,
                total_uncorrected_updates=sum(not value for value in flags),
                reversion_count=sum(a and not b for a, b in zip(flags, flags[1:])),
                final_margin=rows[-1]['margin'], final_greedy_class=rows[-1]['greedy_class'],
                sustained_correction_observed='YES' if onset is not None else 'NO', censored=onset is None)


def aa_validate(data, plan):
    a, b = (restore(data.checkpoint, data.background, plan.seed) for _ in range(2))
    trainer_equal(a, b)
    independent(a, b)
    start = rng_state(plan.seed)
    reset_rng(start)
    started = time.perf_counter()
    for update in range(1, len(plan.batches) + 1):
        before_rng = dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state())
        ra = step(a, data, plan, update, 'immediate')
        after_rng = dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state())
        reset_rng(before_rng)
        rb = step(b, data, plan, update, 'immediate')
        assert_equal(ra, rb)
        trainer_equal(a, b)
        assert_equal(after_rng, dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state()))
        if update % 256 == 0:
            print(f'A/A seed{data.seed}: {update}/{len(plan.batches)} updates exactly equal', flush=True)
    return dict(seed=data.seed, status='PASS', checked_updates=len(plan.batches),
                bitwise_online_target_adam=True, exact_counters_metrics_rng=True,
                elapsed_seconds=time.perf_counter() - started, schedule_fingerprint=fingerprint(plan.payload()))


def run_pair(data, plan, gates):
    assert gates['focused_tests'] == gates['regression_tests'] == 'PASS'
    assert {v['seed'] for v in gates['aa']} == {42, 123}
    assert all(v['status'] == 'PASS' and v['checked_updates'] == H for v in gates['aa'])
    gate = next(v for v in gates['aa'] if v['seed'] == data.seed)
    assert gate['schedule_fingerprint'] == fingerprint(plan.payload())
    branches = [restore(data.checkpoint, data.background, plan.seed) for _ in ARMS]
    trainer_equal(*branches)
    independent(*branches)
    initial_probe, initial_q = current_probe(branches[0], data.current)
    initial = ranking(initial_q, data.record)
    assert initial['greedy_class'] == 'U' and initial['margin'] <= 0
    assert math.isclose(initial['margin'], data.metadata['saved_start_margin'], abs_tol=1e-5)
    all_rows, summaries = [], []
    for arm, trainer in zip(ARMS, branches):
        reset_rng(rng_state(plan.seed))
        rows = []
        for update in range(1, H + 1):
            rows.append(step(trainer, data, plan, update, arm))
            if update % 256 == 0:
                print(f'A/B seed{data.seed} {arm}: {update}/{H}', flush=True)
        assert trainer.target_sync_count == data.checkpoint['target_sync_count']
        summary = dict(training_seed=data.seed, branch=arm, **summarize(rows))
        all_rows.extend(rows)
        summaries.append(summary)
    return all_rows, summaries, dict(ranking=initial, probe=initial_probe)


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def protected_snapshot(paths):
    tracked = [ROOT / name for name in git('ls-files').splitlines()]
    paths = sorted(set(tracked + paths))
    return {str(path.relative_to(ROOT)).replace('\\', '/'): file_hash(path) for path in paths}


def verify_integrity(hashes):
    changed = [name for name, digest in hashes.items() if file_hash(ROOT / name) != digest]
    assert not changed, f'Protected file changed: {changed}'
    return dict(protected_files=len(hashes), changed_files=changed, status='PASS')


def run_tests(output, files):
    command = [sys.executable, '-B', '-m', 'pytest', '-q', '-p', 'no:cacheprovider', *files]
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    output.write_text(result.stdout + result.stderr, encoding='utf-8')
    assert result.returncode == 0, f'Validation failed; see {output}'
    return dict(command=command, status='PASS', output=result.stdout.strip())


def paired(summaries):
    result = []
    for seed in (42, 123):
        a, b = (next(r for r in summaries if r['training_seed'] == seed and r['branch'] == arm) for arm in ARMS)
        fields = ('sustained_correction_onset', 'total_uncorrected_updates',
                  'first_robust_correction_update', 'final_margin', 'reversion_count')
        result.append(dict(training_seed=seed, **{key + '_immediate_minus_delayed':
            a[key] - b[key] if a[key] is not None and b[key] is not None else None for key in fields}))
    return result


def reports(summaries, pairs, updates, validation, elapsed):
    lines = ['# Reconstructed-replay U feedback timing diagnostic', '',
             'D=128; H=1024; batch=128; K=128; robust margin > 1e-6. CPU single-thread deterministic.', '',
             'This is not exact replay/resume of the original 100K training.', '',
             '| Seed | Arm | First robust | Sustained onset / confirmation | Uncorrected | Reversions | Final margin | Final greedy | Censored |',
             '|---|---|---:|---|---:|---:|---:|---|---|']
    for s in summaries:
        lines.append(f'| {s["training_seed"]} | {s["branch"]} | {s["first_robust_correction_update"]} | '
                     f'{s["sustained_correction_onset"]} / {s["sustained_correction_confirmation_update"]} | '
                     f'{s["total_uncorrected_updates"]} | {s["reversion_count"]} | {s["final_margin"]:.9f} | '
                     f'{s["final_greedy_class"]} | {s["censored"]} |')
    lines += ['', '## Paired results', '', 'Negative onset/uncorrected differences favor Immediate; null onset differences are censored, not zero.', '']
    for p in pairs:
        lines.append('- ' + canonical(p))
    mechanistic = []
    for seed in (42, 123):
        for arm in ARMS:
            rows = [r for r in updates[seed] if r['branch'] == arm]
            barrier = sum(r['bellman_target_u'] > r['pre_maxR'] for r in rows)
            small = sum(abs(r['signed_td_u']) <= 1e-3 for r in rows)
            q_drop = rows[-1]['post_online_q_current_u'] < rows[0]['online_q_u']
            other_top = sum(r['topU_id'] != validation['selected_u'][str(seed)] for r in rows)
            item = dict(seed=seed, branch=arm, target_above_pre_maxR_updates=barrier,
                        abs_signed_td_le_1e_minus3_updates=small, current_u_q_decreased=q_drop,
                        other_u_top_updates=other_top,
                        swap_responses={str(k): next(r for r in rows if r['local_update'] == k)['delta_margin'] for k in (1, 129)})
            mechanistic.append(item)
    lines += ['', '## Mechanistic observations', ''] + ['- ' + canonical(m) for m in mechanistic]
    lines += ['', '## Interpretation and limitations', '',
              'The identified intervention is the relative order of U and X; X timing also changes. Adam moments, nonlinear/shared function interference and DDQN bootstrap selection make intervening updates path dependent.',
              'Current-U Q decreasing without ranking correction is consistent with other-U promotion and/or reachable-Q movement; it does not prove a specific interference mechanism.',
              'A persistently high Bellman target relative to pre-update maxR motivates inspecting a bootstrap target barrier before attributing persistence to sampling latency. Small TD is reported with an explicitly descriptive 1e-3 threshold.',
              'Background occurrence multiplicity is retained, with a narrow frozen-policy distribution. Fresh diagnostic RNGs replace unavailable historical RNGs. No environment transitions are collected. UTD=1 per admitted recorded transition, not a replay of original environment decisions.',
              'No target sync occurs in this horizon. Sustained correction is finite-horizon K128 persistence and does not establish survival across a later sync.',
              'Two captured states and one X per seed cannot establish a general timing effect or prove the cause of original training long self-loops.', '',
              f'Total wall time including tests and A/A: {elapsed:.2f} seconds.', '']
    onset_benefit = all(p['sustained_correction_onset_immediate_minus_delayed'] is not None and
                        p['sustained_correction_onset_immediate_minus_delayed'] < 0 and
                        p['total_uncorrected_updates_immediate_minus_delayed'] < 0 for p in pairs)
    verdict = dict(status='completed', causal_scope='reconstructed-replay U/X relative-order intervention',
                   paired=pairs, mechanism=mechanistic,
                   timing_contribution='SUPPORTED_IN_BOTH_CAPTURED_CONDITIONS' if onset_benefit else 'NOT_CONSISTENTLY_SUPPORTED',
                   original_training_latency_cause_proven=False,
                   future_live_experiment='GO' if onset_benefit else 'NO-GO as immediate priority; inspect bootstrap/ranking barriers first',
                   elapsed_seconds=elapsed)
    lines += ['## Verdict', '', canonical(verdict), '']
    (OUT / 'report.md').write_text('\n'.join(lines), encoding='utf-8')
    write_json(OUT / 'verdict.json', verdict)


def main():
    started = time.perf_counter()
    deterministic()
    assert not OUT.exists(), 'New output directory required; never overwrite a prior experiment'
    data = [load_inputs(seed) for seed in (42, 123)]
    plans = [make_plan(d.occurrence_ids, 20261005 + d.seed) for d in data]
    hashes = protected_snapshot([p for d in data for p in d.paths])
    initial_status = git('status', '--short')
    OUT.mkdir()
    write_json(OUT / 'integrity_before.json', dict(hashes=hashes, git_head=git('rev-parse', 'HEAD'), git_status=initial_status))
    manifest = dict(kind='reconstructed_replay_timing_diagnostic', exact_training_replay=False,
                    protocol=dict(D=D, H=H, K=K, batch_size=BATCH, tolerance=TOL, swap_slot=BATCH - 1,
                                  diagnostic_seed_rule='20261005 + training_seed', device='cpu', threads=1,
                                  background_seeds=list(range(1001, 1010)), recorded_admissions_per_update=1,
                                  current_u_admitted_update=1, admission_stream='uniform background occurrences with replacement, fixed before learner execution'),
                    production_source_modified=False, software=dict(python=sys.version, torch=str(torch.__version__), numpy=np.__version__),
                    implementation_hashes={str(p.relative_to(ROOT)): file_hash(p) for p in [Path(__file__), ROOT / 'tests/test_u_feedback_timing_diagnostic.py']},
                    seeds={})
    for d, p in zip(data, plans):
        with gzip.open(OUT / f'seed{d.seed}_schedule.json.gz', 'wt', encoding='utf-8') as f:
            f.write(canonical(p.payload()))
        rs = json_rng(rng_state(p.seed))
        replay_rng = random.Random(p.seed + 1).getstate()
        policy_rng = random.Random(p.seed + 2).getstate()
        write_json(OUT / f'seed{d.seed}_rng_start.json', dict(global_rng=rs, replay_rng=replay_rng, policy_rng=policy_rng))
        manifest['seeds'][str(d.seed)] = dict(**d.metadata, schedule_fingerprint=fingerprint(p.payload()),
            admission_stream_fingerprint=fingerprint(p.admission_sources), occurrence_order_fingerprint=fingerprint(p.occurrence_ids),
            exposure_validation=validate_plan(p), x_occurrence_id=p.occurrence_ids[p.x],
            rng_fingerprint=fingerprint(dict(global_rng=rs, replay_rng=replay_rng, policy_rng=policy_rng)),
            starting_learner_fingerprints={key: tensor_tree_fingerprint(d.checkpoint[key]) for key in
                ('online_state_dict', 'target_state_dict', 'optimizer_state_dict')},
            starting_replay_fingerprint=fingerprint([transition_identity(t) for t in d.background]),
            replay_capacity=d.checkpoint['config']['replay_capacity'],
            replay_ring_position=len(d.background) % d.checkpoint['config']['replay_capacity'],
            checkpoint_counters={k: d.checkpoint[k] for k in ('gradient_updates', 'stored_experiences', 'target_sync_count')},
            input_hashes={str(path.relative_to(ROOT)).replace('\\', '/'): hashes[str(path.relative_to(ROOT)).replace('\\', '/')] for path in d.paths})
    write_json(OUT / 'manifest.json', manifest)
    validation = {}
    try:
        print('Focused tests first', flush=True)
        focused = run_tests(OUT / 'focused_tests.txt', ['tests/test_u_feedback_timing_diagnostic.py'])
        print('Related learner/evaluator regression tests', flush=True)
        regression = run_tests(OUT / 'regression_tests.txt', ['tests/test_rl_learning.py',
            'tests/test_rl_evaluation.py', '-k', 'not planner and not labeling and not native_exception and not noncanonical and not build and not collection'])
        validation.update(focused_tests='PASS', regression_tests='PASS', test_evidence=[focused, regression], aa=[])
        verify_integrity(hashes)
        for d, p in zip(data, plans):
            validation['aa'].append(aa_validate(d, p))
        write_json(OUT / 'validation.json', validation)  # Gate is persisted before A/B.
        (OUT / 'validation_report.md').write_text('# Validation\n\nFocused and related regression tests PASS.\n\n' +
            '\n'.join(f'- Seed {r["seed"]}: {r["checked_updates"]} A/A updates PASS; online/target/Adam bitwise equal, counters/metrics/RNG exact.' for r in validation['aa']) + '\n', encoding='utf-8')
        summaries, updates = [], {}
        validation.update(initial_metrics={}, selected_u={})
        for d, p in zip(data, plans):
            verify_integrity(hashes)
            rows, ss, initial = run_pair(d, p, validation)
            write_csv(OUT / f'seed{d.seed}_updates.csv', rows)
            updates[d.seed] = rows
            summaries.extend(ss)
            validation['initial_metrics'][str(d.seed)] = initial
            validation['selected_u'][str(d.seed)] = d.current.action_id
        write_csv(OUT / 'summary.csv', summaries)
        pairs = paired(summaries)
        write_json(OUT / 'validation.json', validation)
        reports(summaries, pairs, updates, validation, time.perf_counter() - started)
        integrity = verify_integrity(hashes)
        integrity.update(git_status_before=initial_status, git_status_after=git('status', '--short'),
                         git_head_after=git('rev-parse', 'HEAD'), prohibited_git_operations=False)
        write_json(OUT / 'integrity_final.json', integrity)
        print(canonical(dict(status='completed', paired=pairs, integrity=integrity['status'])), flush=True)
    except BaseException as exc:
        write_json(OUT / 'failure.json', dict(status='STOP', error=f'{type(exc).__name__}: {exc}', validation=validation))
        raise


if __name__ == '__main__':
    main()
