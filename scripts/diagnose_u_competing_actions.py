"""Offline competing-U instrumentation; default is a four-update validation smoke.

Uses the saved timing schedule and its hash-verified reconstructed replay inputs.
No environment/planner calls, production edits, or active-run directory discovery.
Full execution must be explicitly requested with --full after the 500K run ends.
The zero-momentum intervention resets moments ONCE, retaining Adam step timing.
Head/representation decomposition is algebraic, not a causal attribution of Adam.
"""
from __future__ import annotations

import argparse
import gzip
import importlib.util
import json
import os
import sys
import time
from collections import Counter
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    'competing_timing_reuse', ROOT / 'scripts/diagnose_u_feedback_timing.py')
timing = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = timing
spec.loader.exec_module(timing)

SOURCE = ROOT / 'analysis/geometric/feedback_timing'
OUT = ROOT / '_validation/u_competing_actions'
CONDITIONS = ('restored_adam', 'zero_momentum_adam')
DECOMP_ATOL, DECOMP_RTOL = 1e-6, 1e-6
COUNTERS = ('gradient_updates', 'stored_experiences', 'target_sync_count')


def allowed_checkpoint_paths():
    return tuple(ROOT / f'runs/geometric/penalty_020/seed_{seed}/100k/checkpoint_final.pt'
                 for seed in (42, 123))


class RunAccessGuard:
    """Python audit fence: runs/ permits only the two historical checkpoint reads.

    Does not discover, stat, hash, or traverse any active run. CPython audit events
    cover file opens and directory traversal; this is evidence for this diagnostic
    process, not monitoring of the independent training process.
    """
    def __init__(self):
        self.enabled = True
        self.reads = Counter()
        self.denied = 0
        self.runs = os.path.normcase(os.path.abspath(ROOT / 'runs'))
        self.allowed = {os.path.normcase(os.path.abspath(p)) for p in allowed_checkpoint_paths()}

    def __call__(self, event, args):
        if not self.enabled or event not in ('open', 'os.listdir', 'os.scandir'):
            return
        value = args[0]
        if not isinstance(value, (str, bytes, os.PathLike)):
            return
        path = os.path.normcase(os.path.abspath(os.fsdecode(value)))
        if path != self.runs and not path.startswith(self.runs + os.sep):
            return
        mode, flags = (args[1:3] if event == 'open' else (None, None))
        writes = ((isinstance(mode, str) and any(c in mode for c in 'wax+'))
                  or (isinstance(flags, int) and flags &
                      (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND)))
        if event != 'open' or path not in self.allowed or writes:
            self.denied += 1
            raise PermissionError('Diagnostic may only read the two fixed historical checkpoints under runs/')
        self.reads[path] += 1

    def evidence(self):
        return dict(allowed_historical_checkpoint_reads=dict(self.reads),
                    denied_access_attempts=self.denied, active_run_accesses=0,
                    scope='Python file-open/directory-traversal audit; no active-run inspection')


def safe_input_path(name):
    """Reject even a modified manifest attempting to redirect us into other runs."""
    path = ROOT / name.replace('\\', '/')
    absolute = Path(os.path.abspath(path))
    checkpoints = {Path(os.path.abspath(p)) for p in allowed_checkpoint_paths()}
    gameplay = ROOT / 'analysis/geometric/frozen_gameplay'
    if absolute not in checkpoints and not absolute.is_relative_to(gameplay):
        raise ValueError(f'Not a historical diagnostic input: {name}')
    return absolute


def protected_snapshot():
    # Intentionally no git ls-files snapshot or traversal of runs/.
    paths = list(SOURCE.rglob('*')) + list((ROOT / 'src').rglob('*.py'))
    paths += [ROOT / 'scripts/diagnose_u_feedback_timing.py',
              ROOT / 'tests/test_u_feedback_timing_diagnostic.py']
    manifest = timing.read_json(SOURCE / 'manifest.json')
    paths += [safe_input_path(name) for seed in manifest['seeds'].values()
              for name in seed['input_hashes']]
    return {str(p.relative_to(ROOT)).replace('\\', '/'): timing.file_hash(p)
            for p in sorted(set(paths)) if p.is_file()}


def verify_snapshot(hashes):
    changed = [name for name, digest in hashes.items() if timing.file_hash(ROOT / name) != digest]
    if changed:
        raise AssertionError(f'Historical artifact or production source changed: {changed}')
    return dict(status='PASS', protected_files=len(hashes), changed_files=[])


def load_saved(seed):
    if seed not in (42, 123):
        raise ValueError('Only the two saved historical seeds are supported')
    manifest = timing.read_json(SOURCE / 'manifest.json')
    meta = manifest['seeds'][str(seed)]
    for name, digest in manifest['implementation_hashes'].items():
        path = ROOT / name.replace('\\', '/')
        if path not in (ROOT / 'scripts/diagnose_u_feedback_timing.py',
                        ROOT / 'tests/test_u_feedback_timing_diagnostic.py'):
            raise ValueError('Unexpected saved implementation path')
        assert timing.file_hash(path) == digest
    for name, digest in meta['input_hashes'].items():
        assert timing.file_hash(safe_input_path(name)) == digest, f'Historical input changed: {name}'
    data = timing.load_inputs(seed)
    assert {str(p.relative_to(ROOT)).replace('\\', '/') for p in data.paths} == set(meta['input_hashes'])
    with gzip.open(SOURCE / f'seed{seed}_schedule.json.gz', 'rt', encoding='utf-8') as stream:
        plan = timing.Plan(**json.load(stream))
    assert timing.fingerprint(plan.payload()) == meta['schedule_fingerprint']
    timing.validate_plan(plan)
    assert plan.current == len(data.background)
    assert plan.occurrence_ids[:plan.current] == data.occurrence_ids
    # The old manifest fingerprints the FULL occurrence order, including admissions.
    assert timing.fingerprint(plan.occurrence_ids) == meta['occurrence_order_fingerprint']
    assert timing.fingerprint(plan.admission_sources) == meta['admission_stream_fingerprint']
    saved_rng = timing.read_json(SOURCE / f'seed{seed}_rng_start.json')
    assert timing.fingerprint(saved_rng) == meta['rng_fingerprint']
    assert timing.fingerprint(timing.json_rng(timing.rng_state(plan.seed))) == timing.fingerprint(saved_rng['global_rng'])
    assert timing.fingerprint([timing.transition_identity(t) for t in data.background]) == meta['starting_replay_fingerprint']
    for name, digest in meta['starting_learner_fingerprints'].items():
        assert timing.tensor_tree_fingerprint(data.checkpoint[name]) == digest
    assert {name: data.checkpoint[name] for name in COUNTERS} == meta['checkpoint_counters']
    assert data.current.action_id == meta['selected_u']
    if seed == 42:
        assert data.current.action_id == 701
        assert 705 in set(data.record.valid_action_ids) - set(data.record.reachable_action_ids)
    return data, plan


def make_conditions(data, plan):
    trainers = [timing.restore(data.checkpoint, data.background, plan.seed) for _ in CONDITIONS]
    restored, zero = trainers
    timing.trainer_equal(restored, zero)
    timing.independent(restored, zero)
    old = deepcopy(zero.optimizer.state_dict())
    assert old['state'], 'A populated Adam checkpoint is required'
    for group in zero.optimizer.param_groups:
        assert group['weight_decay'] == 0 and not group['amsgrad']
    for state in zero.optimizer.state.values():
        assert set(state) == {'step', 'exp_avg', 'exp_avg_sq'}
        state['exp_avg'].zero_()
        state['exp_avg_sq'].zero_()
    new = zero.optimizer.state_dict()
    timing.assert_equal(old['param_groups'], new['param_groups'])
    for key in old['state']:
        timing.assert_equal(old['state'][key]['step'], new['state'][key]['step'])
        assert not torch.count_nonzero(new['state'][key]['exp_avg'])
        assert not torch.count_nonzero(new['state'][key]['exp_avg_sq'])
    timing.assert_equal(restored.online.state_dict(), zero.online.state_dict())
    timing.assert_equal(restored.target.state_dict(), zero.target.state_dict())
    assert [getattr(restored, k) for k in COUNTERS] == [getattr(zero, k) for k in COUNTERS]
    return trainers


@dataclass
class Probe:
    h: torch.Tensor
    w: torch.Tensor
    b: torch.Tensor
    q: np.ndarray


@torch.no_grad()
def capture_probe(trainer, state):
    network = trainer.online
    assert isinstance(network, torch.nn.Sequential)
    assert isinstance(network[-1], torch.nn.Linear) and network[-1].out_features == 800
    x = timing.encode_states([state], trainer.device)
    h = x
    for layer in list(network.children())[:-1]:
        h = layer(h)
    q = network[-1](h)[0].cpu().numpy().copy()
    assert np.isfinite(q).all()
    return Probe(h[0].detach().cpu().double().clone(),
                 network[-1].weight.detach().cpu().double().clone(),
                 network[-1].bias.detach().cpu().double().clone(), q)


def decompose(before, after, action):
    head = torch.dot(after.w[action] - before.w[action], before.h) + after.b[action] - before.b[action]
    representation = torch.dot(after.w[action], after.h - before.h)
    affine_delta = (torch.dot(after.w[action], after.h) + after.b[action]
                    - torch.dot(before.w[action], before.h) - before.b[action])
    delta = float(after.q[action]) - float(before.q[action])
    residual = float(head + representation) - delta
    assert torch.isclose(head + representation, affine_delta, atol=1e-12, rtol=1e-12)
    assert abs(residual) <= DECOMP_ATOL + DECOMP_RTOL * abs(delta)
    return dict(q_before=float(before.q[action]), q_after=float(after.q[action]), delta_q=delta,
                head_contribution=float(head), representation_contribution=float(representation),
                affine_delta_q=float(affine_delta), decomposition_residual=residual,
                affine_identity_residual=float(head + representation - affine_delta))


def ranked_probe(q, record):
    result = timing.ranking(q, record)
    reachable = set(record.reachable_action_ids)
    ordered = sorted(record.valid_action_ids, key=lambda a: (-float(q[a]), a))
    result.update(top5U_ids=[a for a in ordered if a not in reachable][:5],
                  top5R_ids=[a for a in ordered if a in reachable][:5])
    return result


def exposure(batch, probe_hash, actions):
    counts = Counter(t.action_id for t in batch)
    same = [t for t in batch if timing.state_hash(t.state) == probe_hash]
    paired = Counter(t.action_id for t in same)
    return {a: dict(action_id_count=counts[a], probe_state_count=len(same),
                    same_state_same_action_count=paired[a]) for a in sorted(actions)}


def interest_actions(original, competing, before, after):
    roles = {original: ['original_U']}
    if competing is not None:
        roles.setdefault(competing, []).append('competing_U')
    for phase, rank in (('pre', before), ('post', after)):
        for key in ('topU_id', 'topR_id', 'greedy_id'):
            roles.setdefault(rank[key], []).append(f'{phase}_{key}')
        for kind in ('U', 'R'):
            for index, action in enumerate(rank[f'top5{kind}_ids'], 1):
                roles.setdefault(action, []).append(f'{phase}_top{index}{kind}')
    return roles


def instrumented_update(trainer, data, plan, update, arm, condition):
    """Labels are used only by observation; _update receives Transition copies."""
    admitted = data.current if update == 1 else data.background[plan.admission_sources[update - 2]]
    trainer.replay.add(admitted)
    trainer.stored_experiences += 1
    assert len(trainer.replay) == plan.current + update
    ids = plan.batch(update, arm)
    batch = [trainer.replay._items[i].copy() for i in ids]
    before = capture_probe(trainer, data.current.state)
    pre = ranked_probe(before.q, data.record)
    head = trainer.online[-1]
    moment_w = trainer.optimizer.state[head.weight]['exp_avg'].detach().clone()
    moment_b = trainer.optimizer.state[head.bias]['exp_avg'].detach().clone()
    second_w = trainer.optimizer.state[head.weight]['exp_avg_sq'].detach().clone()
    second_b = trainer.optimizer.state[head.bias]['exp_avg_sq'].detach().clone()
    old_sync = trainer.target_sync_count
    metrics = trainer._update(batch, trainer.epsilon)  # unchanged production learner
    assert all(np.isfinite(v) for v in asdict(metrics).values())
    assert trainer.gradient_updates == data.checkpoint['gradient_updates'] + update
    assert trainer.target_sync_count == data.checkpoint['target_sync_count'] == old_sync
    assert all(int(s['step']) == trainer.gradient_updates for s in trainer.optimizer.state.values())
    assert all(torch.isfinite(p).all() for p in trainer.online.parameters())
    assert all(torch.isfinite(v).all() for s in trainer.optimizer.state.values()
               for v in s.values() if isinstance(v, torch.Tensor))
    after = capture_probe(trainer, data.current.state)
    post = ranked_probe(after.q, data.record)
    competing = 705 if data.seed == 42 else None
    roles = interest_actions(data.current.action_id, competing, pre, post)
    counts = exposure(batch, data.record.state_hash, roles)
    assert counts[data.current.action_id]['probe_state_count'] == int(plan.current in ids)
    cosine = torch.nn.functional.cosine_similarity(before.h[None], after.h[None], eps=1e-12)
    row = dict(training_seed=data.seed, timing_arm=arm, condition=condition,
               local_update=update, absolute_gradient_update=trainer.gradient_updates,
               batch_occurrence_indices=ids, batch_schedule_id=timing.fingerprint(ids),
               pre=pre, post=post, topU_identity_changed=pre['topU_id'] != post['topU_id'],
               hidden_delta_norm=float(torch.linalg.vector_norm(after.h - before.h)),
               hidden_cosine_similarity=float(cosine),
               hidden_before_norm=float(torch.linalg.vector_norm(before.h)),
               hidden_after_norm=float(torch.linalg.vector_norm(after.h)), learner_metrics=asdict(metrics))
    actions = []
    for action in sorted(roles):
        gw, gb = head.weight.grad[action], head.bias.grad[action]
        grad_zero = not bool(torch.count_nonzero(gw)) and not bool(gb)
        dw, db = after.w[action] - before.w[action], after.b[action] - before.b[action]
        moved = bool(torch.count_nonzero(dw)) or bool(db)
        moment_nonzero = bool(torch.count_nonzero(moment_w[action])) or bool(moment_b[action])
        item = dict(training_seed=data.seed, timing_arm=arm, condition=condition,
                    local_update=update, action_id=action, roles=roles[action],
                    **decompose(before, after, action), **counts[action],
                    head_gradient_weight_norm=float(torch.linalg.vector_norm(gw)),
                    head_gradient_bias=float(gb), head_gradient_zero=grad_zero,
                    head_weight_delta_norm=float(torch.linalg.vector_norm(dw)), head_bias_delta=float(db),
                    adam_exp_avg_weight_norm_before=float(torch.linalg.vector_norm(moment_w[action])),
                    adam_exp_avg_bias_before=float(moment_b[action]),
                    adam_exp_avg_sq_weight_norm_before=float(torch.linalg.vector_norm(second_w[action])),
                    adam_exp_avg_sq_bias_before=float(second_b[action]),
                    zero_gradient_head_moved=moved and grad_zero,
                    zero_gradient_head_moved_with_prior_moment=moved and grad_zero and moment_nonzero)
        if counts[action]['action_id_count'] == 0:
            assert grad_zero, 'Unexposed head row must have zero current gradient'
        actions.append(item)
    return row, actions


def churn(initial, rows, original):
    """Durations are post-update observations; update0 is separate.

    Switch count includes update0 -> update1. Durations never silently include the
    initial checkpoint as an extra optimizer update.
    """
    ids = [row['post']['topU_id'] for row in rows]
    durations = Counter(ids)
    runs, current, length = [], None, 0
    for action in ids:
        if action != current:
            if length:
                runs.append(length)
            current, length = action, 1
        else:
            length += 1
    if length:
        runs.append(length)
    trajectory = [initial['topU_id']] + ids
    return dict(topU_identity_switch_count=sum(a != b for a, b in zip(trajectory, trajectory[1:])),
                unique_topU_action_count=len(set(trajectory)),
                longest_same_topU_duration=max(runs, default=0), original_topU_updates=durations[original],
                competing_topU_updates=sum(v for a, v in durations.items() if a != original),
                topU_duration_by_action={str(a): v for a, v in sorted(durations.items())},
                initial_topU_id=initial['topU_id'], duration_unit='post-update observations',
                switch_count_includes_initial_to_update1=True)


def summarize(initial, rows, actions, original, competing):
    result = churn(initial, rows, original)
    original_q = {a['local_update']: a['q_after'] for a in actions if a['action_id'] == original}
    result.update(original_action_id=original, competing_action_id=competing,
                  first_competing_topU_update=next((r['local_update'] for r in rows
                      if r['post']['topU_id'] == competing), None),
                  original_below_topR_updates=sum(original_q[r['local_update']]
                      < r['post']['maxR'] - timing.TOL for r in rows),
                  replacement_after_original_below_topR_updates=sum(
                      r['post']['topU_id'] != original and original_q[r['local_update']]
                      < r['post']['maxR'] - timing.TOL for r in rows),
                  identity_switches_with_original_below_topR=sum(
                      r['topU_identity_changed'] and original_q[r['local_update']]
                      < r['post']['maxR'] - timing.TOL for r in rows),
                  max_abs_decomposition_residual=max(abs(a['decomposition_residual']) for a in actions))
    return result


def run_pair(data, plan, arm, updates):
    assert 1 <= updates <= len(plan.batches)
    restored, zero = make_conditions(data, plan)
    initial = ranked_probe(capture_probe(restored, data.current.state).q, data.record)
    all_rows, all_actions, summaries = [], [], []
    prior_exposure = Counter()
    timing.reset_rng(timing.rng_state(plan.seed))
    # Lockstep conditions have the same copied batch order and admission stream.
    for update in range(1, updates + 1):
        pair_rows, pair_actions = [], []
        shared_rng = dict(python=timing.random.getstate(), numpy=np.random.get_state(),
                          torch=torch.get_rng_state())
        after_rngs = []
        for trainer, condition in zip((restored, zero), CONDITIONS):
            timing.reset_rng(shared_rng)
            row, actions = instrumented_update(trainer, data, plan, update, arm, condition)
            pair_rows.append(row)
            pair_actions.append(actions)
            after_rngs.append(dict(python=timing.random.getstate(), numpy=np.random.get_state(),
                                   torch=torch.get_rng_state()))
        timing.assert_equal(*after_rngs)
        assert pair_rows[0]['batch_occurrence_indices'] == pair_rows[1]['batch_occurrence_indices']
        # Interest sets may diverge; exposure is audited on their union.
        union = {a['action_id'] for aa in pair_actions for a in aa}
        actual_batches = [[t.replay._items[i] for i in plan.batch(update, arm)] for t in (restored, zero)]
        timing.assert_equal(exposure(actual_batches[0], data.record.state_hash, union),
                            exposure(actual_batches[1], data.record.state_hash, union))
        timing.assert_equal([timing.transition_identity(t) for t in actual_batches[0]],
                            [timing.transition_identity(t) for t in actual_batches[1]])
        for records in pair_actions:
            for record in records:
                record['prior_action_id_exposures_in_diagnostic'] = prior_exposure[record['action_id']]
        prior_exposure.update(t.action_id for t in actual_batches[0])
        all_rows.extend(pair_rows)
        all_actions.extend(a for aa in pair_actions for a in aa)
        if update % 128 == 0:
            print(f'seed{data.seed} {arm}: {update}/{updates}', flush=True)
    for condition in CONDITIONS:
        rows = [r for r in all_rows if r['condition'] == condition]
        actions = [a for a in all_actions if a['condition'] == condition]
        summaries.append(dict(training_seed=data.seed, timing_arm=arm, condition=condition,
                              **summarize(initial, rows, actions, data.current.action_id,
                                          705 if data.seed == 42 else None)))
    # Counterfactual differences are descriptive: paths/targets diverge after update1.
    comparisons = []
    action_index = {(x['condition'], x['local_update'], x['action_id']): x for x in all_actions}
    for update in range(1, updates + 1):
        for action in (data.current.action_id, *([705] if data.seed == 42 else [])):
            a, b = [action_index[condition, update, action]
                    for condition in CONDITIONS]
            comparisons.append(dict(training_seed=data.seed, timing_arm=arm,
                local_update=update, action_id=action,
                restored_minus_zero_q_after=a['q_after'] - b['q_after'],
                restored_minus_zero_head_contribution=a['head_contribution'] - b['head_contribution'],
                restored_minus_zero_representation_contribution=
                    a['representation_contribution'] - b['representation_contribution']))
    return initial, all_rows, all_actions, summaries, comparisons


def write_jsonl(path, rows):
    with path.open('x', encoding='utf-8') as stream:
        for row in rows:
            stream.write(timing.canonical(row) + '\n')


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--smoke', action='store_true', help='seed42, Immediate, at most 8 updates (default)')
    mode.add_argument('--full', action='store_true', help='H1024, both saved seeds and timing arms; run after training ends')
    parser.add_argument('--updates', type=int, default=None, help='smoke only: 1..8, default 4')
    args = parser.parse_args(argv)
    if args.full and args.updates is not None:
        parser.error('--updates is smoke-only')
    if args.updates is not None and not 1 <= args.updates <= 8:
        parser.error('smoke updates must be 1..8')
    return args


def main(argv=None):
    args = parse_args(argv)
    guard = RunAccessGuard()
    sys.addaudithook(guard)
    started = time.perf_counter()
    timing.deterministic()  # CPU only, torch single thread
    output = OUT / ('full' if args.full else 'smoke')
    if output.exists():
        raise FileExistsError('Refusing to overwrite a previous diagnostic output')
    hashes = protected_snapshot()
    updates = timing.H if args.full else (args.updates or 4)
    seeds = (42, 123) if args.full else (42,)
    arms = timing.ARMS if args.full else ('immediate',)
    output.mkdir(parents=True)
    timing.write_json(output / 'integrity_before.json', hashes)
    manifest = dict(kind='offline_competing_U_mechanistic_instrumentation',
        mode='full' if args.full else 'validation_smoke', updates=updates, seeds=seeds,
        timing_arms=arms, conditions=CONDITIONS, device='cpu', torch_threads=1,
        decomposition_atol=DECOMP_ATOL, decomposition_rtol=DECOMP_RTOL,
        optimizer_intervention='reset exp_avg/exp_avg_sq once at start; preserve step and hyperparameters',
        interpretation='No causal conclusions. Head/representation decomposition is algebraic; paired paths diverge.',
        historical_replay='reconstruct identically from saved states/traces; validate original replay fingerprint',
        labels='post-hoc observation only; unchanged DQNTrainer._update receives Transitions',
        starting_common_target=True, target_sync_in_horizon=False,
        production_source_modified=False, environment_rollout=False, planner_labeling=False,
        input_schedule_fingerprints={}, implementation_hashes={
            'script': timing.file_hash(Path(__file__)),
            'test': timing.file_hash(ROOT / 'tests/test_u_competing_actions_diagnostic.py')})
    all_summaries = []
    try:
        for seed in seeds:
            data, plan = load_saved(seed)
            manifest['input_schedule_fingerprints'][str(seed)] = timing.fingerprint(plan.payload())
            for arm in arms:
                branch_started = time.perf_counter()
                initial, rows, actions, summaries, comparisons = run_pair(data, plan, arm, updates)
                prefix = f'seed{seed}_{arm}'
                timing.write_json(output / f'{prefix}_initial.json', initial)
                write_jsonl(output / f'{prefix}_updates.jsonl', rows)
                write_jsonl(output / f'{prefix}_actions.jsonl', actions)
                write_jsonl(output / f'{prefix}_adam_comparison.jsonl', comparisons)
                all_summaries.extend(summaries)
                manifest.setdefault('branch_elapsed_seconds', {})[prefix] = time.perf_counter() - branch_started
        integrity = verify_snapshot(hashes)
        assert guard.denied == 0
        elapsed = time.perf_counter() - started
        max_residual = max(s['max_abs_decomposition_residual'] for s in all_summaries)
        # Scaling includes paired updates + instrumentation, not a full-run measurement.
        branch_elapsed = sum(manifest['branch_elapsed_seconds'].values())
        estimated = branch_elapsed / (len(seeds) * len(arms) * updates) * 4 * timing.H
        validation = dict(status='PASS', mode=manifest['mode'], paired_schedule_equality=True,
            exposure_count_equality=True, decomposition_checks='every recorded action/update',
            max_abs_decomposition_residual=max_residual, integrity=integrity,
            run_access_guard=guard.evidence(), elapsed_seconds=elapsed,
            full_runtime_linear_estimate_seconds=estimated,
            runtime_estimate_note='short smoke extrapolation; host load and longer-run I/O may vary',
            full_experiment_executed=args.full, causal_conclusion=None)
        timing.write_json(output / 'manifest.json', manifest)
        timing.write_json(output / 'summary.json', all_summaries)
        timing.write_json(output / 'validation.json', validation)
        print(timing.canonical(validation), flush=True)
    except BaseException as exc:
        timing.write_json(output / 'failure.json', dict(error=f'{type(exc).__name__}: {exc}',
                          run_access_guard=guard.evidence()))
        raise
    finally:
        guard.enabled = False


if __name__ == '__main__':
    main()
