"""Focused offline tests: no env, planner, rollout, or source artifact writes."""
import importlib.util
import sys
from collections import Counter
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('u_feedback_diagnostic', ROOT / 'scripts/diagnose_u_feedback_timing.py')
diag = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = diag
spec.loader.exec_module(diag)


@pytest.fixture(scope='module')
def data():
    diag.deterministic()
    loaded = diag.load_inputs(42)
    # Small actual-artifact subset for focused mechanics; main A/A uses all inputs/H1024.
    return replace(loaded, background=loaded.background[:200], occurrence_ids=loaded.occurrence_ids[:200])


def test_checkpoint_online_target_adam_restore_equality(data):
    a = diag.restore(data.checkpoint, data.background, 17)
    diag.assert_equal(a.online.state_dict(), data.checkpoint['online_state_dict'])
    diag.assert_equal(a.target.state_dict(), data.checkpoint['target_state_dict'])
    diag.assert_equal(a.optimizer.state_dict(), data.checkpoint['optimizer_state_dict'])
    assert a.gradient_updates == 95001 and a.target_sync_count == 19


def test_branch_parameter_optimizer_and_replay_independence(data):
    a, b = [diag.restore(data.checkpoint, data.background, 17) for _ in range(2)]
    diag.trainer_equal(a, b)
    diag.independent(a, b)
    param = next(a.online.parameters())
    with torch.no_grad():
        param.add_(.01)
        a.optimizer.state[param]['exp_avg'].add_(.02)
    assert not torch.equal(param, next(b.online.parameters()))
    assert not torch.equal(a.optimizer.state[param]['exp_avg'], b.optimizer.state[next(b.online.parameters())]['exp_avg'])


def test_ordered_schedule_swap_exposure_batch_size(data):
    plan = diag.make_plan(data.occurrence_ids, 12345)
    assert plan.payload() == diag.make_plan(data.occurrence_ids, 12345).payload()
    assert diag.fingerprint(plan.payload()) == diag.fingerprint(diag.make_plan(data.occurrence_ids, 12345).payload())
    counters = []
    for arm in diag.ARMS:
        rows = [plan.batch(k, arm) for k in range(1, 1025)]
        assert all(len(row) == 128 and len(set(row)) == 128 for row in rows)
        assert [k for k, row in enumerate(rows, 1) if plan.current in row] == [1 if arm == 'immediate' else 129]
        counters.append(Counter(i for row in rows for i in row))
    assert counters[0] == counters[1]
    assert plan.batch(1, 'immediate')[-1] == plan.current
    assert plan.batch(1, 'delayed')[-1] == plan.x
    assert plan.batch(129, 'immediate')[-1] == plan.x
    assert plan.batch(129, 'delayed')[-1] == plan.current
    for k in (1, 129):
        assert plan.batch(k, 'immediate')[:-1] == plan.batch(k, 'delayed')[:-1]
        assert plan.x not in plan.batch(k, 'immediate')[:-1]
    for k in range(1, 1025):
        if k not in (1, 129):
            assert plan.batch(k, 'immediate') == plan.batch(k, 'delayed')


def test_schedule_invalid_extra_u_exposure_rejected(data):
    plan = diag.make_plan(data.occurrence_ids, 5)
    plan.batches[1][0] = plan.current
    with pytest.raises(AssertionError):
        diag.validate_plan(plan)


def test_aa_exact_each_update_and_counter_phase(data):
    plan = diag.make_plan(data.occurrence_ids, 99, horizon=8, delay=2)
    result = diag.aa_validate(data, plan)
    assert result['status'] == 'PASS' and result['checked_updates'] == 8
    a = diag.restore(data.checkpoint, data.background, plan.seed)
    for k in range(1, 9):
        row = diag.step(a, data, plan, k, 'immediate')
        assert row['absolute_gradient_update'] == 95001 + k
        assert row['sync_phase'] == (95001 + k) % 5000
        assert not row['target_sync_happened']
        assert a.stored_experiences == data.checkpoint['stored_experiences'] + k


def test_aa_mismatch_stops(monkeypatch, data):
    plan = diag.make_plan(data.occurrence_ids, 99, horizon=3, delay=1)
    real = diag.step
    calls = []
    def corrupted(*args, **kwargs):
        row = real(*args, **kwargs)
        calls.append(1)
        if len(calls) == 2:
            row['margin'] += 1e-9
        return row
    monkeypatch.setattr(diag, 'step', corrupted)
    with pytest.raises(AssertionError):
        diag.aa_validate(data, plan)


def test_actual_pair_requires_both_full_aa_gates(data):
    plan = diag.make_plan(data.occurrence_ids, 99)
    gates = dict(focused_tests='PASS', regression_tests='PASS',
                 aa=[dict(seed=42, status='PASS', checked_updates=1024)])
    with pytest.raises(AssertionError):
        diag.run_pair(data, plan, gates)


def test_reward_reconstruction_preserves_components_and_truncation(data):
    state = data.record.state
    h = diag.state_hash(state)
    states = {h: state}
    row = dict(state_hash=h, next_state_hash=h, action_id=data.current.action_id,
               unreachable=True, reward=-.05, terminated=False, truncated=True)
    t = diag.reconstruct(row, states, -.20)
    assert t.reward == -.20 and t.truncated and not t.terminated
    row.update(unreachable=False, reward=2., terminated=True, truncated=False)
    t = diag.reconstruct(row, states, -.20)
    assert t.reward == 2. and t.terminated  # three lines minus death penalty.
    row.update(unreachable=True, reward=-.05)
    with pytest.raises(AssertionError):
        diag.reconstruct(row, states, -.20)


@pytest.mark.parametrize('seed,expected,missing', [(42, 8026, 1), (123, 9000, 0)])
def test_background_occurrence_multiplicity_source_integrity(seed, expected, missing):
    data = diag.load_inputs(seed)
    before = {p: diag.file_hash(p) for p in data.paths}
    assert len(data.background) == expected
    assert len(data.metadata['excluded']) == missing
    assert len(set(data.occurrence_ids)) == expected
    assert all(diag.state_hash(t.state) != data.record.state_hash for t in data.background)
    semantic_keys = [(diag.state_hash(t.state), t.action_id, diag.state_hash(t.next_state), t.reward) for t in data.background]
    assert len(set(semantic_keys)) < len(semantic_keys)  # repeats deliberately retained.
    assert data.current.reward == -.20
    assert {p: diag.file_hash(p) for p in data.paths} == before
    if missing:
        excluded = data.metadata['excluded'][0]
        assert excluded['trace_row']['terminated'] and 'terminal next' in excluded['reason']


def test_ranking_tie_membership_and_tolerance(data):
    record = data.record
    valid = sorted(record.valid_action_ids)
    reachable = set(record.reachable_action_ids)
    q = np.zeros(800)
    q[next(a for a in range(800) if a not in valid)] = 1000.
    row = diag.ranking(q, record)
    assert row['greedy_id'] == valid[0]
    assert row['greedy_class'] == ('R' if valid[0] in reachable else 'U')
    assert row['topR_id'] == min(reachable)
    assert row['topU_id'] == min(set(valid) - reachable)
    assert row['margin'] == 0 and not row['corrected']
    q[row['topR_id']] = 2e-6
    assert diag.ranking(q, record)['corrected']


def test_labels_do_not_enter_bootstrap_or_learner_masks(data):
    a = diag.restore(data.checkpoint, data.background, 7)
    mask = diag.geometric_mask(data.current.state)
    with torch.no_grad():
        for p in a.online.parameters():
            p.zero_()
        for p in a.target.parameters():
            p.zero_()
        a.online[-1].bias[data.current.action_id] = 2.
        a.target[-1].bias[data.current.action_id] = 3.
    probe, q = diag.current_probe(a, data.current)
    assert probe['bootstrap_selected_id'] == data.current.action_id  # U remains selectable.
    assert probe['target_q_bootstrap'] == 3.
    assert probe['bellman_target_u'] == pytest.approx(-.2 + .99 * 3, abs=1e-6)
    assert probe['signed_td_u'] == pytest.approx(-.2 + .99 * 3 - 2, abs=1e-6)
    assert mask[data.current.action_id]
    alternate = diag.EvaluationRecord(data.record.state, data.record.valid_action_ids,
                                     (data.current.action_id,))
    assert diag.ranking(q, data.record)['greedy_class'] == 'U'
    assert diag.ranking(q, alternate)['greedy_class'] == 'R'
    changed = replace(data, record=alternate)
    p = diag.make_plan(data.occurrence_ids, 7, horizon=3, delay=1)
    b = diag.restore(data.checkpoint, data.background, 7)
    c = diag.restore(data.checkpoint, data.background, 7)
    diag.step(b, data, p, 1, 'immediate')
    diag.step(c, changed, p, 1, 'immediate')
    diag.trainer_equal(b, c)  # Label changes only change metrics.


def test_sustained_correction_and_censoring():
    def rows(flags):
        return [dict(corrected=f, margin=1. if f else -1., greedy_class='R' if f else 'U') for f in flags]
    flags = [False] * 2 + [True] * 127 + [False] + [True] * 128
    s = diag.summarize(rows(flags))
    assert s['first_robust_correction_update'] == 3
    assert s['sustained_correction_onset'] == 131
    assert s['sustained_correction_confirmation_update'] == 258
    assert s['reversion_count'] == 1 and s['total_uncorrected_updates'] == 3
    assert not s['censored']
    s = diag.summarize(rows([False] + [True] * 127))
    assert s['sustained_correction_onset'] is None and s['censored']


def test_integrity_checker_detects_change(tmp_path, monkeypatch):
    monkeypatch.setattr(diag, 'ROOT', tmp_path)
    file = tmp_path / 'source'
    file.write_text('original')
    hashes = {'source': diag.file_hash(file)}
    assert diag.verify_integrity(hashes)['status'] == 'PASS'
    file.write_text('changed')
    with pytest.raises(AssertionError):
        diag.verify_integrity(hashes)
