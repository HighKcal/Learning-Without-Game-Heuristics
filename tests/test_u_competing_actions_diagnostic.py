"""Short deterministic validation only; no rollout or active training access."""
import importlib.util
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    'u_competing_diagnostic', ROOT / 'scripts/diagnose_u_competing_actions.py')
diag = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = diag
spec.loader.exec_module(diag)
t = diag.timing


@pytest.fixture(scope='module')
def historical():
    t.deterministic()
    guard = diag.RunAccessGuard()
    sys.addaudithook(guard)
    hashes = diag.protected_snapshot()
    try:
        yield diag.load_saved(42)
    finally:
        assert diag.verify_snapshot(hashes)['status'] == 'PASS'
        assert guard.denied == 0
        assert set(guard.reads) <= guard.allowed
        guard.enabled = False


def test_exact_decomposition_including_cross_term():
    rng = torch.Generator().manual_seed(123)
    h0, h1 = [torch.randn(256, generator=rng, dtype=torch.double) for _ in range(2)]
    w0, w1 = [torch.randn(800, 256, generator=rng, dtype=torch.double) for _ in range(2)]
    b0, b1 = [torch.randn(800, generator=rng, dtype=torch.double) for _ in range(2)]
    before = diag.Probe(h0, w0, b0, (w0 @ h0 + b0).numpy())
    after = diag.Probe(h1, w1, b1, (w1 @ h1 + b1).numpy())
    for action in (701, 705, 0, 321, 799):
        row = diag.decompose(before, after, action)
        assert row['head_contribution'] + row['representation_contribution'] == pytest.approx(
            row['delta_q'], abs=1e-12)
        assert abs(row['affine_identity_residual']) < 1e-12


def test_decomposition_bad_observed_q_rejected():
    before = diag.Probe(torch.ones(2, dtype=torch.double), torch.ones(800, 2, dtype=torch.double),
                        torch.zeros(800, dtype=torch.double), np.full(800, 2.))
    after = replace(before, q=np.full(800, 3.))
    with pytest.raises(AssertionError):
        diag.decompose(before, after, 705)


def test_restore_weights_targets_counters_hyperparameters_steps_and_moments(historical):
    data, plan = historical
    checkpoint_hash = t.tensor_tree_fingerprint(data.checkpoint)
    restored, zero = diag.make_conditions(data, plan)
    t.assert_equal(restored.online.state_dict(), zero.online.state_dict())
    t.assert_equal(restored.target.state_dict(), zero.target.state_dict())
    t.assert_equal(restored.optimizer.state_dict(), data.checkpoint['optimizer_state_dict'])
    a, b = restored.optimizer.state_dict(), zero.optimizer.state_dict()
    t.assert_equal(a['param_groups'], b['param_groups'])
    assert any(torch.count_nonzero(s['exp_avg']) for s in a['state'].values())
    for key, state in b['state'].items():
        assert not torch.count_nonzero(state['exp_avg'])
        assert not torch.count_nonzero(state['exp_avg_sq'])
        t.assert_equal(state['step'], a['state'][key]['step'])
        assert int(state['step']) == data.checkpoint['gradient_updates']
    assert [getattr(restored, name) for name in diag.COUNTERS] == [getattr(zero, name) for name in diag.COUNTERS]
    assert t.tensor_tree_fingerprint(data.checkpoint) == checkpoint_hash


def test_saved_schedule_and_reconstructed_replay_not_regenerated(historical, monkeypatch):
    def prohibited(*args, **kwargs):
        raise AssertionError('Cannot regenerate the saved schedule')
    monkeypatch.setattr(t, 'make_plan', prohibited)
    data, plan = diag.load_saved(42)
    assert plan.payload() == historical[1].payload()
    assert len(plan.batches) == 1024 and plan.delay == 128
    assert len(data.background) == 8026 and data.current.action_id == 701
    expected = t.read_json(diag.SOURCE / 'manifest.json')['seeds']['42']
    assert t.fingerprint(plan.payload()) == expected['schedule_fingerprint']
    assert t.fingerprint([t.transition_identity(x) for x in data.background]) == expected['starting_replay_fingerprint']


def test_seed123_saved_inputs_and_schedule():
    # A second historical checkpoint read, not a learner experiment.
    data, plan = diag.load_saved(123)
    assert len(data.background) == 9000 and data.current.action_id == 528
    assert len(plan.batches) == 1024


def test_action_id_exposure_distinct_from_probe_pair(historical):
    data, _ = historical
    other = data.background[0].copy()
    assert t.state_hash(other.state) != data.record.state_hash
    elsewhere705 = replace(other, action_id=705)
    probe701 = data.current.copy()
    probe705 = replace(probe701, action_id=705)
    audit = diag.exposure([elsewhere705, elsewhere705.copy(), probe701], data.record.state_hash, (701, 705))
    assert audit[705] == dict(action_id_count=2, probe_state_count=1, same_state_same_action_count=0)
    assert audit[701]['same_state_same_action_count'] == 1
    assert diag.exposure([probe705], data.record.state_hash, (705,))[705]['same_state_same_action_count'] == 1


def test_actual_four_update_pair_decomposition_exposure_and_schedule(historical):
    data, plan = historical
    initial, rows, actions, summaries, comparisons = diag.run_pair(data, plan, 'immediate', 4)
    assert initial['topU_id'] == 701
    assert len(rows) == 8 and len(summaries) == 2 and len(comparisons) == 8
    for update in range(1, 5):
        a, b = [r for r in rows if r['local_update'] == update]
        assert a['batch_occurrence_indices'] == b['batch_occurrence_indices'] == plan.batch(update, 'immediate')
        assert a['batch_schedule_id'] == b['batch_schedule_id']
        for condition in diag.CONDITIONS:
            records = [r for r in actions if r['condition'] == condition and r['local_update'] == update]
            assert {701, 705, a['pre']['topU_id'] if condition == a['condition'] else b['pre']['topU_id']} <= {r['action_id'] for r in records}
            for record in records:
                assert record['head_contribution'] + record['representation_contribution'] == pytest.approx(
                    record['delta_q'], abs=diag.DECOMP_ATOL + diag.DECOMP_RTOL * abs(record['delta_q']))
                if record['action_id_count'] == 0:
                    assert record['head_gradient_zero']
        for action in (701, 705):
            pair = [r for r in actions if r['local_update'] == update and r['action_id'] == action]
            for key in ('action_id_count', 'probe_state_count', 'same_state_same_action_count'):
                assert pair[0][key] == pair[1][key]
    first_zero = next(r for r in actions if r['condition'] == 'zero_momentum_adam'
                      and r['local_update'] == 1 and r['action_id'] == 705)
    if first_zero['action_id_count'] == 0:
        assert first_zero['head_weight_delta_norm'] == first_zero['head_bias_delta'] == 0
    assert all(r['absolute_gradient_update'] == data.checkpoint['gradient_updates'] + r['local_update'] for r in rows)
    assert max(abs(a['decomposition_residual']) for a in actions) < diag.DECOMP_ATOL


def test_delayed_short_prefix_uses_exact_saved_batch(historical):
    data, plan = historical
    _, rows, actions, _, _ = diag.run_pair(data, plan, 'delayed', 1)
    assert all(r['batch_occurrence_indices'] == plan.batch(1, 'delayed') for r in rows)
    assert all(a['probe_state_count'] == 0 for a in actions)


def test_adam_zero_gradient_row_moves_only_with_prior_moment(historical):
    data, plan = historical
    restored, zero = diag.make_conditions(data, plan)
    # Controlled unit mechanism: select only original701 terminal transitions.
    # Probe state705 is never selected, so its head gradient is exactly zero.
    with torch.no_grad():
        for parameter in (restored.online[-1].weight, restored.online[-1].bias):
            restored.optimizer.state[parameter]['exp_avg'][705].fill_(.01)
            restored.optimizer.state[parameter]['exp_avg_sq'][705].fill_(.02)
    before = [diag.capture_probe(x, data.current.state) for x in (restored, zero)]
    batch = [replace(data.current, terminated=True) for _ in range(128)]
    for trainer in (restored, zero):
        trainer._update([x.copy() for x in batch], trainer.epsilon)
        assert not torch.count_nonzero(trainer.online[-1].weight.grad[705])
        assert trainer.online[-1].bias.grad[705] == 0
    after = [diag.capture_probe(x, data.current.state) for x in (restored, zero)]
    assert not torch.equal(before[0].w[705], after[0].w[705])
    assert before[0].b[705] != after[0].b[705]
    assert torch.equal(before[1].w[705], after[1].w[705])
    assert before[1].b[705] == after[1].b[705]


def test_zero_momentum_is_reset_once_then_accumulates_history(historical):
    data, plan = historical
    _, zero = diag.make_conditions(data, plan)
    zero._update([replace(data.current, action_id=705, terminated=True) for _ in range(128)], zero.epsilon)
    before = diag.capture_probe(zero, data.current.state)
    assert torch.count_nonzero(zero.optimizer.state[zero.online[-1].weight]['exp_avg'][705])
    zero._update([replace(data.current, terminated=True) for _ in range(128)], zero.epsilon)
    after = diag.capture_probe(zero, data.current.state)
    assert not torch.count_nonzero(zero.online[-1].weight.grad[705])
    assert not torch.equal(before.w[705], after.w[705])


def test_labels_do_not_enter_learner(historical, monkeypatch):
    data, plan = historical
    alternate_record = t.EvaluationRecord(data.record.state, data.record.valid_action_ids,
                                          (data.current.action_id,))
    changed = replace(data, record=alternate_record)
    a, b = [diag.make_conditions(data, plan)[0] for _ in range(2)]
    received = []
    real = t.DQNTrainer._update
    def observe(self, batch, epsilon):
        assert all(isinstance(x, t.Transition) for x in batch)
        assert all(not hasattr(x, 'reachable_action_ids') for x in batch)
        received.append([t.transition_identity(x) for x in batch])
        return real(self, batch, epsilon)
    monkeypatch.setattr(t.DQNTrainer, '_update', observe)
    for update in (1, 2):
        diag.instrumented_update(a, data, plan, update, 'immediate', 'restored_adam')
        diag.instrumented_update(b, changed, plan, update, 'immediate', 'restored_adam')
        t.trainer_equal(a, b)
    assert received[0] == received[1] and received[2] == received[3]


def test_ranking_ties_match_existing_evaluator(historical):
    from tetris_agent.rl.evaluation import evaluate_network
    data, plan = historical
    trainer = diag.make_conditions(data, plan)[0]
    with torch.no_grad():
        for parameter in trainer.online.parameters():
            parameter.zero_()
        invalid = next(a for a in range(800) if a not in data.record.valid_action_ids)
        trainer.online[-1].bias[invalid] = 1000
    q = diag.capture_probe(trainer, data.current.state).q
    rank = diag.ranked_probe(q, data.record)
    rows = evaluate_network(trainer.online, [data.record])['states']
    assert rank['greedy_id'] == rows[0]['action_id'] == min(data.record.valid_action_ids)
    assert rank['maxR'] == rows[0]['max_reachable_q'] and rank['maxU'] == rows[0]['max_unreachable_q']
    assert rank['topR_id'] == min(data.record.reachable_action_ids)
    assert rank['topU_id'] == min(set(data.record.valid_action_ids) - set(data.record.reachable_action_ids))
    assert rank['top5U_ids'] == sorted(set(data.record.valid_action_ids) - set(data.record.reachable_action_ids))[:5]
    assert rank['top5R_ids'] == sorted(data.record.reachable_action_ids)[:5]


def test_churn_duration_and_initial_switch_convention():
    initial = dict(topU_id=701)
    rows = [dict(post=dict(topU_id=a)) for a in (701, 705, 705, 706, 701, 701)]
    result = diag.churn(initial, rows, 701)
    assert result['topU_identity_switch_count'] == 3
    assert result['unique_topU_action_count'] == 3
    assert result['longest_same_topU_duration'] == 2
    assert result['original_topU_updates'] == 3 and result['competing_topU_updates'] == 3
    assert result['topU_duration_by_action'] == {'701': 3, '705': 2, '706': 1}
    assert diag.churn(initial, [dict(post=dict(topU_id=705))], 701)['topU_identity_switch_count'] == 1


@pytest.mark.parametrize('event,mode,flags', [('open', 'r', 0), ('open', 'w', 1),
                                           ('os.scandir', None, None), ('os.listdir', None, None)])
def test_active_run_guard_denies_without_accessing_directory(event, mode, flags):
    guard = diag.RunAccessGuard()
    # Only strings passed to guard; this path is never opened/stat'ed/traversed.
    hypothetical = ROOT / 'runs/forbidden_active_run/checkpoint_final.pt'
    with pytest.raises(PermissionError):
        guard(event, (str(hypothetical), mode, flags))
    assert guard.denied == 1 and not guard.reads


def test_guard_allows_only_fixed_checkpoint_reads_and_manifest_rejects_other_runs():
    guard = diag.RunAccessGuard()
    cp = diag.allowed_checkpoint_paths()[0]
    guard('open', (str(cp), 'r', 0))
    with pytest.raises(PermissionError):
        guard('open', (str(cp), 'w', 1))
    with pytest.raises(ValueError):
        diag.safe_input_path('runs/forbidden_active_run/checkpoint_final.pt')
    with pytest.raises(ValueError):
        diag.safe_input_path('benchmarks/ddqn_frozen_policy_gameplay_20261005/../../runs/other/file')


def test_smoke_limits_and_full_opt_in():
    assert not diag.parse_args([]).full
    assert diag.parse_args(['--full']).full
    for args in (['--updates', '1024'], ['--updates', '0'], ['--full', '--updates', '4']):
        with pytest.raises(SystemExit):
            diag.parse_args(args)


def test_integrity_detects_historical_change(tmp_path):
    # Validate checker against a temporary file, not a historical artifact write.
    path = tmp_path / 'artifact'
    path.write_text('original')
    digest = t.file_hash(path)
    assert diag.verify_snapshot({str(path): digest})['status'] == 'PASS'
    path.write_text('changed')
    with pytest.raises(AssertionError):
        diag.verify_snapshot({str(path): digest})
