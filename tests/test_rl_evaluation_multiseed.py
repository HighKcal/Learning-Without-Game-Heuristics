"""Exercise orchestration and duplicate continuation through the real build loop.

Synthetic candidate streams isolate quota logic; existing tests cover physics.
"""
from copy import deepcopy
from dataclasses import replace
import json
from types import SimpleNamespace

import pytest

from tetris_agent.rl import evaluation as ev
from tetris_agent.rl import evaluation_build as build
from test_rl_evaluation import load_cli, record


def candidate(number):
    queue = tuple((number // 7**i) % 7 for i in range(4))
    return record(queue=queue)


def synthetic_streams(monkeypatch, streams):
    starts, advances, labels = [], [], []
    class Env:
        def __init__(self, **kwargs): pass
        def reset(self, *, seed):
            starts.append(seed)
            self.seed, self.index = seed, 0
        def close(self): pass
    def current(env): return candidate(streams[env.seed][env.index])
    def label(env, planner, *, timings, deadline, counts):
        row = current(env)
        labels.append((env.seed, env.index))
        timings.extend([.01] * len(row.valid_action_ids))
        counts['reachable'] += len(row.valid_action_ids)
        return row
    def plan(env, target, planner, timings):
        assert build.SPACE.encode(target) in current(env).reachable_action_ids
        timings.append(.001)
        return SimpleNamespace(status=build.PlannerStatus.REACHABLE)
    class Executor:
        def execute(self, env, target, result):
            advances.append((env.seed, env.index))
            env.index += 1
            return SimpleNamespace(success=True, target_verified=True, terminated=False, truncated=False)
    monkeypatch.setattr(build, 'Tetris', Env)
    monkeypatch.setattr(build, 'Executor', Executor)
    monkeypatch.setattr(build, 'read_state', lambda env: current(env).state)
    monkeypatch.setattr(build, 'assert_canonical', lambda env: None)
    monkeypatch.setattr(build, 'native_context', lambda env: {})
    monkeypatch.setattr(build, 'label_candidate', label)
    monkeypatch.setattr(build, 'checked_plan', plan)
    return starts, advances, labels


def test_global_duplicates_continue_same_seed_until_own_quota(monkeypatch, tmp_path):
    starts, advances, labels = synthetic_streams(monkeypatch, {7: [0, 1], 42: [0, 2, 3], 123: [2, 4, 5]})
    rows, meta = build.build_multiseed_dataset([7, 42, 123], states_per_seed=2)
    assert starts == [7, 42, 123]
    assert [r.state_hash for r in rows] == [candidate(i).state_hash for i in range(6)]
    assert [r.metadata['generation_seed'] for r in rows] == [7, 7, 42, 42, 123, 123]
    assert [m['accepted_state_count'] for m in meta['per_seed']] == [2, 2, 2]
    assert [m['build_statistics']['candidate_states_visited'] for m in meta['per_seed']] == [2, 3, 3]
    assert [m['build_statistics']['rejected_duplicates'] for m in meta['per_seed']] == [0, 1, 1]
    assert meta['global_duplicate_count'] == 2 and meta['status'] == 'completed'
    assert meta['build_statistics']['rejected_candidates'] == 0
    assert len(advances) == 5 and len(labels) == 8
    assert (42, 0) in advances and (123, 0) in advances
    assert meta['aggregate_label_statistics'] == ev.label_statistics(rows)
    ev.save_dataset(tmp_path / 'final', rows, meta)
    loaded, restored = ev.load_dataset(tmp_path / 'final')
    assert [r.to_dict() for r in loaded] == [r.to_dict() for r in rows]
    assert restored['dataset_fingerprint'] == ev.dataset_fingerprint(rows)


def test_256_globally_unique_order_and_deterministic_reproduction(monkeypatch, tmp_path):
    seeds = [456, 7, 42, 123]  # Preserve supplied order, do not sort silently.
    streams = {seed: list(range(i * 64, (i + 1) * 64)) for i, seed in enumerate(seeds)}
    starts, _, _ = synthetic_streams(monkeypatch, streams)
    rows, meta = build.build_multiseed_dataset(seeds)
    again, repeat = build.build_multiseed_dataset(seeds)
    assert starts == seeds * 2
    assert len(rows) == len({r.state_hash for r in rows}) == 256
    assert meta['generation_seeds'] == seeds
    assert meta['total_requested_states'] == meta['total_accepted_states'] == 256
    assert [m['generation_seed'] for m in meta['per_seed']] == seeds
    assert [m['accepted_state_count'] for m in meta['per_seed']] == [64] * 4
    assert [r.metadata['generation_seed'] for r in rows] == [s for s in seeds for _ in range(64)]
    assert [r.to_dict() for r in rows] == [r.to_dict() for r in again]
    assert meta['dataset_fingerprint'] == repeat['dataset_fingerprint']
    ev.save_dataset(tmp_path / 'final256', rows, meta)
    loaded, saved = ev.load_dataset(tmp_path / 'final256')
    assert len({r.state_hash for r in loaded}) == 256
    assert saved['aggregate_label_statistics'] == ev.label_statistics(loaded)


def test_incomplete_seed_stops_without_borrowing_quota(monkeypatch, tmp_path):
    starts, _, _ = synthetic_streams(monkeypatch, {7: [0, 1], 42: [0, 1, 2], 123: [3, 4]})
    rows, meta = build.build_multiseed_dataset([7, 42, 123], states_per_seed=2, max_candidates=2)
    assert starts == [7, 42] and len(rows) == 2
    assert meta['status'] == 'incomplete'
    assert [m['accepted_state_count'] for m in meta['per_seed']] == [2, 0]
    assert meta['diagnostics'][-1]['reason'] == 'CANDIDATE_LIMIT'
    assert meta['diagnostics'][-1]['generation_seed'] == 42
    ev.save_dataset(tmp_path / 'partial', rows, meta)
    assert ev.load_dataset(tmp_path / 'partial')[1]['status'] == 'incomplete'
    bad = dict(meta, status='completed')
    with pytest.raises(ValueError, match='Incomplete seed quota'):
        ev.save_dataset(tmp_path / 'false_success', rows, bad)
    with pytest.raises(SystemExit):
        load_cli('evaluate_policy').main(['--dataset', str(tmp_path / 'partial'), '--untrained-seed', '42'])


def test_single_seed_default_and_explicit_empty_exclusions_unchanged(monkeypatch):
    synthetic_streams(monkeypatch, {42: [0, 1]})
    rows, meta = build.build_dataset(build.BuildConfig(state_count=2))
    again, _ = build.build_dataset(build.BuildConfig(state_count=2), excluded_hashes=set())
    assert [r.to_dict() for r in rows] == [r.to_dict() for r in again]
    assert meta['generation_seed'] == 42 and 'generation_seeds' not in meta
    assert all('generation_seed' not in r.metadata for r in rows)
    assert meta['build_statistics']['rejected_duplicates'] == 0


def test_fingerprint_ignores_metadata_but_binds_labels_and_record_order():
    rows = [candidate(0), candidate(1)]
    changed_metadata = [replace(r, metadata={'seed': 99, 'timestamp': 'later'}) for r in rows]
    assert ev.dataset_fingerprint(rows) == ev.dataset_fingerprint(changed_metadata)
    assert ev.dataset_fingerprint(rows) != ev.dataset_fingerprint(rows[::-1])
    changed_labels = [replace(rows[0], reachable_action_ids=rows[0].reachable_action_ids[:-1]), rows[1]]
    assert ev.dataset_fingerprint(rows) != ev.dataset_fingerprint(changed_labels)
    reordered_ids = [replace(r, valid_action_ids=r.valid_action_ids[::-1],
                             reachable_action_ids=r.reachable_action_ids[::-1]) for r in rows]
    assert ev.dataset_fingerprint(rows) == ev.dataset_fingerprint(reordered_ids)


@pytest.mark.parametrize('corruption', ['global_duplicate', 'seed_order', 'contribution',
    'total_count', 'fingerprint', 'record_seed', 'duplicate_count'])
def test_final_loader_rejects_corruption(monkeypatch, tmp_path, corruption):
    synthetic_streams(monkeypatch, {7: [0, 1], 42: [2, 3]})
    rows, meta = build.build_multiseed_dataset([7, 42], states_per_seed=2)
    path = tmp_path / 'final'
    ev.save_dataset(path, rows, meta)
    lines = [r.to_dict() for r in rows]
    if corruption == 'global_duplicate': lines[2] = deepcopy(lines[0])
    elif corruption == 'seed_order': meta['generation_seeds'].reverse()
    elif corruption == 'contribution': meta['per_seed'][0]['accepted_state_count'] = 1
    elif corruption == 'total_count': meta['total_accepted_states'] = 3
    elif corruption == 'fingerprint': meta['dataset_fingerprint'] = 'wrong'
    elif corruption == 'record_seed': lines[0]['metadata']['generation_seed'] = 42
    elif corruption == 'duplicate_count': meta['global_duplicate_count'] = 1
    (path / 'metadata.json').write_text(json.dumps(meta))
    (path / 'states.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in lines))
    with pytest.raises(ValueError): ev.load_dataset(path)


@pytest.mark.parametrize('seeds,quota', [([], 2), ([7, 7], 2), ([-1], 2), ([7], 0), ([True], 2)])
def test_multiseed_input_validation(seeds, quota):
    with pytest.raises(ValueError): build.build_multiseed_dataset(seeds, states_per_seed=quota)


@pytest.mark.parametrize('flags', [['--seeds', '7', '42', '--states', '2'],
    ['--seed', '7', '--seeds', '42'], ['--states-per-seed', '2']])
def test_cli_refuses_ambiguous_modes(tmp_path, flags):
    with pytest.raises(SystemExit):
        load_cli('build_fixed_eval').main([*flags, '--output-dir', str(tmp_path / 'unused')])


def test_cli_single_and_multi_modes(monkeypatch, tmp_path, capsys):
    synthetic_streams(monkeypatch, {42: list(range(32)), 7: [100, 101]})
    cli = load_cli('build_fixed_eval')
    assert cli.main(['--output-dir', str(tmp_path / 'single')]) == 0
    assert len(ev.load_dataset(tmp_path / 'single')[0]) == 32
    assert cli.main(['--seeds', '7', '42', '--states-per-seed', '2',
                     '--output-dir', str(tmp_path / 'multi')]) == 0
    rows, meta = ev.load_dataset(tmp_path / 'multi')
    assert len(rows) == 4 and meta['generation_seeds'] == [7, 42]


def test_real_multiseed_small_smoke_matches_single_seed():
    # Real production Planner/Executor path, only one candidate per seed.
    rows, meta = build.build_multiseed_dataset([7, 42], states_per_seed=1)
    assert meta['status'] == 'completed'
    for seed, row in zip([7, 42], rows):
        single, _ = build.build_dataset(build.BuildConfig(seed=seed, state_count=1))
        assert row.state_hash == single[0].state_hash
        assert row.valid_action_ids == single[0].valid_action_ids
        assert row.reachable_action_ids == single[0].reachable_action_ids
