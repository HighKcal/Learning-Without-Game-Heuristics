"""Fixed exam contracts: real planning on build, labels isolated during inference."""
from copy import deepcopy
from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from tetris_gymnasium.envs.tetris import Tetris

from tetris_agent import Executor, PlannerResult, PlannerStatus, TacticalPlanner, TargetPlacement
from tetris_agent.rl import evaluation as ev
from tetris_agent.rl import evaluation_build as build
from tetris_agent.rl.artifacts import ArtifactConfig, RunArtifacts
from tetris_agent.rl.config import TrainingConfig
from tetris_agent.rl.dqn import DQNTrainer
from tetris_agent.rl.network import QNetwork
from tetris_agent.rl.replay import Transition
from tetris_agent.rl.state import SemanticState, SPACE, assert_canonical, geometric_mask, read_state
from test_planner import select_piece, state_signature


def record(piece=1, reachable_count=None, queue=(0, 2, 4, 6)):
    state = SemanticState(np.zeros((20, 10), dtype=np.uint8), piece, queue)
    valid = tuple(np.flatnonzero(geometric_mask(state)).tolist())
    return ev.EvaluationRecord(state, valid, valid if reachable_count is None else valid[:reachable_count])


def metadata(count=1):
    return dict(git_commit='test', tetris_gymnasium_version='0.3.1', generation_seed=42,
                requested_state_count=count, rejected_state_count=0, generation_configuration={},
                status='completed')


def constant_network(values=None):
    network = QNetwork()
    with torch.no_grad():
        for parameter in network.parameters():
            parameter.zero_()
        if values:
            for action, value in values.items():
                network[-1].bias[action] = value
    return network


@pytest.fixture
def env():
    game = Tetris(gravity=True)
    game.reset(seed=42)
    yield game
    game.close()


def test_semantic_serialization_hash_and_metadata_independence():
    state = record().state
    board = state.board.copy()
    board[7, 4] = 1
    variants = [SemanticState(board, state.current_piece, state.next_queue),
                SemanticState(state.board, 2, state.next_queue),
                SemanticState(state.board, state.current_piece, (2, 0, 4, 6))]
    loaded = ev.deserialize_state(ev.serialize_state(state))
    assert loaded.current_piece == state.current_piece and loaded.next_queue == state.next_queue
    np.testing.assert_array_equal(loaded.board, state.board)
    assert ev.state_hash(loaded) == ev.state_hash(state)
    assert len({ev.state_hash(s) for s in [state, *variants]}) == 4
    a, b = record(), replace(record(), metadata={'seed': 99, 'timestamp': 'later', 'git_commit': 'changed'})
    assert a.state_hash == b.state_hash
    from hashlib import sha256
    assert a.state_hash == sha256(bytes(200) + bytes([1, 0, 2, 4, 6])).hexdigest()
    with pytest.raises(ValueError):
        loaded.board.setflags(write=True)


def test_mirror_is_a_distinct_semantic_state():
    board = np.zeros((20, 10), dtype=np.uint8)
    board[19, 0] = 1
    assert ev.state_hash(SemanticState(board, 1, (0, 2, 4, 6))) != ev.state_hash(
        SemanticState(board[:, ::-1], 1, (0, 2, 4, 6)))


def test_save_load_exact_roundtrip_and_duplicate_rejection(tmp_path):
    rows = [record(), record(piece=2, reachable_count=3)]
    destination = tmp_path / 'exam'
    ev.save_dataset(destination, rows, metadata(2))
    loaded, meta = ev.load_dataset(destination)
    assert [r.to_dict() for r in loaded] == [r.to_dict() for r in rows]
    assert meta['aggregate_label_statistics'] == ev.label_statistics(rows)
    with pytest.raises(ValueError, match='Duplicate state'):
        ev.save_dataset(tmp_path / 'duplicates', [rows[0], rows[0]], metadata(2))
    before = (destination / 'states.jsonl').read_bytes()
    with pytest.raises(FileExistsError):
        ev.save_dataset(destination, rows, metadata(2))
    assert (destination / 'states.jsonl').read_bytes() == before


@pytest.mark.parametrize('corruption', [
    'schema', 'board_height', 'board_width', 'board_binary', 'piece', 'piece_bool',
    'queue_length', 'queue_piece', 'hash', 'duplicate_state', 'valid_range', 'valid_bool',
    'valid_float', 'duplicate_valid', 'duplicate_reachable', 'subset', 'geometric_missing',
    'geometric_extra', 'count', 'statistics', 'zero_reachable', 'missing_metadata',
    'duplicate_json_key', 'nonfinite', 'extra_state_feature', 'blank_line',
])
def test_loader_rejects_corruption(tmp_path, corruption):
    path = tmp_path / 'exam'
    ev.save_dataset(path, [record(reachable_count=2)], metadata())
    state_file, meta_file = path / 'states.jsonl', path / 'metadata.json'
    value = json.loads(state_file.read_text())
    meta = json.loads(meta_file.read_text())
    if corruption == 'schema': meta['schema_version'] = 2
    elif corruption == 'board_height': value['state']['board'].pop()
    elif corruption == 'board_width': value['state']['board'][0] = '0'
    elif corruption == 'board_binary': value['state']['board'][0] = '2000000000'
    elif corruption == 'piece': value['state']['current_piece'] = 7
    elif corruption == 'piece_bool': value['state']['current_piece'] = True
    elif corruption == 'queue_length': value['state']['next_queue'].pop()
    elif corruption == 'queue_piece': value['state']['next_queue'][0] = -1
    elif corruption == 'hash': value['state_hash'] = 'bad'
    elif corruption == 'valid_range': value['valid_action_ids'][0] = 800
    elif corruption == 'valid_bool': value['valid_action_ids'][0] = True
    elif corruption == 'valid_float': value['valid_action_ids'][0] = 1.0
    elif corruption == 'duplicate_valid': value['valid_action_ids'].append(value['valid_action_ids'][0])
    elif corruption == 'duplicate_reachable': value['reachable_action_ids'] *= 2
    elif corruption == 'subset': value['reachable_action_ids'].append(0)
    elif corruption == 'geometric_missing': value['valid_action_ids'].pop()
    elif corruption == 'geometric_extra': value['valid_action_ids'].append(0)
    elif corruption == 'count': meta['accepted_state_count'] = 0
    elif corruption == 'statistics': meta['aggregate_label_statistics']['total_reachable_actions'] += 1
    elif corruption == 'zero_reachable': value['reachable_action_ids'] = []
    elif corruption == 'missing_metadata': del meta['generation_seed']
    elif corruption == 'extra_state_feature': value['state']['reachability'] = [1]
    text = json.dumps(value) + '\n'
    if corruption == 'duplicate_state': text *= 2
    elif corruption == 'duplicate_json_key': text = text.replace('"state_hash":', '"state_hash": "x", "state_hash":', 1)
    elif corruption == 'nonfinite': text = text.replace('"metadata": {}', '"metadata": {"x": NaN}')
    elif corruption == 'blank_line': text += '\n'
    state_file.write_text(text)
    meta_file.write_text(json.dumps(meta))
    with pytest.raises(ValueError):
        ev.load_dataset(path)


def test_exact_analytical_reference_mixed_metrics_and_geometric_greedy():
    first = record(reachable_count=3)  # O: 3/9.
    second = record(piece=0)  # I: 17/17, ensures state rather than action weighting.
    q = constant_network({0: 1000, first.valid_action_ids[-1]: 10})
    result = ev.evaluate_network(q, [first, second])
    assert result['analytical_random_reference']['overall_reachable_rate'] == pytest.approx(2 / 3)
    assert result['analytical_random_reference']['mixed_state_reachable_rate'] == pytest.approx(1 / 3)
    assert result['greedy_first_choice_reachable_rate'] == .5
    assert result['greedy_first_choice_unreachable_rate'] == .5
    assert result['mixed_state_greedy_reachable_rate'] == 0
    assert all(r['action_id'] != 0 for r in result['states'])
    assert result['states'][0]['highest_reachable_rank'] == 2
    assert result['states'][0]['margin'] == -10
    secondary = result['secondary_diagnostics']
    assert secondary['reachable_action_count'] == 20 and secondary['unreachable_action_count'] == 6
    assert secondary['paired_state_count'] == 1
    assert secondary['mean_q_gap'] == pytest.approx(-10 / 6)


def test_all_reachable_empty_mixed_and_tie_breaking():
    row = record()
    row = replace(row, valid_action_ids=tuple(reversed(row.valid_action_ids)))
    result = ev.evaluate_network(constant_network(), [row])
    assert result['greedy_first_choice_reachable_rate'] == 1
    assert result['mixed_state_greedy_reachable_rate'] is None
    assert result['analytical_random_reference']['mixed_state_reachable_rate'] is None
    assert result['secondary_diagnostics']['paired_state_count'] == 0
    assert result['secondary_diagnostics']['mean_unreachable_q'] is None
    assert result['states'][0]['margin'] is None
    assert result['states'][0]['action_id'] == min(row.valid_action_ids)


def test_labels_do_not_change_selected_action():
    row = record(reachable_count=1)
    network = constant_network({row.valid_action_ids[-1]: 4})
    first = ev.evaluate_network(network, [row])['states'][0]
    second = ev.evaluate_network(network, [replace(row, reachable_action_ids=row.valid_action_ids)])['states'][0]
    assert first['action_id'] == second['action_id'] and not first['reachable'] and second['reachable']


def test_evaluation_has_no_training_rng_optimizer_or_environment_effects(monkeypatch):
    trainer = DQNTrainer(TrainingConfig(decision_budget=9, device='cpu'))
    row = record(reachable_count=2)
    trainer.replay.add(Transition(row.state, row.valid_action_ids[0], 0., row.state, False, False))
    trainer.stored_experiences = 8
    trainer.gradient_updates = 3
    trainer.target_sync_count = 2
    trainer.online[0].eval()  # Mixed submodule modes must also survive evaluation.
    for parameter in trainer.online.parameters():
        parameter.grad = torch.ones_like(parameter)
    params = deepcopy(trainer.online.state_dict())
    target = deepcopy(trainer.target.state_dict())
    grads = [p.grad.clone() for p in trainer.online.parameters()]
    modes = [m.training for m in trainer.online.modules()]
    before = (trainer.stored_experiences, trainer.gradient_updates, trainer.target_sync_count,
              len(trainer.replay), trainer.replay._position, trainer.replay._rng.getstate(),
              trainer.policy._rng.getstate(), trainer.epsilon, deepcopy(trainer.optimizer.state_dict()))
    py_rng, np_rng, torch_rng = random.getstate(), deepcopy(np.random.get_state()), torch.get_rng_state()
    def forbidden(*args, **kwargs):
        pytest.fail('Evaluation crossed the offline inference boundary')
    for owner, name in [(trainer, 'observe'), (trainer.optimizer, 'step'),
                        (trainer.policy, 'select'), (trainer.replay, 'sample'),
                        (TacticalPlanner, 'plan'), (Executor, 'execute'),
                        (Tetris, 'step'), (Tetris, 'reset'), (Tetris, '__init__')]:
        monkeypatch.setattr(owner, name, forbidden)
    checks = []
    handle = trainer.online.register_forward_pre_hook(
        lambda net, inputs: checks.append((torch.is_grad_enabled(), net.training, inputs[0].shape)))
    try:
        first = ev.evaluate_network(trainer.online, [row])
        assert first == ev.evaluate_network(trainer.online, [row])
    finally:
        handle.remove()
    assert checks == [(False, False, torch.Size([1, 235]))] * 2
    assert before == (trainer.stored_experiences, trainer.gradient_updates, trainer.target_sync_count,
                      len(trainer.replay), trainer.replay._position, trainer.replay._rng.getstate(),
                      trainer.policy._rng.getstate(), trainer.epsilon, trainer.optimizer.state_dict())
    assert modes == [m.training for m in trainer.online.modules()]
    assert all(torch.equal(params[k], v) for k, v in trainer.online.state_dict().items())
    assert all(torch.equal(target[k], v) for k, v in trainer.target.state_dict().items())
    assert all(torch.equal(g, p.grad) for g, p in zip(grads, trainer.online.parameters()))
    assert random.getstate() == py_rng
    assert np_rng[0] == np.random.get_state()[0]
    np.testing.assert_array_equal(np_rng[1], np.random.get_state()[1])
    assert np_rng[2:] == np.random.get_state()[2:]
    assert torch.equal(torch_rng, torch.get_rng_state())
    # Arbitrary unrelated RNG/epsilon progression cannot affect greedy inference.
    trainer.stored_experiences = 9000
    random.random()
    np.random.random()
    torch.rand(7)
    assert first == ev.evaluate_network(trainer.online, [row])


@pytest.mark.parametrize('failure', ['nan', 'shape', 'exception'])
def test_eval_failure_restores_mode(failure):
    network = constant_network()
    def broken(inputs):
        if failure == 'exception': raise RuntimeError('injected')
        return torch.full((1, 800 if failure == 'nan' else 2), float('nan') if failure == 'nan' else 0.)
    network.forward = broken
    with pytest.raises((ValueError, RuntimeError)):
        ev.evaluate_network(network, [record()])
    assert network.training


def test_empty_dataset_cannot_evaluate():
    with pytest.raises(ValueError, match='empty'):
        ev.evaluate_network(QNetwork(), [])


def test_real_planner_partition_and_exact_restoration(env):
    select_piece(env, 1)
    env.board[17, 4:14] = 2
    env.randomizer.index = len(env.randomizer.bag) - 1
    before = state_signature(env)
    refs = {name: getattr(env, name) for name in ('board', 'queue', 'randomizer', 'holder', 'active_tetromino')}
    times = []
    row = build.label_candidate(env, timings=times)
    reachable = {SPACE.encode(TargetPlacement(x, 15, 0)) for x in range(4, 13)}
    unreachable = {SPACE.encode(TargetPlacement(x, 18, 0)) for x in range(4, 13)}
    assert set(row.valid_action_ids) == reachable | unreachable
    assert set(row.reachable_action_ids) == reachable
    assert len(times) == 18
    assert state_signature(env) == before
    assert all(getattr(env, k) is v for k, v in refs.items())


@pytest.mark.parametrize('failure', ['exception', 'timeout', 'unknown', 'mutation', 'zero'])
def test_bad_labeling_rejects_entire_candidate_and_restores(env, failure):
    before = state_signature(env)
    class BrokenPlanner:
        calls = 0
        def plan(self, game, target):
            self.calls += 1
            if failure == 'zero':
                return PlannerResult(PlannerStatus.UNREACHABLE, (), False, 0, 0, 0.)
            if self.calls > 1:
                if failure == 'exception': raise RuntimeError('injected')
                if failure == 'timeout': raise TimeoutError('injected')
                if failure == 'unknown': return SimpleNamespace(status='UNKNOWN')
                if failure == 'mutation': game.step(game.actions.hard_drop)
            return PlannerResult(PlannerStatus.REACHABLE, (), True, 0, 0, 0.)
    with pytest.raises(build.CandidateRejected) as error:
        build.label_candidate(env, BrokenPlanner())
    assert error.value.reason == dict(exception='PLANNER_ERROR', timeout='PLANNER_TIMEOUT',
        unknown='PLANNER_UNKNOWN', mutation='RESTORATION_FAILURE', zero='ZERO_REACHABLE')[failure]
    assert state_signature(env) == before


def test_real_native_exception_after_lock_restores(env, monkeypatch):
    original_step = env.step
    def broken(action):
        original_step(action)
        raise RuntimeError('after lock')
    monkeypatch.setattr(env, 'step', broken)
    before = state_signature(env)
    with pytest.raises(build.CandidateRejected, match='after lock'):
        build.label_candidate(env)
    assert state_signature(env) == before


def test_noncanonical_and_deadline_never_label(env):
    times = []
    with pytest.raises(build.CandidateRejected, match='wall-time'):
        build.label_candidate(env, timings=times, deadline=0)
    assert times == []
    env.step(env.actions.move_right)
    with pytest.raises(ValueError, match='canonical'):
        build.label_candidate(env)


def test_seeded_natural_collection_determinism_and_architecture():
    results = []
    class CheckedExecutor(Executor):
        calls = 0
        def execute(self, game, target, plan):
            assert_canonical(game)
            assert isinstance(target, TargetPlacement) and plan.reachable
            self.calls += 1
            result = super().execute(game, target, plan)
            assert result.success and result.target_verified
            if not result.terminated: assert_canonical(game)
            return result
    for seed in (42, 42, 7):
        executor = CheckedExecutor()
        rows, meta = build.build_dataset(build.BuildConfig(seed=seed, state_count=2), executor=executor)
        assert meta['status'] == 'completed' and executor.calls == 1
        assert meta['build_statistics']['verified_trajectory_placements'] == 1
        assert meta['build_statistics']['planner_labeling']['call_count'] == sum(len(r.valid_action_ids) for r in rows)
        results.append([r.to_dict() for r in rows])
    assert results[0] == results[1] and results[0] != results[2]
    assert results[0][0]['state']['board'] != results[0][1]['state']['board']


@pytest.mark.parametrize('failure', ['zero', 'unknown', 'exception', 'timeout', 'restoration'])
def test_failed_build_has_no_uncertain_records_and_persists_context(tmp_path, failure):
    class BrokenPlanner:
        def plan(self, game, target):
            if failure == 'exception': raise RuntimeError('bad')
            if failure == 'timeout': raise TimeoutError('slow')
            if failure == 'restoration': game.step(game.actions.hard_drop)
            return SimpleNamespace(status=PlannerStatus.UNREACHABLE if failure == 'zero' else 'UNKNOWN')
    rows, meta = build.build_dataset(build.BuildConfig(state_count=1), planner=BrokenPlanner())
    assert not rows and meta['status'] == 'diagnostic_failure'
    assert meta['rejected_state_count'] == 1
    assert meta['build_statistics']['zero_reachable_states'] == int(failure == 'zero')
    assert meta['diagnostics'][0]['native_context']['queue']
    assert meta['diagnostics'][0]['state']['board'] == ['0' * 10] * 20
    ev.save_dataset(tmp_path / 'failed', rows, meta)
    loaded, restored = ev.load_dataset(tmp_path / 'failed')
    assert not loaded and restored['diagnostics'] == meta['diagnostics']


def test_duplicate_collection_is_bounded_and_reported(monkeypatch):
    class SameReset(Tetris):
        def reset(self, **kwargs): return super().reset(seed=42)
    class ResetExecutor:
        def execute(self, *args):
            return SimpleNamespace(success=True, target_verified=True, terminated=True, truncated=False)
    class ReachablePlanner:
        def plan(self, *args): return SimpleNamespace(status=PlannerStatus.REACHABLE)
    monkeypatch.setattr(build, 'Tetris', SameReset)
    rows, meta = build.build_dataset(build.BuildConfig(state_count=2, max_candidates=3),
                                    planner=ReachablePlanner(), executor=ResetExecutor())
    assert len(rows) == 1 and meta['status'] == 'incomplete'
    assert meta['build_statistics']['rejected_duplicates'] == 2
    assert meta['build_statistics']['rejected_candidates'] == 0
    assert meta['diagnostics'][0]['reason'] == 'CANDIDATE_LIMIT'


def load_cli(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parents[1] / f'scripts/{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_actual_snapshot_load_and_evaluation_cli(tmp_path, capsys):
    trainer = DQNTrainer(TrainingConfig(decision_budget=1, device='cpu'))
    row = record(reachable_count=2)
    writer = RunArtifacts(ArtifactConfig(tmp_path / 'snapshot'), trainer, 1)
    writer.finish(trainer, dict(decision_count=0, episode=0), 'completed')
    checkpoint = tmp_path / 'snapshot/checkpoint_final.pt'
    rng = torch.get_rng_state()
    network = ev.load_checkpoint_network(checkpoint)
    assert torch.equal(rng, torch.get_rng_state())
    assert all(torch.equal(v, trainer.online.state_dict()[k]) for k, v in network.state_dict().items())
    assert ev.evaluate_network(network, [row]) == ev.evaluate_network(trainer.online, [row])
    ev.save_dataset(tmp_path / 'exam', [row], metadata())
    cli = load_cli('evaluate_policy')
    for flags, name in [(['--checkpoint', str(checkpoint)], 'checkpoint'), (['--untrained-seed', '42'], 'untrained')]:
        output = tmp_path / f'{name}.json'
        assert cli.main(['--dataset', str(tmp_path / 'exam'), *flags, '--output', str(output)]) == 0
        assert json.loads(output.read_text())['evaluated_states'] == 1


def test_incomplete_exam_cli_refused(tmp_path):
    ev.save_dataset(tmp_path / 'exam', [record()], dict(metadata(), status='incomplete'))
    with pytest.raises(SystemExit):
        load_cli('evaluate_policy').main(['--dataset', str(tmp_path / 'exam'), '--untrained-seed', '42'])


@pytest.mark.parametrize('config', [dict(seed=-1), dict(seed=True), dict(state_count=0),
    dict(max_candidates=0), dict(max_seconds=0), dict(max_seconds=float('nan'))])
def test_build_config_validation(config):
    with pytest.raises(ValueError): build.BuildConfig(**config)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
def test_cpu_cuda_evaluation_sanity():
    rows = [record(reachable_count=2), record(piece=0)]
    network = QNetwork()
    cpu = ev.evaluate_network(network, rows)
    gpu = ev.evaluate_network(deepcopy(network).cuda(), rows)
    assert [r['action_id'] for r in cpu['states']] == [r['action_id'] for r in gpu['states']]
    assert cpu['greedy_first_choice_reachable_rate'] == gpu['greedy_first_choice_reachable_rate']
    assert cpu['secondary_diagnostics']['mean_reachable_q'] == pytest.approx(
        gpu['secondary_diagnostics']['mean_reachable_q'], abs=1e-6)
