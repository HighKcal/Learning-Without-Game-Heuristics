"""Experiment 2: compare directly with f961818, including native branch boundaries."""
from collections import Counter
from contextlib import ExitStack
from dataclasses import fields
import copy
import hashlib
import json
from pathlib import Path
import pickle
import sys
import types
from unittest.mock import patch

from _planner_benchmark import (
    ROOT, Tetris, TacticalPlanner, TargetPlacement, load_corpus, controlled_cases,
    restore, capture, state_signature, result_record, verify, write_json, select_piece,
)
from tetris_gymnasium.envs.tetris import TetrisState
from tetris_gymnasium.components.tetromino_randomizer import BagRandomizer

OUTPUT = ROOT / '_validation/planner_experiment2'
BASELINE_SOURCE = ROOT / 'benchmarks/planner/live_drop_probe/experiment1_planner.py'
FIELDS = ('status', 'actions', 'action_length', 'needs_hard_drop',
          'expanded_states', 'visited_states', 'target_verified')


def baseline_class():
    module = types.ModuleType('tetris_agent._experiment1_baseline')
    module.__file__ = str(BASELINE_SOURCE)
    sys.modules[module.__name__] = module
    exec(compile(Path(module.__file__).read_text(), module.__file__, 'exec'), module.__dict__)
    return module.TacticalPlanner


def snapshot_signature(state):
    # Compare semantic RNG state, not PCG64's construction-time SeedSequence
    # entropy in its pickle (each native copy creates a fresh PCG64 first).
    def piece(value):
        if value is None:
            return None
        return (value.id, tuple(value.color_rgb), value.matrix.shape,
                value.matrix.dtype.str, value.matrix.tobytes())
    randomizer = state.randomizer
    values = (state.board.shape, state.board.dtype.str, state.board.tobytes(),
              piece(state.active_tetromino), int(state.x), int(state.y),
              state.queue.size, tuple(state.queue.queue), state.queue.queue.maxlen,
              state.queue.randomizer is randomizer, state.holder.size,
              tuple(map(piece, state.holder.queue)), state.holder.queue.maxlen,
              type(randomizer).__name__, randomizer.size, randomizer.index,
              randomizer.bag.dtype.str, randomizer.bag.tobytes(),
              json.dumps(randomizer.rng.bit_generator.state, sort_keys=True),
              bool(state.has_swapped), bool(state.game_over))
    return hashlib.sha256(pickle.dumps(values, protocol=5)).hexdigest()


def checked_plan(env, target, cls, counts=None, audit=False, fail_child=False):
    before = state_signature(env)
    refs = {f.name: getattr(env, f.name) for f in fields(TetrisState)}
    gym_rng = env.np_random
    saved_children = []
    trace = []
    child_pending = False
    step = env.step
    get_state = env.get_state
    with ExitStack() as stack:
        if counts is not None:
            for owner, name, label in ((Tetris, 'get_state', 'Tetris.get_state'),
                    (BagRandomizer, '__copy__', 'BagRandomizer.__copy__'),
                    (Tetris, 'step', 'Tetris.step'), (Tetris, '_get_obs', 'Tetris._get_obs')):
                original = getattr(owner, name)
                def counted(instance, *args, _original=original, _label=label, **kwargs):
                    counts[_label] += 1
                    if _label == 'Tetris.step' and args[0] == instance.actions.hard_drop:
                        counts['HARD_DROP probe'] += 1
                    return _original(instance, *args, **kwargs)
                stack.enter_context(patch.object(owner, name, counted))
        if audit:
            def observed_snapshot():
                nonlocal child_pending
                state = get_state()
                caller = sys._getframe(1)
                if caller.f_code.co_name == 'plan' and 'key' in caller.f_locals:
                    saved_children.append((state, snapshot_signature(state), state.score))
                    child_pending = True
                    assert not __import__('numpy').shares_memory(state.board, env.board)
                    assert state.active_tetromino is not env.active_tetromino
                    assert not __import__('numpy').shares_memory(state.active_tetromino.matrix,
                                                               env.active_tetromino.matrix)
                    assert state.active_tetromino.color_rgb is not env.active_tetromino.color_rgb
                    assert state.queue is not env.queue and state.queue.queue is not env.queue.queue
                    assert state.queue.randomizer is state.randomizer
                    assert state.randomizer is not env.randomizer
                    assert state.randomizer.rng is not env.randomizer.rng
                    assert not __import__('numpy').shares_memory(state.randomizer.bag, env.randomizer.bag)
                    assert state.holder is not env.holder and state.holder.queue is not env.holder.queue
                    assert all(a is b for a, b in zip(state.holder.queue, env.holder.queue))
                return state

            def observed_step(action):
                nonlocal child_pending
                trace.append((int(action), snapshot_signature(env)))
                is_child = child_pending and action == env.actions.hard_drop
                if is_child:
                    assert snapshot_signature(env) == saved_children[-1][1]
                child_pending = False
                returned = step(action)
                for state, signature, score in saved_children:
                    assert snapshot_signature(state) == signature
                    assert state.score is score
                if fail_child and is_child:
                    raise RuntimeError('injected after live child HARD_DROP')
                return returned

            stack.enter_context(patch.object(env, 'get_state', observed_snapshot))
            stack.enter_context(patch.object(env, 'step', observed_step))
        try:
            result = cls().plan(env, target)
        finally:
            assert state_signature(env) == before
            assert all(getattr(env, name) is ref for name, ref in refs.items()
                       if name != 'score' or 'score' in vars(env))
            assert env.np_random is gym_rng
    return result, trace, len(saved_children)


def record(env, entry, cls, counts=None, audit=False):
    restore(env, entry['snapshot'])
    target = TargetPlacement(**entry['target'])
    result, trace, children = checked_plan(env, target, cls, counts, audit)
    return {'id': entry['id'], 'target': entry['target'], **result_record(result),
            'target_verified': verify(env, target, result)}, trace, children


def boundary_cases():
    cases = []
    for index in (0, 1, 6):
        for kind in ('spawn', 'line_clear', 'spawn_game_over', 'sealed', 'auto_lock'):
            env = Tetris(gravity=True)
            env.reset(seed=42)
            select_piece(env, 1)
            env.holder.queue.append(env.tetrominoes[2])
            env.has_swapped = True
            env.randomizer.index = index
            target = TargetPlacement(10, 18, 0)
            if kind == 'line_clear':
                env.board[19, 4:14] = 2
                env.board[19, 10:12] = 0
            elif kind == 'spawn_game_over':
                env.board[1, 7] = 2
                env.queue.queue[0] = 0
            elif kind == 'sealed':
                env.board[17, 4:14] = 2
                target = TargetPlacement(7, 18, 0)
            elif kind == 'auto_lock':
                env.x, env.y = 7, 18
                target = TargetPlacement(8, 18, 0)
            cases.append({'id': f'{kind}-bag-index-{index}', 'snapshot': capture(env),
                          'target': dict(x=target.x, y=target.y, orientation=target.orientation)})
            env.close()
    return cases


def main():
    OUTPUT.mkdir(exist_ok=True)
    baseline = baseline_class()
    corpus, manifest = load_corpus(ROOT / 'benchmarks/planner/baseline/planner_v001_corpus.pkl.gz')
    counts = {'experiment1': Counter(), 'experiment2': Counter()}
    records = {'experiment1': [], 'experiment2': []}
    env = Tetris(gravity=True)
    env.reset(seed=42)
    try:
        if '--boundaries-only' in sys.argv:
            for label in records:
                records[label] = json.loads((OUTPUT / f'{label}_strict_correctness.json').read_text())
            counts = json.loads((OUTPUT / 'corpus_counts.json').read_text())
        else:
            for index, entry in enumerate(corpus, 1):
                for label, cls in (('experiment1', baseline), ('experiment2', TacticalPlanner)):
                    records[label].append(record(env, entry, cls, counts[label])[0])
                assert records['experiment1'][-1] == records['experiment2'][-1], entry['id']
                if index % 200 == 0:
                    print(f'strict corpus {index}/{len(corpus)} matched', flush=True)
            for label in records:
                write_json(OUTPUT / f'{label}_strict_correctness.json', records[label])
            write_json(OUTPUT / 'corpus_counts.json', counts)
        controlled = []
        boundaries = []
        for group, entries, output in (('controlled', controlled_cases(), controlled),
                                        ('boundary', boundary_cases(), boundaries)):
            for entry in entries:
                first, first_trace, first_children = record(env, entry, baseline, audit=True)
                second, second_trace, second_children = record(env, entry, TacticalPlanner, audit=True)
                assert first == second, entry['id']
                assert first_trace == second_trace, entry['id']
                assert first_children == second_children
                if entry['id'].startswith('spawn_game_over'):
                    assert env.game_over and second['needs_hard_drop']
                if entry['id'].startswith('line_clear'):
                    assert __import__('numpy').count_nonzero(env.board[:20, 4:14]) == 2
                if entry['id'].startswith('auto_lock'):
                    assert not second['needs_hard_drop'] and second['action_length'] == 1
                output.append({**second, 'native_step_trace_matched': len(second_trace),
                               'child_snapshots_unchanged': second_children})
                print(f'{group}: {entry["id"]} matched; children={second_children}', flush=True)
        write_json(OUTPUT / 'strict_controlled_correctness.json', controlled)
        write_json(OUTPUT / 'boundary_validation.json', boundaries)
        # Fail specifically after the optimized child probe, with/without prior observer.
        exceptions = []
        for existing in (False, True):
            for cls in (baseline, TacticalPlanner):
                restore(env, boundary_cases()[2]['snapshot'])
                observer = env.place_active_tetromino
                if existing:
                    env.place_active_tetromino = observer
                try:
                    checked_plan(env, TargetPlacement(10, 18, 0), cls,
                                 audit=True, fail_child=True)
                except RuntimeError as exc:
                    assert str(exc) == 'injected after live child HARD_DROP'
                else:
                    raise AssertionError('child exception not reached')
                assert ('place_active_tetromino' in vars(env)) == existing
                if existing:
                    assert env.place_active_tetromino is observer
                    del env.place_active_tetromino
                exceptions.append({'implementation': cls.__module__, 'existing_observer': existing,
                                   'restoration_verified': True})
        write_json(OUTPUT / 'exception_validation.json', exceptions)
    finally:
        env.close()
    expected_before = {'Tetris.get_state': 317397, 'BagRandomizer.__copy__': 317397,
                       'Tetris.step': 269349, 'Tetris._get_obs': 269349, 'HARD_DROP probe': 48041}
    expected_after = {**expected_before, 'Tetris.get_state': 270352,
                      'BagRandomizer.__copy__': 270352}
    assert dict(counts['experiment1']) == expected_before
    assert dict(counts['experiment2']) == expected_after
    write_json(OUTPUT / 'strict_validation.json', {
        'baseline_commit': 'f961818', 'corpus_sha256': manifest['corpus_sha256'],
        'compared_fields': FIELDS, 'corpus_matched': len(corpus),
        'controlled_matched': len(controlled), 'boundary_matched': len(boundaries),
        'exception_checks': len(exceptions), 'counts': counts,
        'expanded_states': sum(r['expanded_states'] for r in records['experiment2']),
        'visited_states': sum(r['visited_states'] for r in records['experiment2']),
        'counts_scope': '1003 plan calls only, excludes restoration/oracle/controlled/boundaries'})
    print(json.dumps(counts, indent=2), flush=True)


if __name__ == '__main__':
    main()
