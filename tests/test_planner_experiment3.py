"""Observation suppression, native transition equivalence and fallback boundaries."""
import copy
import json
import sys
from dataclasses import fields

import numpy as np
import pytest
from tetris_gymnasium.envs.tetris import Tetris, TetrisState
from tetris_gymnasium.mappings.actions import ActionsMapping

import tetris_agent.planner as planner_module
from tetris_agent import TacticalPlanner, TargetPlacement
from test_planner import select_piece, state_signature, piece_signature


def semantic(env):
    # No cross-environment method identity or creation-time RNG entropy.
    r = env.randomizer
    return (env.board.shape, env.board.dtype.str, env.board.tobytes(),
            piece_signature(env.active_tetromino), env.x, env.y,
            tuple(env.queue.queue), env.queue.size, env.queue.queue.maxlen,
            env.queue.randomizer is r, tuple(map(piece_signature, env.holder.queue)),
            env.holder.size, env.holder.queue.maxlen, r.index, r.bag.tobytes(),
            json.dumps(r.rng.bit_generator.state, sort_keys=True),
            json.dumps(env.np_random.bit_generator.state, sort_keys=True),
            env.has_swapped, env.game_over, tuple(env.score(n) for n in range(5)),
            tuple(map(piece_signature, env.tetrominoes)))


def new_env(**kwargs):
    env = Tetris(**kwargs)
    env.reset(seed=42)
    select_piece(env, 1)
    return env


@pytest.mark.parametrize('gravity', [False, True])
@pytest.mark.parametrize('mapped', [False, True])
@pytest.mark.parametrize('action_name', ['move_left', 'move_right', 'move_down',
    'rotate_clockwise', 'rotate_counterclockwise', 'no_op', 'hard_drop', 'swap'])
def test_native_vs_search_step(gravity, mapped, action_name):
    kwargs = {'gravity': gravity}
    if mapped:
        kwargs['actions_mapping'] = ActionsMapping(7, 0, 1, 2, 3, 4, 5, 6)
    native, search = new_env(**kwargs), new_env(**kwargs)
    try:
        for env in (native, search):
            select_piece(env, 2)
            env.holder.queue.append(copy.copy(env.tetrominoes[0]))
            env.randomizer.index = 6
        assert planner_module._can_skip_search_observation(search)
        search._get_obs = planner_module._search_observation
        a = native.step(getattr(native.actions, action_name))
        b = search.step(getattr(search.actions, action_name))
        assert b[0] is None and isinstance(a[0], dict)
        assert a[1:] == b[1:]
        assert semantic(native) == semantic(search)
    finally:
        vars(search).pop('_get_obs', None)
        native.close()
        search.close()


@pytest.mark.parametrize('kind', ['auto_lock', 'line_clear', 'two_rows', 'spawn_game_over'])
@pytest.mark.parametrize('index', [0, 1, 6])
def test_lock_boundaries(kind, index):
    native, search = new_env(), new_env()
    try:
        for env in (native, search):
            env.holder.queue.append(copy.copy(env.tetrominoes[2]))
            env.randomizer.index = index
            if kind == 'auto_lock':
                env.x, env.y = 7, 18
            elif kind in ('line_clear', 'two_rows'):
                rows = slice(18, 20) if kind == 'two_rows' else slice(19, 20)
                env.board[rows, 4:14] = 2
                env.board[rows, 8:10] = 0
            else:
                env.board[1, 7] = 2
                env.queue.queue[0] = 0
        assert planner_module._can_skip_search_observation(search)
        search._get_obs = planner_module._search_observation
        action = 'move_right' if kind == 'auto_lock' else 'hard_drop'
        a = native.step(getattr(native.actions, action))
        b = search.step(getattr(search.actions, action))
        assert a[1:] == b[1:]
        assert semantic(native) == semantic(search)
        if kind == 'spawn_game_over':
            assert b[2]
        if kind in ('line_clear', 'two_rows'):
            assert b[-1]['lines_cleared'] == (2 if kind == 'two_rows' else 1)
        assert [native.queue.get_next_tetromino() for _ in range(40)] == [
            search.queue.get_next_tetromino() for _ in range(40)]
        assert semantic(native) == semantic(search)
    finally:
        vars(search).pop('_get_obs', None)
        native.close()
        search.close()


@pytest.mark.parametrize('kind', ['normal', 'invalid', 'game_over', 'unreachable'])
def test_binding_and_objects_restored(kind):
    env = new_env()
    try:
        if kind == 'game_over':
            env.game_over = True
        elif kind == 'unreachable':
            env.board[17, 4:14] = 2
        target = TargetPlacement(7, 18, 0) if kind == 'unreachable' else TargetPlacement(10, 18, 0)
        if kind == 'invalid':
            target = TargetPlacement(13, 18, 0)
        before = state_signature(env)
        refs = {f.name: getattr(env, f.name) for f in fields(TetrisState)}
        obs = env._get_obs
        result = TacticalPlanner().plan(env, target)
        assert result.reachable == (kind == 'normal')
        assert state_signature(env) == before
        assert env._get_obs == obs and '_get_obs' not in vars(env)
        # Accessing a class-bound score method creates a fresh bound object.
        assert all(getattr(env, k) is v for k, v in refs.items() if k != 'score')
        assert env.score == refs['score'] and 'score' not in vars(env)
        assert isinstance(env._get_obs(), dict)
    finally:
        env.close()


@pytest.mark.parametrize('failure', ['primitive', 'hard_drop', 'observer'])
@pytest.mark.parametrize('existing_observer', [False, True])
def test_exception_restoration(failure, existing_observer, monkeypatch):
    env = new_env()
    calls = []
    try:
        if existing_observer:
            old_place = env.place_active_tetromino
            env.place_active_tetromino = lambda: old_place()
        place = env.place_active_tetromino
        if failure == 'observer':
            original = planner_module.target_at_lock
            def broken(*args):
                calls.append(env._get_obs())
                original(*args)
                raise RuntimeError('observer failure')
            monkeypatch.setattr(planner_module, 'target_at_lock', broken)
        else:
            contains = env.action_space.contains
            def broken(action):
                if (failure == 'hard_drop' and action == env.actions.hard_drop
                        or failure == 'primitive' and action != env.actions.hard_drop):
                    calls.append(env._get_obs())
                    raise RuntimeError(f'{failure} failure')
                return contains(action)
            monkeypatch.setattr(env.action_space, 'contains', broken)
        before = state_signature(env)
        obs = env._get_obs
        with pytest.raises(RuntimeError, match=f'{failure} failure'):
            TacticalPlanner().plan(env, TargetPlacement(10, 18, 0))
        assert state_signature(env) == before
        assert env._get_obs == obs and '_get_obs' not in vars(env)
        assert env.place_active_tetromino == place
        assert calls and all(isinstance(v, dict) if existing_observer else v is None for v in calls)
    finally:
        env.close()


@pytest.mark.parametrize('kind', ['subclass_obs', 'subclass_step', 'subclass_plain',
    'instance_obs', 'instance_step', 'class_obs', 'class_step', 'projection',
    'queue_obs', 'holder_obs', 'place', 'score'])
def test_custom_fallback_preserves_observation_side_effects(kind, monkeypatch):
    class CustomObs(Tetris):
        def _get_obs(self):
            self.custom_obs_count += 1
            return super()._get_obs()
    class CustomStep(Tetris):
        def step(self, action):
            result = super().step(action)
            assert isinstance(result[0], dict)
            return result
    class Plain(Tetris):
        pass
    cls = {'subclass_obs': CustomObs, 'subclass_step': CustomStep,
           'subclass_plain': Plain}.get(kind, Tetris)
    env = cls()
    env.custom_obs_count = 0
    env.reset(seed=42)
    select_piece(env, 1)
    native_obs = Tetris._get_obs
    try:
        if kind in ('instance_obs', 'class_obs'):
            def counted(instance):
                instance.custom_obs_count += 1
                return native_obs(instance)
            if kind == 'instance_obs':
                env._get_obs = lambda: counted(env)
            else:
                monkeypatch.setattr(Tetris, '_get_obs', counted)
        if kind in ('instance_step', 'class_step'):
            native_step = Tetris.step
            def step(instance, action):
                value = native_step(instance, action)
                assert isinstance(value[0], dict)
                return value
            if kind == 'instance_step':
                env.step = lambda action: step(env, action)
            else:
                monkeypatch.setattr(Tetris, 'step', step)
        if kind in ('projection', 'place', 'score'):
            attr = {'projection': 'project_tetromino', 'place': 'place_active_tetromino', 'score': 'score'}[kind]
            monkeypatch.setattr(env, attr, getattr(env, attr))
        if kind == 'queue_obs':
            monkeypatch.setattr(env.queue, 'get_queue', env.queue.get_queue)
        if kind == 'holder_obs':
            monkeypatch.setattr(env.holder, 'get_tetrominoes', env.holder.get_tetrominoes)
        assert not planner_module._can_skip_search_observation(env)
        present = '_get_obs' in vars(env)
        obs = env._get_obs
        before = env.custom_obs_count
        native_calls = []
        def hook(frame, event, arg):
            if event == 'call' and frame.f_code is native_obs.__code__:
                native_calls.append(True)
        previous = sys.getprofile()
        try:
            sys.setprofile(hook)
            TacticalPlanner().plan(env, TargetPlacement(10, 18, 0))
        finally:
            sys.setprofile(previous)
        assert native_calls
        if kind in ('instance_obs', 'class_obs', 'subclass_obs'):
            assert env.custom_obs_count > before
        assert ('_get_obs' in vars(env)) == present and env._get_obs == obs
    finally:
        env.close()


def test_existing_observation_exception_preserves_exact_binding():
    env = new_env()
    try:
        def broken():
            raise RuntimeError('custom observation failure')
        env._get_obs = broken
        before = state_signature(env)
        with pytest.raises(RuntimeError, match='custom observation failure'):
            TacticalPlanner().plan(env, TargetPlacement(8, 18, 0))
        assert env._get_obs is broken
        assert state_signature(env) == before
    finally:
        env.close()


def test_nested_plan_retains_outer_binding():
    env = new_env()
    nested = []
    previous = sys.getprofile()
    def hook(frame, event, arg):
        if not nested and event == 'call' and frame.f_code is Tetris.step.__code__:
            outer_obs = env._get_obs
            assert outer_obs is planner_module._search_observation
            nested.append(True)
            TacticalPlanner().plan(env, TargetPlacement(8, 18, 0))
            assert env._get_obs is outer_obs
    try:
        before = state_signature(env)
        sys.setprofile(hook)
        result = TacticalPlanner().plan(env, TargetPlacement(10, 18, 0))
        assert nested and result.actions == (env.actions.move_right,) * 2
        assert state_signature(env) == before and '_get_obs' not in vars(env)
    finally:
        sys.setprofile(previous)
        env.close()


@pytest.mark.parametrize('primitive', [False, True])
def test_exception_after_native_step_mutation(primitive):
    env = new_env()
    previous = sys.getprofile()
    reached = []
    before = state_signature(env)
    refs = {f.name: getattr(env, f.name) for f in fields(TetrisState)}
    def hook(frame, event, arg):
        if event == 'return' and frame.f_code is Tetris.step.__code__:
            action = frame.f_locals['action']
            if (action != env.actions.hard_drop) == primitive:
                assert env._get_obs is planner_module._search_observation
                assert arg[0] is None
                reached.append(True)
                raise RuntimeError('after native mutation')
    try:
        sys.setprofile(hook)
        with pytest.raises(RuntimeError, match='after native mutation'):
            TacticalPlanner().plan(env, TargetPlacement(10, 18, 0))
        assert reached and state_signature(env) == before
        assert all(getattr(env, k) is v for k, v in refs.items() if k != 'score')
        assert '_get_obs' not in vars(env)
    finally:
        sys.setprofile(previous)
        env.close()


@pytest.mark.parametrize('gravity', [False, True])
@pytest.mark.parametrize('mapped', [False, True])
def test_optimized_plan_with_gravity_and_action_mapping(gravity, mapped, monkeypatch):
    kwargs = {'gravity': gravity}
    if mapped:
        kwargs['actions_mapping'] = ActionsMapping(7, 0, 1, 2, 3, 4, 5, 6)
    env = new_env(**kwargs)
    previous = sys.getprofile()
    calls = []
    def hook(frame, event, arg):
        if event == 'call' and frame.f_code is Tetris._get_obs.__code__:
            calls.append(True)
    try:
        # Force the native-observation fallback without a benchmark-file dependency.
        with monkeypatch.context() as native:
            native.setattr(planner_module, '_AUDITED_VERSION', False)
            expected = TacticalPlanner().plan(env, TargetPlacement(10, 18, 0))
        before = state_signature(env)
        assert planner_module._can_skip_search_observation(env)
        sys.setprofile(hook)
        actual = TacticalPlanner().plan(env, TargetPlacement(10, 18, 0))
        assert not calls and state_signature(env) == before
        assert (actual.status, actual.actions, actual.path_length, actual.needs_hard_drop,
                actual.expanded_states, actual.visited_states) == (
                expected.status, expected.actions, expected.path_length, expected.needs_hard_drop,
                expected.expanded_states, expected.visited_states)
    finally:
        sys.setprofile(previous)
        env.close()
