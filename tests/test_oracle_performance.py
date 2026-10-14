"""Ownership and native differential gates for the optimized branch restore."""
from copy import copy
from dataclasses import fields
import json

import numpy as np
import pytest
from tetris_gymnasium.envs.tetris import Tetris
from tetris_gymnasium.components.tetromino_randomizer import BagRandomizer
from tetris_agent import planner
from tetris_agent._search_snapshot import BranchRestorer, can_reuse_randomizer
from test_planner import state_signature, piece_signature


def value(env):
    return (env.board.shape, env.board.dtype.str, env.board.tobytes(),
        piece_signature(env.active_tetromino), env.x, env.y,
        tuple(env.queue.queue), env.queue.queue.maxlen, env.queue.size,
        env.queue.randomizer is env.randomizer,
        tuple(map(piece_signature,env.holder.queue)), env.holder.queue.maxlen,
        env.holder.size, env.has_swapped, env.game_over, env.score(2),
        env.randomizer.size, env.randomizer.index, env.randomizer.bag.tobytes(),
        json.dumps(env.randomizer.rng.bit_generator.state,sort_keys=True))


@pytest.mark.parametrize('primitives',[False,True])
def test_branch_restore_full_native_differential(primitives):
    a, b = Tetris(), Tetris()
    a.reset(seed=11); b.reset(seed=11)
    if primitives:
        from tetris_agent._search_primitives import SearchPrimitives
        SearchPrimitives(b).install()
    lock_a,lock_b=[],[]
    def observer(env,events):
        place=env.place_active_tetromino
        def observed():
            events.append((env.x,env.y,piece_signature(env.active_tetromino)))
            place()
        return observed
    a.place_active_tetromino=observer(a,lock_a)
    b.place_active_tetromino=observer(b,lock_b)
    restore = BranchRestorer(a.get_state())
    rng = np.random.default_rng(5290)
    checks, locks, clears, deaths = 0, 0, 0, 0
    for i in range(2048):
        if a.game_over or i % 16 == 0:
            a.reset(seed=i+1)
            a.active_tetromino = a.tetrominoes[i%7]
            a.reset_tetromino_position()
            if i % 32 == 0:
                a.board[19,4:14] = 2
                a.board[19,7:11] = 0
            if i % 64 == 0:
                a.board[3,4:14] = 2
        a.gravity_enabled = b.gravity_enabled = (i%5 != 0)
        a.randomizer.index = i%7  # Includes bag reshuffle on spawn.
        if i%8 == 0: a.holder.queue.append(a.tetrominoes[(i+2)%7])
        state = a.get_state()
        saved = (state.board.tobytes(), state.randomizer.bag.tobytes(),
                 json.dumps(state.randomizer.rng.bit_generator.state), tuple(state.queue.queue),
                 piece_signature(state.active_tetromino), tuple(map(piece_signature,state.holder.queue)))
        for action in (0,1,2,3,4,5,7):
            a.set_state(state); a.set_state(a.get_state())
            restore(b,state)
            assert value(a) == value(b)
            lock_a.clear(); lock_b.clear()
            aa,bb = a.step(action), b.step(action)
            assert lock_a==lock_b
            assert value(a) == value(b), (i,action)
            assert aa[1:] == bb[1:]
            assert all(np.array_equal(aa[0][key],bb[0][key]) for key in aa[0])
            assert (state.board.tobytes(),state.randomizer.bag.tobytes(),
                json.dumps(state.randomizer.rng.bit_generator.state),tuple(state.queue.queue),
                piece_signature(state.active_tetromino),tuple(map(piece_signature,state.holder.queue))) == saved
            assert not np.shares_memory(state.board,b.board)
            assert state.randomizer is not b.randomizer
            assert state.randomizer.rng is not b.randomizer.rng
            assert state.queue is not b.queue and state.holder is not b.holder
            checks += 1
            locks += int(tuple(b.queue.queue) != tuple(state.queue.queue))
            clears += int(bb[4]['lines_cleared'])
            deaths += int(bb[2])
        a.set_state(state); a.set_state(a.get_state())
        a.step(int(rng.choice((0,1,2,3,4,5,7))))
    assert checks == 14336 and locks and clears and deaths
    print(dict(transitions=checks, locks=locks, lines=clears, terminal=deaths))


@pytest.mark.parametrize('when', ['before_step','after_step'])
def test_fast_path_exception_restores_original(monkeypatch, when):
    env = Tetris(); env.reset(seed=91)
    env.randomizer.index = 6
    before = state_signature(env)
    refs = {f.name:getattr(env,f.name) for f in fields(planner.TetrisState)}
    native_restore = BranchRestorer.__call__
    def fail(self, base, state):
        native_restore(self,base,state)
        if when == 'after_step': base.step(base.actions.hard_drop)
        raise RuntimeError('injected fast branch failure')
    monkeypatch.setattr(BranchRestorer,'__call__',fail)
    with pytest.raises(RuntimeError,match='injected'):
        planner.TacticalPlanner().enumerate_reachable_targets(env)
    assert state_signature(env)==before
    assert all(getattr(env,k) is v for k,v in refs.items() if k!='score')
    assert '_get_obs' not in vars(env)


def test_custom_component_falls_back(monkeypatch):
    env = Tetris(); env.reset(seed=5)
    assert can_reuse_randomizer(env)
    monkeypatch.setattr(BagRandomizer,'get_next_tetromino',lambda self: 0)
    assert not can_reuse_randomizer(env)


def test_instance_randomizer_hook_falls_back():
    env = Tetris(); env.reset(seed=5)
    env.randomizer.get_next_tetromino = lambda: 0
    assert not can_reuse_randomizer(env)


def test_action_space_hook_falls_back():
    env = Tetris(); env.reset(seed=5)
    env.action_space.contains = lambda action: True
    assert not can_reuse_randomizer(env)


def test_runtime_version_guard_retained(monkeypatch):
    from tetris_agent.rl import oracle
    env=Tetris(); env.reset(seed=42)
    monkeypatch.setattr(oracle,'version',lambda package:'0.3.2')
    with pytest.raises(oracle.OracleInvariantError,match='audited'):
        oracle.compute_boundary(env,planner.TacticalPlanner())


def test_lock_verifier_hook_retains_native_copy_path(monkeypatch):
    env=Tetris(); env.reset(seed=42)
    original=planner.target_at_lock
    monkeypatch.setattr(planner,'target_at_lock',lambda *args:original(*args))
    def forbidden(*args): raise AssertionError('custom verifier must use fallback')
    monkeypatch.setattr(BranchRestorer,'__call__',forbidden)
    before=state_signature(env)
    assert planner.TacticalPlanner().enumerate_reachable_targets(env).placements
    assert state_signature(env)==before
