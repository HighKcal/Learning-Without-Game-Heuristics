import numpy as np
import pytest
from tetris_gymnasium.envs.tetris import Tetris
from tetris_agent._search_primitives import SearchPrimitives


def test_collision_all_slices_and_mutation():
    env=Tetris(); env.reset(seed=2)
    fast=SearchPrimitives(env)
    rng=np.random.default_rng(3)
    env.board[:20,4:14]=rng.integers(0,4,size=(20,10),dtype=np.uint8)
    for piece in env.tetrominoes:
        for k in range(4):
            rotated=env.rotate(piece,k%2==0)
            for y in range(-4,26):
                for x in range(-4,20):
                    try:
                        expected=env.collision(rotated,x,y)
                    except Exception as exc:
                        with pytest.raises(type(exc)):
                            fast.collision(rotated,x,y)
                    else:
                        actual=fast.collision(rotated,x,y)
                        assert type(actual) is type(expected)
                        assert actual==expected
    # Cache must never retain board values; matrix changes select a new entry.
    env.board[:]=0
    piece.matrix[:]=0
    assert fast.collision(piece,5,5)==env.collision(piece,5,5)


def test_rotation_exact_values_strides_and_aliases():
    env=Tetris(); env.reset(seed=42)
    for piece in env.tetrominoes:
        for k in range(4):
            piece=env.rotate(piece)
            for cw in (False,True):
                a,b=env.rotate(piece,cw),SearchPrimitives.rotate(piece,cw)
                assert a.id==b.id and a.color_rgb is b.color_rgb is piece.color_rgb
                assert np.array_equal(a.matrix,b.matrix)
                assert a.matrix.strides==b.matrix.strides
                assert np.shares_memory(b.matrix,piece.matrix)


def test_primitive_exception_restoration(monkeypatch):
    from tetris_agent.planner import TacticalPlanner
    from test_planner import state_signature
    env=Tetris(); env.reset(seed=42)
    before=state_signature(env)
    methods=(env.collision,env.rotate)
    real=SearchPrimitives.collision
    calls=0
    def broken(self,*args):
        nonlocal calls
        calls+=1
        if calls==70: raise RuntimeError('geometry failure')
        return real(self,*args)
    monkeypatch.setattr(SearchPrimitives,'collision',broken)
    with pytest.raises(RuntimeError,match='geometry failure'):
        TacticalPlanner().enumerate_reachable_targets(env)
    assert state_signature(env)==before
    assert (env.collision,env.rotate)==methods
    assert not {'collision','rotate','_get_obs'} & vars(env).keys()
