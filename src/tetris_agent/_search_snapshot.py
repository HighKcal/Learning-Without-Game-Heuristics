"""Private owned branch workspace for the audited native search.

Every saved frontier state still comes from native get_state(). The workspace
is never enqueued or exposed. Its RNG can be reused because every transition
restores the complete bit-generator state before native step (including spawn
and bag reshuffle). No installed package is modified.
"""
from copy import copy
from collections import deque

import numpy as np
from gymnasium.spaces import Discrete
from tetris_gymnasium.envs.tetris import TetrisState
from tetris_gymnasium.components.tetromino import Tetromino
from tetris_gymnasium.components.tetromino_holder import TetrominoHolder
from tetris_gymnasium.components.tetromino_queue import TetrominoQueue
from tetris_gymnasium.components.tetromino_randomizer import BagRandomizer, Randomizer
from tetris_gymnasium.mappings.actions import ActionsMapping
from tetris_gymnasium.mappings.rewards import RewardsMapping


_COMPONENT_METHODS = tuple((cls, name, method)
    for cls in (Randomizer, BagRandomizer, Tetromino, TetrominoHolder, TetrominoQueue)
    for name, method in vars(cls).items() if callable(method))
_NATIVE_CONTAINS = Discrete.contains


def can_reuse_randomizer(base):
    """Called only with the Planner's native Tetris/observation guard passing.

    Unknown components and method overrides retain the original copy path.
    Strict instance fields also reject hidden state on native component types.
    """
    r = base.randomizer
    if (type(r) is not BagRandomizer
            or type(base.actions) is not ActionsMapping
            or not all(type(v) is int for v in vars(base.actions).values())
            or type(base.rewards) is not RewardsMapping
            or not all(type(v) in (int,float) for v in vars(base.rewards).values())
            or set(vars(r)) != {'size','rng','bag','index'}
            or type(r.bag) is not np.ndarray
            or type(r.rng) is not np.random.Generator
            or type(r.rng.bit_generator) is not np.random.PCG64
            or base.queue.randomizer is not r
            or type(base.action_space) is not Discrete
            or 'contains' in vars(base.action_space)
            or Discrete.contains is not _NATIVE_CONTAINS
            or type(base.queue.queue) is not deque
            or type(base.holder.queue) is not deque
            or set(vars(base.queue)) != {'randomizer','queue','size'}
            or set(vars(base.holder)) != {'queue','size'}):
        return False
    pieces = (*base.tetrominoes, base.active_tetromino, *base.holder.queue)
    return (all(type(p.color_rgb) is list for p in pieces)
            and all(getattr(cls,name) is method for cls,name,method in _COMPONENT_METHODS))


class BranchRestorer:
    """Reusable RNG; all other native snapshot ownership rules are retained."""
    def __init__(self, original):
        self.randomizer = copy(original.randomizer)

    def __call__(self, base, state):
        source, owned = state.randomizer, self.randomizer
        owned.size = source.size
        owned.index = source.index
        owned.bag = source.bag.copy()
        owned.rng.bit_generator.state = source.rng.bit_generator.state
        base.set_state(TetrisState(
            board=state.board.copy(), active_tetromino=copy(state.active_tetromino),
            x=state.x, y=state.y, queue=state.queue.copy(owned),
            holder=copy(state.holder), randomizer=owned,
            has_swapped=state.has_swapped, game_over=state.game_over, score=state.score))
