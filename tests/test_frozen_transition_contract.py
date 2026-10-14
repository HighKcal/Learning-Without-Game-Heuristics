import numpy as np
from tetris_agent.rl.replay import Transition
from tetris_agent.rl.state import SemanticState

def test_transition_owns_masks_without_replay_mutation():
    from tetris_gymnasium.envs.tetris import Tetris
    from tetris_agent.rl.state import read_state, geometric_mask
    env=Tetris(gravity=True)
    try:
        env.reset(seed=123)
        state=read_state(env)
        mask=geometric_mask(state)
        expected=mask.copy()
        action=int(np.flatnonzero(mask)[0])
        value=Transition(state,action,-.05,state,False,True,mask,mask)
        mask[:]=False
        assert np.array_equal(value.state_reachable_mask,expected)
        assert not value.state_reachable_mask.flags.writeable
        copy=value.copy()
        assert np.array_equal(copy.state_reachable_mask,expected)
        assert not np.shares_memory(copy.state.board,state.board)
    finally:env.close()
