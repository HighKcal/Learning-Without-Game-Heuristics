from dataclasses import asdict,replace
import importlib
from pathlib import Path
import sys

import numpy as np
import pytest
import torch
from tetris_agent.rl._oracle_validation import geometric_subsets
from tetris_agent.rl.config import TrainingConfig
from tetris_agent.rl.dqn import DQNTrainer
from tetris_agent.rl.evaluation import load_dataset
from tetris_agent.rl.oracle import OracleInvariantError
from tetris_agent.rl.replay import Transition
from tetris_agent.rl.state import geometric_mask


def records():
    return load_dataset(Path(__file__).parents[1]/'evaluation/fixed_eval_final_v1')[0]


def test_every_geometric_slot_and_mixed_subsets():
    samples=records()
    one_hot=np.eye(800,dtype=bool)
    rng=np.random.default_rng(551)
    for record in samples[::4]:
        state=record.state
        expected=geometric_mask(state)
        assert np.array_equal(geometric_subsets([state]*800,one_hot),expected)
    states=[r.state for r in samples]
    for density in (0.,.001,.01,.5,1.):
        masks=rng.random((len(states),800)) < density
        expected=[not np.any(m & ~geometric_mask(s)) for s,m in zip(states,masks)]
        assert np.array_equal(geometric_subsets(states,masks),expected)
    masks=np.stack([geometric_mask(s) for s in states])
    assert geometric_subsets(states,masks).all()


@pytest.mark.parametrize('device',['cpu','cuda'])
@pytest.mark.parametrize('counter',[0,4999])
def test_original_vs_batched_update_and_sync_bit_exact(device,counter):
    if device=='cuda' and not torch.cuda.is_available(): pytest.skip('CUDA unavailable')
    sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
    from oracle_reference import baseline,weights_digest
    baseline()
    original=importlib.import_module('_performance_baseline.rl.dqn').DQNTrainer
    cfg=TrainingConfig(4,device=device,batch_size=4,reachability_mask_mode='oracle')
    a,b=original(cfg),DQNTrainer(cfg)
    batch=[]
    for i,r in enumerate(records()[:4]):
        mask=geometric_mask(r.state)
        batch.append(Transition(r.state,int(np.flatnonzero(mask)[0]),float(i),r.state,
                                i==2,i==3,mask,None if i==2 else mask))
    a.gradient_updates=b.gradient_updates=counter
    for _ in range(3):
        assert asdict(a._update(batch,.05))==asdict(b._update(batch,.05))
        assert weights_digest(a)==weights_digest(b)


def test_revalidation_detects_replaced_replay_masks_and_states():
    r=records()[0]; state=r.state; mask=geometric_mask(state)
    action=int(np.flatnonzero(mask)[0])
    t=Transition(state,action,0.,state,False,False,mask,mask)
    invalid=mask.copy(); invalid[np.flatnonzero(~mask)[0]]=True
    bads=[replace(t,state_reachable_mask=None),replace(t,next_state_reachable_mask=None),
          replace(t,state_reachable_mask=invalid),replace(t,next_state_reachable_mask=invalid),
          replace(t,next_state_reachable_mask=np.zeros(800,dtype=bool)),
          replace(t,action_id=int(np.flatnonzero(~mask)[0]))]
    from tetris_agent.rl.state import SemanticState
    filled=SemanticState(np.ones((20,10),dtype=np.uint8),state.current_piece,state.next_queue)
    bads += [replace(t,state=filled),replace(t,next_state=filled)]
    for bad in bads:
        with pytest.raises(OracleInvariantError): DQNTrainer._oracle_next_masks([t,bad,t])
    assert np.array_equal(DQNTrainer._oracle_next_masks([t])[0],mask)
    terminal=replace(t,terminated=True,next_state_reachable_mask=None)
    assert not DQNTrainer._oracle_next_masks([terminal])[0].any()


def test_custom_scalar_validator_is_not_bypassed(monkeypatch):
    r=records()[0]; mask=geometric_mask(r.state)
    t=Transition(r.state,int(np.flatnonzero(mask)[0]),0.,r.state,False,False,mask,mask)
    trainer=DQNTrainer(TrainingConfig(1,device='cpu',reachability_mask_mode='oracle'))
    def custom(t): raise RuntimeError('custom validator')
    monkeypatch.setattr(trainer,'_oracle_next_mask',custom)
    with pytest.raises(RuntimeError,match='custom validator'):
        trainer._update([t],.05)
