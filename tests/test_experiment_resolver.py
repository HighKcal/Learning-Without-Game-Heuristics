from copy import deepcopy
import json
from pathlib import Path
import pytest
import torch
import yaml
from tetris_agent.experiments import Registry, ResolutionError, digest, relative_path, resolve_checkpoint
from tetris_agent.rl.network import QNetwork
from tetris_agent.inference import FrozenInference

ROOT=Path(__file__).resolve().parents[1]

@pytest.fixture
def context(tmp_path):
    folder=tmp_path/'project'
    (folder/'experiments/profiles').mkdir(parents=True)
    (folder/'runs/fixture').mkdir(parents=True)
    native_sources=Registry(ROOT).profile('native_tetris_031_oracle_perf_v2')['native_environment']['source_hashes']
    profile=dict(schema_version=1,id='fixture',staging_compatibility='supported',observation_values=235,
        target_actions=800,board=dict(width=10,height=20),gravity='per_native_primitive_action',
        default_inference_mode='research_geometric_top1',native_environment=dict(package='tetris-gymnasium',
        version='0.3.1',source_hashes=native_sources))
    payload=dict(schema_version=1,checkpoint_contract=dict(kind='training_snapshot',exact_resume_supported=False),
        decision_count=12,config={},online_state_dict=QNetwork().state_dict())
    path=folder/'runs/fixture/model.pt';torch.save(payload,path)
    artifact=dict(artifact_id='periodic12',role='periodic',step=12,filename='model.pt',sha256=digest(path),
        availability='local',staging_compatibility='supported')
    entry=dict(id='fixture',current_path='runs/fixture',environment_profile='fixture',checkpoints=[artifact],
        runner_compatibility=[dict(mode='research_geometric_top1',state='supported')])
    return folder,entry,profile,payload

def registry(context):
    folder,entry,profile,_=context
    (folder/'experiments/index.yaml').write_text(yaml.safe_dump(dict(schema_version=1,experiments=[entry])),encoding='utf-8')
    (folder/'experiments/profiles/fixture.yaml').write_text(yaml.safe_dump(profile),encoding='utf-8')
    return Registry(folder)

def test_exact_role_step_final_and_periodic_contract(context):
    folder,e,_,_=context
    final=deepcopy(e['checkpoints'][0]);final.update(artifact_id='final',role='final')
    e['checkpoints'].append(final)
    reg=registry(context)
    assert reg.select('fixture',step=12)[1]['artifact_id']=='periodic12'
    assert resolve_checkpoint('fixture',role='final',registry=reg).role=='final'
    assert resolve_checkpoint('fixture',step=12,role='periodic',registry=reg).step==12

@pytest.mark.parametrize('case,code',[
 ('id','UNKNOWN_EXPERIMENT'),('step','CHECKPOINT_NOT_FOUND'),('ambiguous','AMBIGUOUS_CHECKPOINT'),
 ('pending','PENDING_COMPATIBILITY'),('artifact_pending','PENDING_COMPATIBILITY'),
 ('profile_pending','PENDING_COMPATIBILITY'),('unsupported','UNSUPPORTED_RUNNER'),
 ('unavailable','MODEL_UNAVAILABLE'),('missing','MODEL_NOT_INSTALLED'),('wrong_root','MODEL_NOT_INSTALLED'),
 ('hash','HASH_MISMATCH'),('count','CHECKPOINT_METADATA_MISMATCH'),('mask','CHECKPOINT_METADATA_MISMATCH'),
 ('architecture','CHECKPOINT_METADATA_MISMATCH'),('nonfinite','CHECKPOINT_METADATA_MISMATCH'),
 ('native_hash','ENVIRONMENT_MISMATCH'),('dimensions','PROFILE_MISMATCH')])
def test_explicit_failure_without_fallback(context,case,code,tmp_path):
    folder,e,p,payload=context
    args=dict(experiment_id='fixture',step=12)
    a=e['checkpoints'][0]
    path=folder/'runs/fixture/model.pt'
    if case=='id':args['experiment_id']='other'
    elif case=='step':args['step']=13
    elif case=='ambiguous':
        duplicate=deepcopy(a);duplicate['artifact_id']='other';e['checkpoints'].append(duplicate)
    elif case=='pending':e['runner_compatibility'][0]['state']='pending'
    elif case=='artifact_pending':a['staging_compatibility']='pending'
    elif case=='profile_pending':p['staging_compatibility']='pending'
    elif case=='unsupported':e['runner_compatibility'][0]['state']='unsupported'
    elif case=='unavailable':a['availability']='remote'
    elif case=='missing':path.unlink()
    elif case=='hash':path.write_bytes(b'tampered')
    elif case=='native_hash':p['native_environment']['source_hashes']['envs/tetris.py']='0'*64
    elif case=='dimensions':p['observation_values']=234
    elif case in ('count','mask','architecture','nonfinite'):
        if case=='count':payload['decision_count']=13
        if case=='mask':payload['reachability_mask_mode']='oracle'
        if case=='architecture':payload['online_state_dict']={}
        if case=='nonfinite':next(iter(payload['online_state_dict'].values())).view(-1)[0]=float('nan')
        torch.save(payload,path);a['sha256']=digest(path)
    reg=registry(context)
    if case=='wrong_root':reg.artifact_root=tmp_path/'absent'
    with pytest.raises(ResolutionError) as error:resolve_checkpoint(**args,registry=reg)
    assert error.value.code==code

@pytest.mark.parametrize('path',['../model.pt','runs/../model.pt','/absolute.pt','C:/model.pt',
    'C:\\model.pt','//server/model.pt','runs\\model.pt','runs//model.pt','runs/./model.pt',
    'runs/model.pt:stream','runs/model.pt.','runs/CON.pt','',None])
def test_malformed_relative_path_rejected(tmp_path,path):
    with pytest.raises(ResolutionError,match='INVALID_PATH'):relative_path(tmp_path,path)

def test_duplicate_id_rejected(context):
    reg=registry(context)
    data=deepcopy(reg.data);data['experiments'].append(data['experiments'][0])
    (reg.repo_root/'experiments/index.yaml').write_text(yaml.safe_dump(data),encoding='utf-8')
    with pytest.raises(ResolutionError,match='Duplicate'):Registry(reg.repo_root)

def test_final_only_unique_numeric(context):
    context[1]['checkpoints'][0]['role']='final'
    assert registry(context).select('fixture',step=12)[1]['role']=='final'

@pytest.mark.parametrize('step',[True,12.0,-1,'12'])
def test_invalid_step_rejected(context,step):
    with pytest.raises(ResolutionError,match='INVALID_SELECTOR'):registry(context).select('fixture',step=step)

def test_facade_freezes_model_and_rejects_wrong_gravity(context):
    from tetris_gymnasium.envs.tetris import Tetris
    session=FrozenInference('fixture',step=12,registry=registry(context))
    assert not session.network.training
    assert all(not p.requires_grad and p.grad is None for p in session.network.parameters())
    env=Tetris(gravity=False)
    try:
        with pytest.raises(ResolutionError,match='ENVIRONMENT_MISMATCH'):
            session.run(env,seed=123,limit=1)
    finally:env.close()

def test_facade_cpu_episode_has_no_weight_change(context):
    from tetris_gymnasium.envs.tetris import Tetris
    session=FrozenInference('fixture',step=12,registry=registry(context))
    before={k:v.clone() for k,v in session.network.state_dict().items()}
    env=Tetris(gravity=True)
    try:result=session.run(env,seed=123,limit=2)
    finally:env.close()
    assert result.diagnostic is None and result.decision_count==2
    assert all(torch.equal(before[k],v) for k,v in session.network.state_dict().items())
    assert all(p.grad is None for p in session.network.parameters())
