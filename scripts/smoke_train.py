"""Bounded fresh RL verification; production hyperparameters remain unchanged."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import math
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import torch
from tetris_gymnasium.envs.tetris import Tetris
from tetris_agent.native_preflight import native_preflight
from tetris_agent.rl.config import TrainingConfig
from tetris_agent.rl.dqn import DQNTrainer
from tetris_agent.rl.artifacts import ArtifactConfig
from tetris_agent.rl.training import train
from tetris_agent.rl.evaluation import load_checkpoint_network

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=('cpu','cuda'), required=True)
    parser.add_argument('--reachability-mask', choices=('geometric','oracle'), required=True)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error('Output must be a new directory')
    preflight = native_preflight()
    torch.set_num_threads(1)
    production = TrainingConfig(decision_budget=32)
    config = TrainingConfig(decision_budget=32, seed=args.seed, device=args.device,
             reachability_mask_mode=args.reachability_mask, learning_starts=4,
             batch_size=4, replay_capacity=64, target_sync_updates=8)
    trainer = DQNTrainer(config)
    before = {k:v.detach().cpu().clone() for k,v in trainer.online.state_dict().items()}
    losses=[]
    env=Tetris(gravity=True)
    try:
        summary=train(env, trainer, episode_decision_limit=16,
                      on_learning=lambda m: losses.append(m.loss),
                      artifacts=ArtifactConfig(args.output_dir,4,16))
    finally:
        env.close()
    deltas={k:float((v.detach().cpu()-before[k]).abs().max()) for k,v in trainer.online.state_dict().items()}
    checkpoints=[]
    for path in sorted(args.output_dir.glob('checkpoint_*.pt')):
        loaded=load_checkpoint_network(path,'cpu')
        payload=torch.load(path,map_location='cpu',weights_only=True)
        equal=all(torch.equal(v.detach().cpu(),loaded.state_dict()[k]) for k,v in trainer.online.state_dict().items()) if path.name=='checkpoint_final.pt' else all(torch.equal(v,loaded.state_dict()[k]) for k,v in payload['online_state_dict'].items())
        checkpoints.append(dict(name=path.name,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),weights_reloaded_exactly=equal,gradient_updates=payload['gradient_updates']))
    optimizer_steps=[int(state['step'].item()) for state in trainer.optimizer.state.values()]
    report=dict(native=preflight,device=args.device,mask=args.reachability_mask,
                production_learning_starts=production.learning_starts,smoke_config=asdict(config),
                status=summary.status,decision_count=summary.decision_count,
                stored_experiences=trainer.stored_experiences,replay_size=len(trainer.replay),
                gradient_updates=trainer.gradient_updates,target_sync_count=trainer.target_sync_count,
                losses=losses,parameter_max_abs_changes=deltas,optimizer_steps=optimizer_steps,
                checkpoints=checkpoints)
    passed=(summary.status=='completed' and trainer.stored_experiences==32 and
            trainer.gradient_updates==29 and len(losses)==29 and all(math.isfinite(v) for v in losses)
            and max(deltas.values())>0 and all(v==29 for v in optimizer_steps)
            and all(c['weights_reloaded_exactly'] for c in checkpoints)
            and production.learning_starts==5000)
    report['passed']=passed
    (args.output_dir/'smoke_evidence.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('losses','parameter_max_abs_changes','checkpoints','smoke_config')}))
    return 0 if passed else 1

if __name__=='__main__':
    raise SystemExit(main())
