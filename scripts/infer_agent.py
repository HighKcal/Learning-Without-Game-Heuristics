"""Registry-based frozen inference using the shared GUI boundary."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import torch
from tetris_gymnasium.envs.tetris import Tetris
from tetris_agent.experiments import MODES, ResolutionError, digest
from tetris_agent.inference import FrozenInference

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--experiment',required=True)
    p.add_argument('--step',type=int)
    p.add_argument('--role',choices=('periodic','final'))
    p.add_argument('--mode',choices=MODES)
    p.add_argument('--artifact-root',type=Path,
                   help='Explicit private data root; overrides TETRIS_ARTIFACT_ROOT')
    p.add_argument('--device',choices=('cpu','cuda'),default='cpu')
    p.add_argument('--seed',type=int,default=100000)
    p.add_argument('--limit',type=int,default=6)
    p.add_argument('--output',type=Path,required=True,help='new output directory')
    args=p.parse_args(argv)
    if args.device=='cpu':torch.set_num_threads(1)
    if args.limit<=0 or args.output.exists():p.error('Positive limit and new output directory required')
    try:
        session=FrozenInference(args.experiment,step=args.step,role=args.role,mode=args.mode,
            device=args.device,repo_root=ROOT,artifact_root=args.artifact_root)
    except ResolutionError as exc:
        print(str(exc),file=sys.stderr);return 2
    weights={k:v.clone() for k,v in session.network.state_dict().items()}
    env=Tetris(gravity=True)
    try:result=session.run(env,seed=args.seed,limit=args.limit)
    finally:env.close()
    stats=result.summary() if hasattr(result,'summary') else asdict(result)
    report=dict(experiment_id=session.checkpoint.experiment_id,artifact_id=session.checkpoint.artifact_id,
        mode=session.checkpoint.mode,device=args.device,seed=args.seed,epsilon=0,training=False,
        optimizer_steps=0,replay_mutations=0,checkpoint_unchanged=digest(session.checkpoint.path)==session.checkpoint.sha256,
        weights_unchanged=all(torch.equal(v,weights[k]) for k,v in session.network.state_dict().items()),
        no_gradients=all(p.grad is None for p in session.network.parameters()),result=stats)
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/'summary.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(report,allow_nan=False))
    return int(bool(stats['diagnostic']) or not all(report[k] for k in ('checkpoint_unchanged','weights_unchanged','no_gradients')))

if __name__=='__main__':raise SystemExit(main())
