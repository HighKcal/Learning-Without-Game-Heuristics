"""Read-only Python/Native/CPU-or-CUDA runtime readiness check."""
import argparse
from importlib.metadata import version
import json
from pathlib import Path
import platform
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import torch
from tetris_agent.native_preflight import native_preflight

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--device',choices=('cpu','cuda'),default='cpu')
    p.add_argument('--output',type=Path)
    args=p.parse_args()
    if args.output and args.output.exists(): p.error('Output must be a new file')
    native=native_preflight()
    if args.device=='cuda' and not torch.cuda.is_available(): p.error('CUDA unavailable')
    x=torch.arange(16,dtype=torch.float32,device=args.device).reshape(4,4)
    total=float((x@x.T).sum().item())
    assert total==3680.0
    report=dict(python=platform.python_version(),torch=str(torch.__version__),
          cuda_runtime=torch.version.cuda,cuda_available=torch.cuda.is_available(),
          device=args.device,gpu=torch.cuda.get_device_name() if args.device=='cuda' else None,
          tensor_sum=total,native=native,
          packages={name:version(name) for name in ['numpy','gymnasium','opencv-python','PyYAML','pytest','tetris-gymnasium']})
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        with args.output.open('x') as stream: json.dump(report,stream,indent=2)
    print(json.dumps(report))

if __name__=='__main__': main()
