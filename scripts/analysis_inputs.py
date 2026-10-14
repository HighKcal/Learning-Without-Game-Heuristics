"""Read-only path adapter for five preserved canonical analyses."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from tetris_agent.artifact_catalog import ANALYSIS_EXPERIMENTS,analysis_inputs
def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--name',choices=tuple(ANALYSIS_EXPERIMENTS),required=True)
    p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    p.add_argument('--artifact-root',type=Path,
                   help='Explicit private data root; overrides TETRIS_ARTIFACT_ROOT')
    args=p.parse_args(argv)
    print(json.dumps(analysis_inputs(args.name,root=args.root,
                                    artifact_root=args.artifact_root),indent=2))
    return 0
if __name__=='__main__':raise SystemExit(main())
