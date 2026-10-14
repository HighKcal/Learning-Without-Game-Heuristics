"""Install the byte-exact Native overlay into an isolated venv only."""
import argparse
import hashlib
import importlib.util
import importlib.metadata
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]

def verify_tree(tree, hashes):
    actual = {p.relative_to(tree).as_posix() for p in tree.rglob('*.py')}
    if actual != set(hashes):
        raise RuntimeError('Native Python source inventory mismatch')
    for rel, expected in hashes.items():
        if hashlib.sha256((tree / rel).read_bytes()).hexdigest() != expected:
            raise RuntimeError('Native source SHA256 mismatch: ' + rel)

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--report', type=Path)
    args = p.parse_args()
    if args.report and args.report.exists():
        p.error('--report must be a new file')
    if sys.prefix == sys.base_prefix:
        p.error('A dedicated virtual environment is required')
    source = ROOT / 'vendor/native_tetris_031/tetris_gymnasium'
    manifest = json.loads((source.parent / 'manifest.json').read_text())
    verify_tree(source, manifest['source_hashes'])
    if importlib.metadata.version('tetris-gymnasium') != '0.3.1':
        raise RuntimeError('Install tetris-gymnasium==0.3.1 first')
    spec = importlib.util.find_spec('tetris_gymnasium')
    destination = Path(spec.origin).resolve().parent
    if not destination.is_relative_to(Path(sys.prefix).resolve()):
        raise RuntimeError('Refusing to modify a package outside this venv')
    # Before any write, reject an unknown package or unexpected extra Python source.
    current = dict(manifest['source_hashes'])
    current['envs/tetris.py'] = manifest['changes'][0]['upstream_sha256']
    try:
        verify_tree(destination, current)
    except RuntimeError:
        verify_tree(destination, manifest['source_hashes'])
    for rel in manifest['source_hashes']:
        shutil.copyfile(source / rel, destination / rel)
    verify_tree(destination, manifest['source_hashes'])
    report = dict(native_sources=19, native_sha256_verified=True,
                  upstream_wheel_sha256=manifest['wheel_sha256'],
                  destination=str(destination), source_hashes=manifest['source_hashes'])
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        with args.report.open('x') as stream:
            json.dump(report, stream, indent=2)
    print(json.dumps(report))

if __name__ == '__main__':
    main()
