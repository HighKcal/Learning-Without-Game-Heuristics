"""Run selected source with explicit dependencies and a separate artifact root.

Uses the chosen interpreter's installed packages. Optional dependency roots are
provided explicitly; no staging, historical source, or copied runtime is found
automatically. All script arguments follow the project-relative target.
"""
import argparse
import os
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifact-root', type=Path)
    parser.add_argument('--dependency-root', type=Path, action='append', default=[])
    parser.add_argument('--module', action='store_true')
    parser.add_argument('target', help='Project-relative script or installed module name')
    parser.add_argument('arguments', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    dependencies = [path.resolve() for path in args.dependency_root]
    if any(not path.is_dir() for path in dependencies):
        parser.error('Every explicit dependency root must be an existing directory')
    if args.artifact_root is not None:
        root = args.artifact_root.resolve()
        if not root.is_dir():
            parser.error('--artifact-root must be an existing directory')
        os.environ['TETRIS_ARTIFACT_ROOT'] = str(root)
    sys.dont_write_bytecode = True
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    sys.path[:0] = [str(ROOT / 'src'), *map(str, dependencies)]
    if args.module:
        sys.argv = [args.target, *args.arguments]
        runpy.run_module(args.target, run_name='__main__')
    else:
        target = (ROOT / args.target).resolve()
        if not target.is_relative_to(ROOT) or not target.is_file():
            parser.error('Script must exist inside this source checkout')
        sys.path.insert(0, str(target.parent))
        sys.argv = [str(target), *args.arguments]
        runpy.run_path(str(target), run_name='__main__')


if __name__ == '__main__':
    main()
