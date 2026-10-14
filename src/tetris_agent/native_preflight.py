"""Shared fail-closed entrypoint gate for the preserved Native contract."""
from pathlib import Path
from .experiments import Registry, validate_environment

def native_preflight(repo_root=None):
    root = Path(repo_root) if repo_root else Path(__file__).resolve().parents[2]
    profile = Registry(root).profile('native_tetris_031_oracle_perf_v2')
    validate_environment(profile)
    return dict(profile=profile['id'], source_files=len(profile['native_environment']['source_hashes']), native_source_verified=True)
