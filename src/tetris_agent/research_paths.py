"""Explicit roots for research artifacts kept outside the source checkout."""
import os
from pathlib import Path


def artifact_root(repo_root, explicit=None):
    """Explicit argument wins; otherwise use an absolute configured root.

    Without configuration the existing repo-local contract is unchanged. There
    is no legacy-path search or fallback when the selected root is unavailable.
    """
    if explicit is not None:
        return Path(explicit).resolve()
    configured = os.environ.get('TETRIS_ARTIFACT_ROOT')
    if configured:
        path = Path(configured)
        if not path.is_absolute():
            raise ValueError('TETRIS_ARTIFACT_ROOT must be an absolute path')
        return path.resolve()
    return Path(repo_root).resolve()
