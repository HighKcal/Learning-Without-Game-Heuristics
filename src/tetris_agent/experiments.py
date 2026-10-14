"""Exact registry selectors and fail-closed, read-only model resolution.

Paths are relative to explicit roots. Historical provenance is never an import
or artifact fallback. Loading this module does not load Torch or any checkpoint.
"""
from dataclasses import dataclass
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path, PureWindowsPath
import re
import yaml
from .research_paths import artifact_root as configured_artifact_root

MODES = ('research_geometric_top1', 'main_geometric_play_fallback',
         'research_oracle_greedy', 'main_oracle_play_witness')


class ResolutionError(ValueError):
    def __init__(self, code, message):
        self.code = code
        super().__init__(f'{code}: {message}')


def digest(path):
    h = sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def relative_path(root, value):
    if (not isinstance(value, str) or not value or '\\' in value
            or PureWindowsPath(value).drive or value.startswith('/')
            or any(c in value for c in ':*?"<>|')):
        raise ResolutionError('INVALID_PATH', repr(value))
    parts = value.split('/')
    if any(p in ('', '.', '..') or p.endswith((' ', '.'))
           or re.match(r'^(CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(?:\.|$)', p, re.I)
           for p in parts):
        raise ResolutionError('INVALID_PATH', value)
    root = Path(root).resolve()
    path = root.joinpath(*parts)
    # Reject junctions/symlinks even if their current targets remain in root.
    for parent in (path, *path.parents):
        if parent == root:
            break
        if parent.is_symlink() or (hasattr(parent, 'is_junction') and parent.is_junction()):
            raise ResolutionError('INVALID_PATH', f'Linked path: {value}')
    if not path.resolve().is_relative_to(root):
        raise ResolutionError('INVALID_PATH', value)
    return path


def _yaml(path):
    try:
        data = yaml.safe_load(Path(path).read_text(encoding='utf-8'))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise ResolutionError('INVALID_REGISTRY', str(exc)) from exc
    if not isinstance(data, dict) or data.get('schema_version') != 1:
        raise ResolutionError('INVALID_REGISTRY', 'Expected schema_version 1')
    return data


class Registry:
    def __init__(self, repo_root=None, *, artifact_root=None, index_path=None):
        self.repo_root = Path(repo_root or Path(__file__).resolve().parents[2]).resolve()
        self.artifact_root = configured_artifact_root(self.repo_root, artifact_root)
        self.data = _yaml(index_path or self.repo_root/'experiments/index.yaml')
        self.entries = {}
        for entry in self.data.get('experiments', []):
            name = entry.get('id')
            if not isinstance(name, str) or name in self.entries:
                raise ResolutionError('INVALID_REGISTRY', f'Duplicate/invalid ID: {name}')
            relative_path(self.artifact_root, entry['current_path'])
            artifacts = entry.get('checkpoints', [])
            ids = [a.get('artifact_id') for a in artifacts]
            if len(ids) != len(set(ids)) or any(not isinstance(i, str) for i in ids):
                raise ResolutionError('INVALID_REGISTRY', f'Duplicate artifact in {name}')
            for a in artifacts:
                relative_path(self.artifact_root, entry['current_path']+'/'+a['filename'])
                if not re.fullmatch('[0-9a-f]{64}', a.get('sha256', '')):
                    raise ResolutionError('INVALID_REGISTRY', f'Invalid SHA256 in {name}')
            self.entries[name] = entry

    def select(self, experiment_id, *, step=None, role=None):
        if experiment_id not in self.entries:
            raise ResolutionError('UNKNOWN_EXPERIMENT', repr(experiment_id))
        if step is not None and (type(step) is not int or step < 0):
            raise ResolutionError('INVALID_SELECTOR', 'step must be a nonnegative integer')
        if role not in (None, 'final', 'periodic') or (step is None and role is None):
            raise ResolutionError('INVALID_SELECTOR', 'Supply role or step')
        entry = self.entries[experiment_id]
        candidates = [a for a in entry['checkpoints'] if step is None or a['step'] == step]
        if role is not None:
            candidates = [a for a in candidates if a['role'] == role]
        elif any(a['role'] == 'periodic' for a in candidates):
            candidates = [a for a in candidates if a['role'] == 'periodic']
        if not candidates:
            raise ResolutionError('CHECKPOINT_NOT_FOUND', f'{experiment_id}, step={step}, role={role}')
        if len(candidates) != 1:
            raise ResolutionError('AMBIGUOUS_CHECKPOINT', f'{experiment_id}: {len(candidates)} matches')
        return entry, candidates[0]

    def profile(self, name):
        path = relative_path(self.repo_root, f'experiments/profiles/{name}.yaml')
        result = _yaml(path)
        if result.get('id') != name:
            raise ResolutionError('PROFILE_MISMATCH', name)
        return result


def validate_environment(profile):
    import tetris_gymnasium
    if (profile.get('observation_values') != 235 or profile.get('target_actions') != 800
            or profile.get('board') != {'width': 10, 'height': 20}
            or profile.get('gravity') != 'per_native_primitive_action'):
        raise ResolutionError('PROFILE_MISMATCH', '235/800, 20x10, native gravity required')
    native = profile['native_environment']
    if version(native['package']) != native['version']:
        raise ResolutionError('ENVIRONMENT_MISMATCH', 'Package version differs')
    sources = native.get('source_hashes')
    if not sources:
        raise ResolutionError('PENDING_COMPATIBILITY', 'Native source hashes unverified')
    root = Path(tetris_gymnasium.__file__).resolve().parent
    for rel, expected in sources.items():
        path = relative_path(root, rel)
        if not path.is_file() or digest(path) != expected:
            raise ResolutionError('ENVIRONMENT_MISMATCH', f'Native source differs: {rel}')


@dataclass(frozen=True)
class ResolvedCheckpoint:
    experiment_id: str
    artifact_id: str
    path: Path
    sha256: str
    step: int
    role: str
    mode: str
    profile: dict
    metadata: dict


def resolve_checkpoint(experiment_id, *, step=None, role=None, mode=None,
                       repo_root=None, artifact_root=None, registry=None):
    registry = registry or Registry(repo_root, artifact_root=artifact_root)
    entry, artifact = registry.select(experiment_id, step=step, role=role)
    profile = registry.profile(entry['environment_profile'])
    mode = mode or profile.get('default_inference_mode')
    compatibility = next((c for c in entry.get('runner_compatibility', [])
                          if c['mode'] == mode), None)
    if mode not in MODES or compatibility is None or compatibility['state'] == 'unsupported':
        raise ResolutionError('UNSUPPORTED_RUNNER', str(mode))
    if (compatibility['state'] != 'supported'
            or artifact.get('staging_compatibility') != 'supported'
            or profile.get('staging_compatibility') != 'supported'):
        raise ResolutionError('PENDING_COMPATIBILITY', f'{experiment_id}/{artifact["artifact_id"]}/{mode}')
    if artifact.get('availability') != 'local':
        raise ResolutionError('MODEL_UNAVAILABLE', str(artifact.get('availability')))
    path = relative_path(registry.artifact_root, entry['current_path']+'/'+artifact['filename'])
    if not path.is_file():
        raise ResolutionError('MODEL_NOT_INSTALLED', str(path))
    if digest(path) != artifact['sha256']:
        raise ResolutionError('HASH_MISMATCH', str(path))
    validate_environment(profile)
    import torch
    from .rl.dqn import QNetwork
    try:
        payload = torch.load(path, map_location='cpu', weights_only=True)
        contract = payload.get('checkpoint_contract', {})
        if (payload.get('schema_version') != 1 or contract.get('kind') != 'training_snapshot'
                or contract.get('exact_resume_supported') is not False):
            raise ValueError('Only schema-v1 training snapshots support production inference')
        if type(payload.get('decision_count')) is not int or payload['decision_count'] != artifact['step']:
            raise ValueError('Internal decision_count differs from selector')
        mask = payload.get('reachability_mask_mode', payload.get('config', {}).get('reachability_mask_mode', 'geometric'))
        expected = 'oracle' if 'oracle' in mode else 'geometric'
        if mask != expected:
            raise ValueError(f'Checkpoint mask {mask} differs from runner {expected}')
        if expected == 'oracle' and payload.get('planner_semantics_version') != 'native-0.3.1-bfs-v1':
            raise ValueError('Unknown Oracle planner semantics')
        with torch.random.fork_rng(devices=[]):
            network = QNetwork()
        network.load_state_dict(payload['online_state_dict'], strict=True)
        if any(not bool(torch.isfinite(t).all()) for t in network.state_dict().values()):
            raise ValueError('Non-finite model weights')
        with torch.inference_mode():
            q = network(torch.zeros((1, 235)))
        if q.shape != (1, 800) or not bool(torch.isfinite(q).all()):
            raise ValueError('Expected finite 235-input/800-output model')
    except Exception as exc:
        raise ResolutionError('CHECKPOINT_METADATA_MISMATCH', str(exc)) from exc
    metadata = {k: payload.get(k) for k in ('schema_version', 'decision_count', 'config',
                'reachability_mask_mode', 'oracle_implementation_version', 'planner_semantics_version')}
    return ResolvedCheckpoint(experiment_id, artifact['artifact_id'], path, artifact['sha256'],
                              artifact['step'], artifact['role'], mode, profile, metadata)
