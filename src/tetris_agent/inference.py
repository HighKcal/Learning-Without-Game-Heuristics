"""Shared frozen inference boundary for CLI and the planned GUI.

Caller owns the environment and serializes access. No trainer/optimizer/replay
is constructed. Research geometric U self-loops and deployment fallback remain
separate explicit modes; both Oracle modes reuse one BFS and its witnesses.
"""
import torch
from .experiments import resolve_checkpoint, digest, ResolutionError, validate_environment
from .play import FrozenQRankingPolicy, PlayRunner
from .rl.evaluation import load_checkpoint_network
from .rl.policy import EpsilonGreedyPolicy
from .rl.runner import RLRunner


class FrozenInference:
    def __init__(self, experiment_id, *, step=None, role=None, mode=None,
                 device='cpu', repo_root=None, artifact_root=None, registry=None):
        self.checkpoint = resolve_checkpoint(experiment_id, step=step, role=role,
            mode=mode, repo_root=repo_root, artifact_root=artifact_root, registry=registry)
        if device not in ('cpu', 'cuda') or (device == 'cuda' and not torch.cuda.is_available()):
            raise ResolutionError('DEVICE_UNAVAILABLE', device)
        if device not in self.checkpoint.profile.get('validated_devices', ['cpu']):
            raise ResolutionError('DEVICE_UNVALIDATED', device)
        # Revalidate hash immediately before loading; no path/glob substitution.
        if digest(self.checkpoint.path) != self.checkpoint.sha256:
            raise ResolutionError('HASH_MISMATCH', str(self.checkpoint.path))
        self.network = load_checkpoint_network(self.checkpoint.path, device).eval().requires_grad_(False)

    @torch.inference_mode()
    def run(self, env, *, seed, limit, on_decision=None):
        validate_environment(self.checkpoint.profile)
        base=env.unwrapped
        if (base.width,base.height)!=(10,20) or not base.gravity_enabled:
            raise ResolutionError('ENVIRONMENT_MISMATCH', '20x10 native gravity=True required')
        if self.checkpoint.mode == 'main_geometric_play_fallback':
            return PlayRunner(FrozenQRankingPolicy(self.network)).run(
                env, env_seed=seed, max_placements=limit, on_decision=on_decision)
        if on_decision is not None:
            raise ValueError('on_decision is currently supported only by geometric Play')
        mask = 'oracle' if 'oracle' in self.checkpoint.mode else 'geometric'
        return RLRunner(EpsilonGreedyPolicy(self.network), reachability_mask_mode=mask).run(
            env, env_seed=seed, max_decisions=limit)
