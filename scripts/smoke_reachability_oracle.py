"""6K clean DDQN smoke and full replay/snapshot audit; no strategy claims."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import numpy as np
import torch
from tetris_gymnasium.envs.tetris import Tetris
from tetris_agent.rl.artifacts import ArtifactConfig
from tetris_agent.rl.config import TrainingConfig
from tetris_agent.rl.dqn import DQNTrainer
from tetris_agent.rl.evaluation import load_checkpoint_network
from tetris_agent.rl.state import encode_states
from tetris_agent.rl.training import train


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--decision-budget', type=int, default=6000)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    args = parser.parse_args()
    if not 5001 <= args.decision_budget <= 10000:
        parser.error('Smoke budget must be 5001..10000 to exercise learning.')
    if args.output_dir.exists():
        parser.error('--output-dir must be new')
    trainer = DQNTrainer(TrainingConfig(args.decision_budget, device=args.device,
                                      reachability_mask_mode='oracle'))
    env = Tetris(gravity=True)
    learning_rows = []
    def learning(metrics):
        row = asdict(metrics)
        assert all(np.isfinite(value) for value in row.values()), 'Non-finite loss/Q/gradient metric'
        learning_rows.append(row)
    try:
        summary = train(env, trainer, episode_decision_limit=10000, on_learning=learning,
                        artifacts=ArtifactConfig(args.output_dir, 100, 5000))
    finally:
        env.close()
    assert summary.status == 'completed'
    assert trainer.gradient_updates == args.decision_budget - 4999
    assert sum(e.selected_physical_u_count + e.unreachable_count for e in summary.episodes) == 0
    assert all(e.diagnostic is None for e in summary.episodes)
    for transition in trainer.replay._items:
        trainer._oracle_next_mask(transition)
        assert not transition.state_reachable_mask.flags.writeable
        if not transition.terminated:
            assert not transition.next_state_reachable_mask.flags.writeable
    snapshot_path = args.output_dir / 'checkpoint_final.pt'
    snapshot = torch.load(snapshot_path, map_location='cpu', weights_only=True)
    assert snapshot['schema_version'] == 1 and snapshot['reachability_mask_mode'] == 'oracle'
    assert snapshot['checkpoint_contract']['exact_resume_supported'] is False
    restored = load_checkpoint_network(snapshot_path, 'cpu')
    for key, value in trainer.online.state_dict().items():
        assert torch.equal(value.cpu(), restored.state_dict()[key])
    # Optimizer loading is inspection only, not an exact-resume claim.
    inspected_optimizer = torch.optim.Adam(restored.parameters(), lr=1e-4)
    inspected_optimizer.load_state_dict(snapshot['optimizer_state_dict'])
    sample = trainer.replay.sample(128)
    with torch.no_grad():
        q = restored(encode_states([t.state for t in sample]))
    assert bool(torch.isfinite(q).all())
    report = dict(status=summary.status, decision_count=summary.decision_count,
        stored_experiences=trainer.stored_experiences, gradient_updates=trainer.gradient_updates,
        episode_count=len(summary.episodes), selected_physical_u=0,
        replay_oracle_transitions_audited=len(trainer.replay), finite_learning_updates=len(learning_rows),
        loss_min=min(r['loss'] for r in learning_rows), loss_max=max(r['loss'] for r in learning_rows),
        full_q_values_audited=q.numel(), full_q_finite=True,
        checkpoint_schema_version=1, checkpoint_weights_roundtrip_exact=True,
        optimizer_state_load_verified=True, exact_resume_supported=False,
        exact_resume_policy='Existing snapshot contract preserved; no replay/RNG/env resume API.',
        throughput=summary.throughput, last_learning=learning_rows[-1])
    (args.output_dir / 'smoke_audit.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n',
                                                  encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
