"""Frozen census, independent pre-change Planner equivalence, and timings."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys
from time import perf_counter
import tracemalloc
from types import ModuleType

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import numpy as np
from tetris_gymnasium.envs.tetris import Tetris
from tetris_agent.planner import TacticalPlanner
from tetris_agent.rl.evaluation import load_dataset, dataset_fingerprint
from tetris_agent.rl.oracle import compute_boundary, restore_semantic_boundary, oracle_metadata
from tetris_agent.rl.state import SPACE, PIECE_NAMES


def baseline_planner(ref, source_path=None):
    # Independent, unmodified source at the frozen base ref, never the new core.
    root = Path(__file__).resolve().parents[1]
    if source_path is not None:
        source = Path(source_path).read_text(encoding='utf-8')
    elif (root / '.git').exists():
        source = subprocess.check_output(['git', 'show', f'{ref}:src/tetris_agent/planner.py'], cwd=root,text=True)
    else:
        raise ValueError('Exported staging requires --baseline-source; source provenance is recorded by SHA256')
    module = ModuleType('tetris_agent._oracle_baseline')
    sys.modules[module.__name__] = module
    exec(compile(source, f'{ref}:planner.py', 'exec'), module.__dict__)
    return module.TacticalPlanner(), sha256(source.encode()).hexdigest()


def distribution(values):
    return dict(count=len(values), mean=float(np.mean(values)),
                p95=float(np.quantile(values, .95)), maximum=float(max(values)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    parser.add_argument('--dataset', type=Path, default=root / 'evaluation/fixed_eval_final_v1')
    parser.add_argument('--baseline-source', type=Path,default=root / 'archive/worktrees/main/source/src/tetris_agent/planner.py')
    parser.add_argument('--baseline-ref', default='0a07550')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('--output must be a new file')
    records, metadata = load_dataset(args.dataset)
    if metadata['tetris_gymnasium_version'] != '0.3.1' or metadata.get('status') != 'completed':
        raise ValueError('Corpus must be completed and use audited native 0.3.1.')
    baseline, source_hash = baseline_planner(args.baseline_ref,args.baseline_source)
    planner, env = TacticalPlanner(), Tetris(gravity=True)
    env.reset(seed=42)
    rows, census, times, paired_benchmark = [], [], [], []
    reachable_plan_times, unreachable_plan_times = [], []
    cross_checks = 0
    try:
        for piece_id, piece_name in enumerate(PIECE_NAMES):
            env.reset(seed=42)
            env.active_tetromino = env.tetrominoes[piece_id]
            env.reset_tetromino_position()
            value = compute_boundary(env, planner)
            census.append(dict(piece=piece_name, geometric=value.geometric_count,
                               reachable=int(value.mask.sum()), visited=value.search.visited_states))
            for action in np.flatnonzero(value.mask):
                old = baseline.plan(env, SPACE.decode(int(action)))
                assert old.reachable
                witness = value.plan_for(int(action))
                assert (old.actions, old.needs_hard_drop) == (witness.actions, witness.needs_hard_drop)
                cross_checks += 1
        assert [c['reachable'] for c in census] == [17, 9, 34, 17, 17, 34, 34]
        exhaustive_indices = set(np.linspace(0, len(records) - 1, 12, dtype=int).tolist())
        for index, record in enumerate(records):
            restore_semantic_boundary(env, record.state)
            value = compute_boundary(env, planner)
            times.append(value.elapsed_seconds)
            actual = set(np.flatnonzero(value.mask).tolist())
            expected = set(record.reachable_action_ids)
            assert actual == expected, f'Frozen label mismatch in state {index}: {actual ^ expected}'
            row = dict(index=index, state_hash=record.state_hash,
                mask_sha256=sha256(value.mask.tobytes()).hexdigest(),
                geometric=value.geometric_count, reachable=len(actual),
                unreachable=value.geometric_count - len(actual),
                visited=value.search.visited_states, expanded=value.search.expanded_states,
                seconds=value.elapsed_seconds)
            rows.append(row)
            if index in exhaustive_indices:
                repeat = compute_boundary(env, planner)
                assert np.array_equal(value.mask, repeat.mask)
                action_times = []
                for action in record.valid_action_ids:
                    started = perf_counter()
                    old = baseline.plan(env, SPACE.decode(action))
                    duration = perf_counter() - started
                    action_times.append(duration)
                    assert old.reachable == bool(value.mask[action]), (index, action)
                    (reachable_plan_times if old.reachable else unreachable_plan_times).append(duration)
                    if old.reachable:
                        witness = value.plan_for(action)
                        assert (old.actions, old.needs_hard_drop) == (witness.actions, witness.needs_hard_drop)
                    cross_checks += 1
                paired_benchmark.append(dict(state_index=index,
                    enumeration_seconds=value.elapsed_seconds,
                    baseline_mean_single_target_seconds=float(np.mean(action_times)),
                    baseline_all_valid_targets_seconds=sum(action_times),
                    valid_target_count=len(action_times)))
        restore_semantic_boundary(env, records[0].state)
        tracemalloc.start()
        memory_boundary = compute_boundary(env, planner)
        _, search_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        mask = memory_boundary.mask
        owned_mask_bytes = sys.getsizeof(mask) + sys.getsizeof(mask.base)
        result = dict(**oracle_metadata('oracle'), dataset=str(args.dataset.resolve()),
            dataset_fingerprint=dataset_fingerprint(records),
            baseline_ref=subprocess.check_output(['git', 'rev-parse', args.baseline_ref], text=True).strip(),
            baseline_planner_source_sha256=source_hash,
            empty_board_census=census, state_count=len(rows),
            geometric=sum(r['geometric'] for r in rows), reachable=sum(r['reachable'] for r in rows),
            unreachable=sum(r['unreachable'] for r in rows),
            frozen_label_disagreements=0, false_reachable=0, false_unreachable=0,
            geometric_subset_violations=0, independent_baseline_action_checks=cross_checks,
            exhaustive_baseline_state_count=len(exhaustive_indices),
            enumeration_seconds=distribution(times),
            baseline_reachable_plan_seconds=distribution(reachable_plan_times),
            baseline_unreachable_plan_seconds=distribution(unreachable_plan_times),
            paired_baseline_states=paired_benchmark,
            paired_enumeration_seconds=distribution([p['enumeration_seconds'] for p in paired_benchmark]),
            paired_baseline_all_targets_seconds=distribution([p['baseline_all_valid_targets_seconds']
                                                            for p in paired_benchmark]),
            visited_nodes=distribution([r['visited'] for r in rows]),
            reachable_target_count=distribution([r['reachable'] for r in rows]),
            memory=dict(mask_payload_bytes=800, owned_mask_bytes=owned_mask_bytes,
                max_replay_extra_payload_bytes=160_000_000,
                max_replay_extra_array_and_backing_bytes=owned_mask_bytes * 2 * 100_000,
                state_hash_cache_entries=0, simultaneous_persistent_boundaries=2,
                empty_board_search_tracemalloc_peak_bytes=search_peak), states=rows)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')
        print(json.dumps({k: v for k, v in result.items() if k != 'states'}, indent=2))
    finally:
        env.close()


if __name__ == '__main__':
    main()
