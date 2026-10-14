"""Offline fixed exams. Labels never enter policy inputs or training code."""
from dataclasses import dataclass, field
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import torch

from .network import QNetwork
from .state import SemanticState, encode_states, geometric_mask


SCHEMA_VERSION = 1


def serialize_state(state: SemanticState) -> dict:
    return dict(board=[''.join(map(str, row)) for row in state.board.tolist()],
                current_piece=state.current_piece, next_queue=list(state.next_queue))


def deserialize_state(value: dict) -> SemanticState:
    if not isinstance(value, dict) or set(value) != {'board', 'current_piece', 'next_queue'}:
        raise ValueError('Expected exactly the Frozen SemanticState v1 fields.')
    rows = value['board']
    if (not isinstance(rows, list) or len(rows) != 20
            or any(not isinstance(row, str) or len(row) != 10 or set(row) - {'0', '1'}
                   for row in rows)):
        raise ValueError('Expected 20 binary row strings of width 10.')
    if not isinstance(value['next_queue'], list):
        raise ValueError('Expected next_queue list.')
    return SemanticState(np.array([[int(c) for c in row] for row in rows], dtype=np.uint8),
                         value['current_piece'], tuple(value['next_queue']))


def state_hash(state: SemanticState) -> str:
    """SHA256 of 200 row-major binary bytes + current ID + ordered four IDs only."""
    return sha256(state.board.tobytes() + bytes((state.current_piece, *state.next_queue))).hexdigest()


@dataclass(frozen=True)
class EvaluationRecord:
    state: SemanticState
    valid_action_ids: tuple[int, ...]
    reachable_action_ids: tuple[int, ...]
    metadata: dict = field(default_factory=dict, compare=False)

    @property
    def state_hash(self):
        return state_hash(self.state)

    def to_dict(self):
        return dict(state_hash=self.state_hash, state=serialize_state(self.state),
                    valid_action_ids=list(self.valid_action_ids),
                    reachable_action_ids=list(self.reachable_action_ids), metadata=self.metadata)


def _check_ids(ids, name):
    if (not isinstance(ids, (list, tuple))
            or any(type(i) is not int or not 0 <= i < 800 for i in ids)):
        raise ValueError(f'{name} must contain integer IDs in 0..799.')
    if len(set(ids)) != len(ids):
        raise ValueError(f'Duplicate {name}.')


def validate_records(records):
    seen = set()
    for record in records:
        _check_ids(record.valid_action_ids, 'valid_action_ids')
        _check_ids(record.reachable_action_ids, 'reachable_action_ids')
        valid, reachable = set(record.valid_action_ids), set(record.reachable_action_ids)
        if not valid or valid != set(np.flatnonzero(geometric_mask(record.state)).tolist()):
            raise ValueError('valid_action_ids must equal all geometric-valid actions.')
        if not reachable <= valid:
            raise ValueError('reachable_action_ids must be a subset of valid_action_ids.')
        if not reachable:
            raise ValueError('Zero-reachable state is a diagnostic failure, not an exam record.')
        if record.state_hash in seen:
            raise ValueError('Duplicate state_hash.')
        seen.add(record.state_hash)
        if not isinstance(record.metadata, dict):
            raise ValueError('Record metadata must be an object.')


def label_statistics(records):
    valid = [len(r.valid_action_ids) for r in records]
    reachable = [len(r.reachable_action_ids) for r in records]
    fractions = [1 - r / v for r, v in zip(reachable, valid)]
    return dict(total_states=len(records),
                only_reachable_states=sum(r == v for r, v in zip(reachable, valid)),
                mixed_states=sum(0 < r < v for r, v in zip(reachable, valid)),
                zero_reachable_states=sum(r == 0 for r in reachable),
                total_valid_actions=sum(valid), total_reachable_actions=sum(reachable),
                total_unreachable_actions=sum(valid) - sum(reachable),
                per_state_unreachable_fractions=fractions,
                unreachable_fraction_quantiles=(dict(zip(('min', 'p25', 'median', 'p75', 'max'),
                    np.quantile(fractions, [0, .25, .5, .75, 1]).tolist())) if fractions else None))


def _no_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f'Duplicate JSON key: {key}')
        result[key] = value
    return result


def _invalid_constant(value):
    raise ValueError(f'Non-finite JSON constant: {value}')


def read_json(text):
    return json.loads(text, object_pairs_hook=_no_duplicate_keys, parse_constant=_invalid_constant)


def json_text(value):
    return json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n'


def dataset_fingerprint(records):
    """Ordered canonical semantic content + labels only; exclude all metadata."""
    digest = sha256()
    for record in records:
        value = dict(state_hash=record.state_hash, state=serialize_state(record.state),
                     valid_action_ids=sorted(record.valid_action_ids),
                     reachable_action_ids=sorted(record.reachable_action_ids))
        digest.update((json.dumps(value, sort_keys=True, separators=(',', ':'),
                                  allow_nan=False) + '\n').encode('utf-8'))
    return digest.hexdigest()


def _validate_multiseed_metadata(records, metadata):
    """Additional orchestration checks; old single-seed schema v1 stays valid."""
    if 'dataset_fingerprint' in metadata and metadata['dataset_fingerprint'] != dataset_fingerprint(records):
        raise ValueError('Dataset fingerprint disagrees with canonical records.')
    if 'generation_seeds' not in metadata:
        return
    seeds = metadata['generation_seeds']
    quota = metadata.get('states_per_seed_requested')
    if (not isinstance(seeds, list) or not seeds
            or any(type(s) is not int or not 0 <= s < 2**32 for s in seeds)
            or len(set(seeds)) != len(seeds) or type(quota) is not int or quota <= 0):
        raise ValueError('Invalid multi-seed configuration.')
    if (metadata.get('total_requested_states') != len(seeds) * quota
            or metadata.get('requested_state_count') != len(seeds) * quota
            or metadata.get('total_accepted_states') != len(records)):
        raise ValueError('Multi-seed total counts disagree.')
    per_seed = metadata.get('per_seed', [])
    if [m.get('generation_seed') for m in per_seed] != seeds[:len(per_seed)] or not per_seed:
        raise ValueError('Per-seed metadata order disagrees with generation order.')
    expected_order = []
    for m in per_seed:
        seed = m['generation_seed']
        contribution = [r for r in records if r.metadata.get('generation_seed') == seed]
        if len(contribution) != m.get('accepted_state_count') or len(contribution) > quota:
            raise ValueError('Per-seed contribution count disagrees.')
        if m.get('aggregate_label_statistics') != label_statistics(contribution):
            raise ValueError('Per-seed aggregate statistics disagree.')
        expected_order.extend([seed] * len(contribution))
    if [r.metadata.get('generation_seed') for r in records] != expected_order:
        raise ValueError('Record generation seed order disagrees.')
    if metadata.get('global_duplicate_count') != sum(m['build_statistics']['rejected_duplicates'] for m in per_seed):
        raise ValueError('Global duplicate count disagrees.')
    if metadata.get('status') == 'completed' and (
            len(per_seed) != len(seeds) or len(records) != len(seeds) * quota
            or any(m.get('status') != 'completed' or m['accepted_state_count'] != quota for m in per_seed)):
        raise ValueError('Incomplete seed quota cannot be completed.')


def save_dataset(path, records, metadata):
    """New directory only. Metadata written last; incomplete builds cannot load."""
    records = tuple(records)
    validate_records(records)
    metadata = dict(metadata, schema_version=SCHEMA_VERSION, accepted_state_count=len(records),
                    aggregate_label_statistics=label_statistics(records))
    _validate_multiseed_metadata(records, metadata)
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    with (path / 'states.jsonl').open('x', encoding='utf-8', newline='\n') as stream:
        for record in records:
            stream.write(json.dumps(record.to_dict(), sort_keys=True, allow_nan=False) + '\n')
    (path / 'metadata.json').write_text(json_text(metadata), encoding='utf-8')


def load_dataset(path):
    path = Path(path)
    metadata = read_json((path / 'metadata.json').read_text(encoding='utf-8'))
    required = {'schema_version', 'git_commit', 'tetris_gymnasium_version', 'generation_seed',
                'requested_state_count', 'accepted_state_count', 'rejected_state_count',
                'generation_configuration', 'aggregate_label_statistics'}
    if not isinstance(metadata, dict) or not required <= metadata.keys():
        raise ValueError('Missing required dataset metadata.')
    if type(metadata['schema_version']) is not int or metadata['schema_version'] != SCHEMA_VERSION:
        raise ValueError('Unsupported dataset schema_version.')
    for name in ('requested_state_count', 'accepted_state_count', 'rejected_state_count'):
        if type(metadata[name]) is not int or metadata[name] < 0:
            raise ValueError(f'Invalid {name}.')
    if metadata['accepted_state_count'] > metadata['requested_state_count']:
        raise ValueError('Accepted count exceeds requested count.')
    records = []
    for line in (path / 'states.jsonl').read_text(encoding='utf-8').splitlines():
        value = read_json(line)
        if not isinstance(value, dict) or not {
                'state_hash', 'state', 'valid_action_ids', 'reachable_action_ids'} <= value.keys():
            raise ValueError('Missing required record fields.')
        state = deserialize_state(value['state'])
        if value['state_hash'] != state_hash(state):
            raise ValueError('state_hash disagrees with semantic content.')
        _check_ids(value['valid_action_ids'], 'valid_action_ids')
        _check_ids(value['reachable_action_ids'], 'reachable_action_ids')
        records.append(EvaluationRecord(state, tuple(value['valid_action_ids']),
                                        tuple(value['reachable_action_ids']), value.get('metadata', {})))
    validate_records(records)
    if metadata['accepted_state_count'] != len(records):
        raise ValueError('Accepted count disagrees with records.')
    if metadata['aggregate_label_statistics'] != label_statistics(records):
        raise ValueError('Aggregate statistics disagree with records.')
    _validate_multiseed_metadata(records, metadata)
    return tuple(records), metadata


def _mean(values):
    return sum(values) / len(values) if values else None


def analytical_random_reference(records):
    """State-weighted exact expectation, not an episode baseline or Monte Carlo."""
    probabilities = [len(r.reachable_action_ids) / len(r.valid_action_ids) for r in records]
    mixed = [p for p in probabilities if 0 < p < 1]
    return dict(overall_reachable_rate=_mean(probabilities), mixed_state_reachable_rate=_mean(mixed),
                state_count=len(probabilities), mixed_state_count=len(mixed))


@torch.no_grad()
def evaluate_network(network, records, *, reachability_mask_mode='geometric'):
    """Greedy epsilon=0; default exam consults labels AFTER selection.

    Explicit oracle mode reconstructs each native canonical boundary and
    computes a fresh physical mask; frozen labels cross-check the result.

    Ties use the lowest action ID, matching the training policy's torch.argmax.
    Q summaries are expected-return diagnostics, never reachability probabilities.
    """
    records = tuple(records)
    if reachability_mask_mode not in ('geometric', 'oracle'):
        raise ValueError('Unknown reachability mask mode.')
    validate_records(records)
    if not records:
        raise ValueError('Cannot evaluate an empty dataset.')
    device = next(network.parameters()).device
    modes = [(module, module.training) for module in network.modules()]
    rows = []
    reachable_q, unreachable_q = [], []
    oracle_env = None
    oracle_seconds = 0.0
    if reachability_mask_mode == 'oracle':
        from tetris_gymnasium.envs.tetris import Tetris
        from ..planner import TacticalPlanner
        from .oracle import compute_boundary, restore_semantic_boundary
        oracle_env = Tetris(gravity=True)
        planner = TacticalPlanner()
    try:
        network.eval()
        for record in records:
            # Only SemanticState reaches the frozen 235-element encoder.
            q = network(encode_states([record.state], device))
            if q.shape != (1, 800) or not bool(torch.isfinite(q).all()):
                raise ValueError('Expected finite Q values of shape (1, 800).')
            q = q[0].detach().cpu().tolist()
            valid = sorted(record.valid_action_ids)
            ranked = sorted(valid, key=lambda action: (-q[action], action))
            reachable = set(record.reachable_action_ids)
            if oracle_env is not None:
                restore_semantic_boundary(oracle_env, record.state)
                boundary = compute_boundary(oracle_env, planner)
                oracle_seconds += boundary.elapsed_seconds
                allowed = set(np.flatnonzero(boundary.mask).tolist())
                if allowed != reachable:
                    raise ValueError('Oracle disagrees with frozen evaluation labels.')
                chosen = next(a for a in ranked if a in allowed)
            else:
                chosen = ranked[0]
            rq = [q[a] for a in valid if a in reachable]
            uq = [q[a] for a in valid if a not in reachable]
            reachable_q.extend(rq)
            unreachable_q.extend(uq)
            rows.append(dict(state_hash=record.state_hash, action_id=chosen,
                reachable=chosen in reachable, mixed=bool(rq and uq),
                mean_reachable_q=_mean(rq), mean_unreachable_q=_mean(uq),
                mean_q_gap=_mean(rq) - _mean(uq) if rq and uq else None,
                max_reachable_q=max(rq) if rq else None,
                max_unreachable_q=max(uq) if uq else None,
                margin=max(rq) - max(uq) if rq and uq else None,
                highest_reachable_rank=next((i + 1 for i, a in enumerate(ranked) if a in reachable), None)))
    finally:
        if oracle_env is not None:
            oracle_env.close()
        for module, mode in modes:
            module.training = mode
    mixed_rows = [r for r in rows if r['mixed']]
    rate = _mean([r['reachable'] for r in rows])
    from .oracle import oracle_metadata
    return dict(**oracle_metadata(reachability_mask_mode),
        oracle_elapsed_seconds=oracle_seconds, geometric_valid_action_count=len(reachable_q) + len(unreachable_q),
        reachable_action_count=len(reachable_q), masked_physical_u_count=len(unreachable_q) if oracle_env else 0,
        selected_physical_u_count=sum(not r['reachable'] for r in rows),
        reachable_fraction=len(reachable_q) / (len(reachable_q) + len(unreachable_q)),
        evaluated_states=len(rows), greedy_first_choice_reachable_rate=rate,
        greedy_first_choice_unreachable_rate=1 - rate,
        mixed_states=len(mixed_rows),
        mixed_state_greedy_reachable_rate=_mean([r['reachable'] for r in mixed_rows]),
        analytical_random_reference=analytical_random_reference(records),
        secondary_diagnostics=dict(
            interpretation='Q is expected discounted return, not reachability probability.',
            aggregation='Q means are action-weighted; gaps/margins are state-weighted on mixed states.',
            reachable_action_count=len(reachable_q), unreachable_action_count=len(unreachable_q),
            paired_state_count=len(mixed_rows), rank_state_count=len(rows),
            mean_reachable_q=_mean(reachable_q), mean_unreachable_q=_mean(unreachable_q),
            mean_q_gap=_mean([r['mean_q_gap'] for r in mixed_rows]),
            mean_margin=_mean([r['margin'] for r in mixed_rows]),
            mean_highest_reachable_rank=_mean([r['highest_reachable_rank'] for r in rows])),
        states=rows)


def load_checkpoint_network(path, device='cpu'):
    """Read existing snapshot schema; restore only online weights, never training."""
    snapshot = torch.load(path, map_location='cpu', weights_only=True)
    if (not isinstance(snapshot, dict) or type(snapshot.get('schema_version')) is not int
            or snapshot['schema_version'] != 1
            or snapshot.get('checkpoint_contract', {}).get('kind') != 'training_snapshot'
            or snapshot['checkpoint_contract'].get('exact_resume_supported') is not False):
        raise ValueError('Expected existing schema v1 training_snapshot checkpoint.')
    # Initial allocation must not consume the live trainer's Torch RNG.
    with torch.random.fork_rng(devices=[]):
        network = QNetwork()
    network.load_state_dict(snapshot['online_state_dict'], strict=True)
    return network.to(device)
