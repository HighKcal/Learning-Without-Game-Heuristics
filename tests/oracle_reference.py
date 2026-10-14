"""Independent pre-perf-v2 reference; synthetic fixtures only, no Git writes."""
import hashlib
import importlib
import json
from pathlib import Path
import sys
from types import ModuleType
import torch


def baseline():
    directory = Path(__file__).parent / 'fixtures/oracle_pre_perf_v2'
    provenance = json.loads((directory / 'PROVENANCE.json').read_text(encoding='utf-8'))
    for row in provenance['files']:
        actual = hashlib.sha256((directory / row['fixture_path']).read_bytes()).hexdigest()
        if actual != row['sha256']:
            raise AssertionError('Independent baseline fixture hash differs')
    package = ModuleType('_performance_baseline')
    package.__path__ = [str(directory)]
    sys.modules.setdefault(package.__name__, package)
    return importlib.import_module('_performance_baseline.planner')


def weights_digest(trainer):
    """Original optimization gate's network/target/Adam-state digest."""
    h = hashlib.sha256()
    def visit(value):
        if torch.is_tensor(value):
            h.update(value.detach().cpu().numpy().tobytes())
        elif isinstance(value, dict):
            for key, val in value.items():
                h.update(str(key).encode())
                visit(val)
        elif isinstance(value, (list, tuple)):
            for val in value:
                visit(val)
        else:
            h.update(str(value).encode())
    for state in (trainer.online.state_dict(), trainer.target.state_dict(), trainer.optimizer.state_dict()):
        visit(state)
    return h.hexdigest()
