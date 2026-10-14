"""Source/data separation must not introduce legacy fallback or grant support."""
import json
from pathlib import Path
import pytest
import yaml

from tetris_agent.research_paths import artifact_root
from tetris_agent.experiments import Registry, ResolutionError, digest
from tetris_agent.artifact_catalog import artifact_path, analysis_inputs


def test_unconfigured_root_remains_repo_local(tmp_path, monkeypatch):
    monkeypatch.delenv('TETRIS_ARTIFACT_ROOT', raising=False)
    assert artifact_root(tmp_path) == tmp_path.resolve()


def test_explicit_root_overrides_environment(tmp_path, monkeypatch):
    monkeypatch.setenv('TETRIS_ARTIFACT_ROOT', str(tmp_path / 'other'))
    assert artifact_root(tmp_path / 'repo', tmp_path / 'chosen') == (tmp_path / 'chosen').resolve()


def test_relative_environment_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setenv('TETRIS_ARTIFACT_ROOT', 'relative-data')
    with pytest.raises(ValueError, match='absolute'):
        artifact_root(tmp_path)


@pytest.fixture
def separated(tmp_path):
    repo, data = tmp_path / 'repo', tmp_path / 'private'
    (repo / 'experiments').mkdir(parents=True)
    run = data / 'runs/fixture'
    run.mkdir(parents=True)
    (run / 'model.pt').write_bytes(b'synthetic catalog fixture, not a model')
    (run / 'config.json').write_text('{}', encoding='utf-8')
    (data / 'analysis/geometric/penalty_020/seed_42/500k').mkdir(parents=True)
    (data / 'evaluation/fixed_eval_final_v1').mkdir(parents=True)
    entry = dict(id='geometric.penalty020.seed42.500k', current_path='runs/fixture',
                 config='runs/fixture/config.json', checkpoints=[dict(
                     artifact_id='final', role='final', step=5, filename='model.pt',
                     sha256=digest(run / 'model.pt'), staging_compatibility='pending')])
    (repo / 'experiments/index.yaml').write_text(
        yaml.safe_dump(dict(schema_version=1, experiments=[entry])), encoding='utf-8')
    return repo, data


def test_registry_uses_configured_data_without_replacing_metadata(separated, monkeypatch):
    repo, data = separated
    monkeypatch.setenv('TETRIS_ARTIFACT_ROOT', str(data))
    registry = Registry(repo)
    assert registry.repo_root == repo.resolve()
    assert registry.artifact_root == data.resolve()
    assert registry.select('geometric.penalty020.seed42.500k', role='final')[1]['staging_compatibility'] == 'pending'


def test_catalog_reads_separate_evidence_and_keeps_pending(separated):
    repo, data = separated
    result = analysis_inputs('geometric_500k', root=repo, artifact_root=data)
    assert Path(result['evidence']).is_relative_to(data)
    assert Path(result['fixed_evaluation']).is_relative_to(data)
    assert Path(result['inputs'][0]['config']).is_relative_to(data)
    assert result['inputs'][0]['inference_compatibility'] == 'pending'
    assert result['canonical_evidence_modified'] is False


def test_configured_missing_catalog_does_not_search_other_roots(tmp_path, monkeypatch):
    monkeypatch.setenv('TETRIS_ARTIFACT_ROOT', str(tmp_path / 'missing'))
    with pytest.raises(ResolutionError, match='ARTIFACT_NOT_INSTALLED'):
        artifact_path('fixed_eval')


def test_catalog_explicit_root_overrides_environment(separated, monkeypatch):
    _, data = separated
    monkeypatch.setenv('TETRIS_ARTIFACT_ROOT', str(data / 'absent'))
    assert artifact_path('fixed_eval', root=data) == (data / 'evaluation/fixed_eval_final_v1').resolve()
