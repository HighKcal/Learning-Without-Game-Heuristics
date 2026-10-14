"""Portable read-only navigation for preserved analysis and benchmark evidence.

Historical scripts/reports retain their provenance. Callers obtain their inputs
here and supply a new output directory rather than running historical writers.
"""
from pathlib import Path
from .experiments import relative_path, ResolutionError
from .experiments import Registry, digest
from .research_paths import artifact_root as configured_artifact_root

CATALOG={
 'oracle_2m':'analysis/oracle_mask/seed_42/2m',
 'oracle_500k':'analysis/oracle_mask/seed_123/500k',
 'geometric_500k':'analysis/geometric/penalty_020/seed_42/500k',
 'oracle_vs_geometric':'analysis/comparisons/oracle_vs_geometric/100k',
 'u_margin_vs_bootstrap':'analysis/comparisons/u_margin_vs_bootstrap/500k',
 'fixed_eval':'evaluation/fixed_eval_final_v1',
 'planner_baseline':'benchmarks/planner/baseline',
 'planner_dedup':'benchmarks/planner/snapshot_after_dedup',
 'planner_probe':'benchmarks/planner/live_drop_probe',
 'learner_performance':'benchmarks/learner/performance',
 's2_original':'runs/interventions/u_margin/seed_42/500k/original',
 's2_resume':'runs/interventions/u_margin/seed_42/500k/resume_from_450k',
 's2_source':'archive/worktrees/u-margin-lab/source',
 's2_fixture':'archive/u_margin/evidence/.s2_long_training_validation',
 's4':'runs/interventions/bootstrap/seed_42/500k',
}

def artifact_path(name, *, root=None, member=None):
    if name not in CATALOG:raise ResolutionError('UNKNOWN_ARTIFACT', name)
    repo_root=Path(__file__).resolve().parents[2]
    root=configured_artifact_root(repo_root, root)
    relative=CATALOG[name]+('/'+member if member is not None else '')
    path=relative_path(root,relative)
    if not path.exists():raise ResolutionError('ARTIFACT_NOT_INSTALLED',str(path))
    return path


ANALYSIS_EXPERIMENTS={
 'oracle_2m':('oracle_mask.seed42.2m',),
 'oracle_500k':('oracle_mask.seed123.500k',),
 'geometric_500k':('geometric.penalty020.seed42.500k',),
 'oracle_vs_geometric':('oracle_mask.seed42.100k','geometric.baseline.paired_oracle.seed42.100k'),
 'u_margin_vs_bootstrap':('interventions.u_margin.seed42.500k','interventions.bootstrap.seed42.500k'),
}

def analysis_inputs(name, *, root=None, artifact_root=None):
    """Locate historical report inputs without executing or rewriting its code.

    Artifact existence/hash does not grant inference compatibility. In particular
    archived S2/S4 models remain pending and cannot enter FrozenInference.
    """
    if name not in ANALYSIS_EXPERIMENTS:raise ResolutionError('UNKNOWN_ANALYSIS', name)
    registry=Registry(root, artifact_root=artifact_root)
    inputs=[]
    for experiment_id in ANALYSIS_EXPERIMENTS[name]:
        entry=registry.entries[experiment_id]
        final=[a for a in entry['checkpoints'] if a['artifact_id']=='final']
        if len(final)!=1:raise ResolutionError('AMBIGUOUS_CHECKPOINT',experiment_id)
        a=final[0]
        path=relative_path(registry.artifact_root,entry['current_path']+'/'+a['filename'])
        if not path.is_file():raise ResolutionError('MODEL_NOT_INSTALLED',str(path))
        if digest(path)!=a['sha256']:raise ResolutionError('HASH_MISMATCH',str(path))
        config=relative_path(registry.artifact_root,entry['config'])
        if not config.is_file():raise ResolutionError('ARTIFACT_NOT_INSTALLED',str(config))
        inputs.append(dict(experiment_id=experiment_id,run=str(path.parent),checkpoint=str(path),
            config=str(config),sha256=a['sha256'],inference_compatibility=a['staging_compatibility']))
    return dict(analysis=name,evidence=str(artifact_path(name,root=registry.artifact_root)),
                fixed_evaluation=str(artifact_path('fixed_eval',root=registry.artifact_root)),
                inputs=inputs,historical_script_executed=False,canonical_evidence_modified=False)
