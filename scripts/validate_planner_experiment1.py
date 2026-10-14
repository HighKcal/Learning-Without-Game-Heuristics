"""Strict Experiment 1 comparison and plan-only counts; never time benchmarks here."""
from collections import Counter
from contextlib import ExitStack
import json
from pathlib import Path
from unittest.mock import patch

from _planner_benchmark import (
    ROOT, Tetris, TacticalPlanner, TargetPlacement, load_corpus, controlled_cases,
    restore, state_signature, result_record, verify, write_json,
)


FIELDS = (
    "status", "actions", "action_length", "needs_hard_drop",
    "expanded_states", "visited_states", "target_verified",
)
BASELINE = ROOT / "benchmarks/planner/baseline"
OUTPUT = ROOT / "_validation/planner_experiment1"


def check(env, entry, baseline, counts=None):
    restore(env, entry["snapshot"])
    before = state_signature(env)
    target = TargetPlacement(**entry["target"])
    with ExitStack() as stack:
        if counts is not None:
            for cls, name, label in (
                (Tetris, "get_state", "Tetris.get_state"),
                (type(env.randomizer), "__copy__", "BagRandomizer.__copy__"),
                (Tetris, "step", "Tetris.step"),
                (Tetris, "_get_obs", "Tetris._get_obs"),
            ):
                original = getattr(cls, name)

                def counted(instance, *args, _original=original, _label=label, **kwargs):
                    counts[_label] += 1
                    if _label == "Tetris.step" and args[0] == instance.actions.hard_drop:
                        counts["HARD_DROP probe"] += 1
                    return _original(instance, *args, **kwargs)

                stack.enter_context(patch.object(cls, name, counted))
        plan = TacticalPlanner().plan(env, target)
    # Restoration and the independent path oracle are outside count instrumentation.
    assert state_signature(env) == before, entry["id"]
    record = {"id": entry["id"], "target": entry["target"],
              **result_record(plan), "target_verified": verify(env, target, plan)}
    assert record["target"] == baseline["target"], entry["id"]
    for field in FIELDS:
        assert record[field] == baseline[field], (
            entry["id"], field, baseline[field], record[field])
    if "baseline" in entry:
        for field in FIELDS[:-1]:
            assert record[field] == entry["baseline"][field], (entry["id"], field)
    return record


def main():
    corpus, manifest = load_corpus(BASELINE / "planner_v001_corpus.pkl.gz")
    original = json.loads((BASELINE / "planner_v001_correctness.json").read_text(encoding="utf-8"))
    assert [e["id"] for e in corpus] == [r["id"] for r in original]
    counts = Counter()
    env = Tetris(gravity=True)
    env.reset(seed=42)
    try:
        records = []
        for index, (entry, baseline) in enumerate(zip(corpus, original), 1):
            records.append(check(env, entry, baseline, counts))
            if index % 200 == 0:
                print(f"strict corpus: {index}/{len(corpus)} matched", flush=True)
        write_json(OUTPUT / "strict_correctness.json", records)
        print(f"strict corpus: all {len(corpus)} matched on {FIELDS}", flush=True)

        cases = controlled_cases()
        original_cases = json.loads((BASELINE / "planner_v001_controlled_correctness.json").read_text(encoding="utf-8"))
        assert [e["id"] for e in cases] == [r["id"] for r in original_cases]
        controlled = [check(env, entry, baseline) for entry, baseline in zip(cases, original_cases)]
        write_json(OUTPUT / "strict_controlled_correctness.json", controlled)
        print(f"strict controlled: all {len(cases)} matched", flush=True)
    finally:
        env.close()

    expected = {
        "Tetris.get_state": 317397, "BagRandomizer.__copy__": 317397,
        "Tetris.step": 269349, "Tetris._get_obs": 269349, "HARD_DROP probe": 48041,
    }
    result = {"corpus_sha256": manifest["corpus_sha256"], "compared_fields": FIELDS,
              "corpus_matched": len(records), "controlled_matched": len(controlled),
              "expanded_states": sum(r["expanded_states"] for r in records),
              "visited_states": sum(r["visited_states"] for r in records),
              "counts_scope": "Only the 1003 plan calls; excludes restore, oracle and controlled cases",
              "counts": dict(counts), "expected_counts": expected,
              "count_mismatches": {k: {"expected": v, "actual": counts[k]}
                                   for k, v in expected.items() if counts[k] != v}}
    write_json(OUTPUT / "strict_validation.json", result)
    print(json.dumps(result, indent=2), flush=True)
    assert not result["count_mismatches"], result["count_mismatches"]


if __name__ == "__main__":
    main()
