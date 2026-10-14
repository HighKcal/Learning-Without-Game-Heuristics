"""Separate cProfile runs: timings here are NOT benchmark wall-clock results."""
import argparse
import cProfile
from pathlib import Path
import pstats

from _planner_benchmark import (ROOT, Tetris, TacticalPlanner, TargetPlacement,
                                restore, load_corpus, controlled_cases, write_json, metadata)


def function_name(key):
    filename, line, name = key
    try:
        filename = str(Path(filename).relative_to(ROOT))
    except ValueError:
        filename = filename.replace("\\", "/").split("site-packages/")[-1]
    return f"{filename}:{line}:{name}"


def profile_group(out, group, entries):
    profiler = cProfile.Profile()
    env = Tetris(gravity=True)
    env.reset(seed=42)
    try:
        for entry in entries[:5]:
            restore(env, entry["snapshot"])
            TacticalPlanner().plan(env, TargetPlacement(**entry["target"]))
        for entry in entries:
            restore(env, entry["snapshot"])
            # Only the original plan call is profiled, including finally.
            profiler.enable()
            try:
                TacticalPlanner().plan(env, TargetPlacement(**entry["target"]))
            finally:
                profiler.disable()
    finally:
        env.close()
    profiler.dump_stats(str(out / f"planner_v001_{group}.prof"))
    return summarize_profile(out, group, len(entries))


def summarize_profile(out, group, calls_count):
    stat = pstats.Stats(str(out / f"planner_v001_{group}.prof"))
    plan_total = sum(v[3] for k, v in stat.stats.items()
                     if k[2] == "plan" and Path(k[0]).name == "planner.py")
    assert plan_total > 0
    functions = []
    for key, (primitive, calls, own, cumulative, callers) in stat.stats.items():
        functions.append({"function": function_name(key), "primitive_calls": primitive,
                          "calls": calls, "total_s": own, "cumulative_s": cumulative,
                          "own_ms_per_call": own * 1000 / calls,
                          "cumulative_ms_per_primitive_call": cumulative * 1000 / primitive,
                          "cumulative_share": cumulative / plan_total,
                          "own_share": own / plan_total})
    cumulative_top = sorted(functions, key=lambda r: r["cumulative_s"], reverse=True)
    own_top = sorted(functions, key=lambda r: r["total_s"], reverse=True)
    # Caller breakdown helps distinguish HARD_DROP/native projection from verification.
    important = ("get_state", "set_state", "step", "restore_branch", "transition", "_state_key",
                 "target_at_lock", "observe_lock", "commit_active_tetromino", "drop_active_tetromino",
                 "_get_obs", "project_tetromino", "collision", "deepcopy", "copy", "occupied_offset",
                 "orientation_matrices", "env_origin_to_target")
    selected = []
    for key, value in stat.stats.items():
        if key[2] in important or any(n in key[2] for n in ("tobytes", "popleft", "append")):
            primitive, calls, own, cumulative, callers = value
            selected.append({"function": function_name(key), "calls": calls,
                             "total_s": own, "cumulative_s": cumulative,
                             "cumulative_share": cumulative / plan_total,
                             "callers": [{"function": function_name(k), "stats": list(v)}
                                         for k, v in callers.items() if isinstance(v, tuple)]})
    result = {"calls": calls_count, "profile_total_s": plan_total,
              "exclusive_total_s": stat.total_tt,
              "top15_cumulative": cumulative_top[:15], "top15_own": own_top[:15],
              "selected_functions": selected,
              "note": "Cumulative times overlap; never sum their shares. Profile time is diagnostic only."}
    print(f"profile {group}: {calls_count} calls, diagnostic seconds={plan_total:.3f}", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=ROOT / "benchmarks/planner/baseline/planner_v001_corpus.pkl.gz")
    parser.add_argument("--output", type=Path, default=ROOT / "_validation/planner_profile")
    parser.add_argument("--reuse-profiles", action="store_true",
                        help="Rebuild JSON summaries from saved .prof files without profiling again")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    corpus, manifest = load_corpus(args.corpus)
    # Full actual corpus once, subgroup files are separately profiled runs.
    groups = {"overall": corpus,
              "reachable": [e for e in corpus if e["baseline"]["status"] == "REACHABLE"],
              "unreachable": [e for e in corpus if e["baseline"]["status"] == "UNREACHABLE"],
              "controlled": controlled_cases()}
    result = {"metadata": metadata(), "corpus_sha256": manifest["corpus_sha256"],
              "groups": {name: (summarize_profile(args.output, name, len(entries)) if args.reuse_profiles
                                else profile_group(args.output, name, entries))
                         for name, entries in groups.items()}}
    write_json(args.output / "planner_v001_profile.json", result)


if __name__ == "__main__":
    main()
