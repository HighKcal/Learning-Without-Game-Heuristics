"""Reproducible v0.01 workload collection, unprofiled replay and clean E2E timing."""
import argparse
import csv
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

from _planner_benchmark import (
    ROOT, Tetris, TacticalPlanner, TargetPlacement, TetrisAdapter, capture, restore,
    measured_plan, result_record, verify, state_signature, board_metrics, metadata,
    write_json, write_csv, save_corpus, load_corpus, controlled_cases, summarize, stats, correlation,
)
from tetris_agent import GameRunner, RandomDecisionAgent


class Collector(TacticalPlanner):
    def __init__(self):
        self.inputs = []

    def plan(self, env, target):
        before = state_signature(env.unwrapped)
        snapshot = capture(env.unwrapped)
        metrics = board_metrics(env.unwrapped)
        plan = super().plan(env, target)
        assert state_signature(env.unwrapped) == before
        self.inputs.append({"snapshot": snapshot, "target": {
            "x": int(target.x), "y": int(target.y), "orientation": int(target.orientation)},
            "baseline": result_record(plan), **metrics})
        return plan


def collect(out, minimum, minimum_unreachable):
    corpus, games = [], []
    unreachable = 0
    index = 0
    while index < 20 or len(corpus) < minimum or unreachable < minimum_unreachable:
        # Extend a deterministic list, independent of timing or sample outcomes.
        env_seed, decision_seed = 42 + index * 101, 42 + index * 313
        env = Tetris(gravity=True)
        collector = Collector()
        try:
            game = GameRunner(RandomDecisionAgent(decision_seed), planner=collector).run(
                env, env_seed=env_seed, verbose=False)
            assert game.executor_failure_count == 0
            assert len(collector.inputs) == game.decision_count
            for entry, attempt in zip(collector.inputs, game.attempts):
                entry.update(id=f"g{index:03d}-t{attempt.turn:03d}-a{attempt.attempt:03d}",
                    game_index=index, env_seed=env_seed, decision_seed=decision_seed,
                    turn=attempt.turn, attempt=attempt.attempt, piece_type=attempt.piece_type,
                    action_id=attempt.decision.action_id)
                entry["baseline"]["target_verified"] = attempt.execution_success
                corpus.append(entry)
            unreachable += game.unreachable_count
            games.append({"game_index": index, "env_seed": env_seed, "decision_seed": decision_seed,
                          "calls": game.decision_count, "unreachable": game.unreachable_count,
                          "pieces": game.pieces_placed, "lines": game.lines_cleared,
                          "stop_reason": game.stop_reason.value})
        finally:
            env.close()
        index += 1
        print(f"collect: games={index} calls={len(corpus)} unreachable={unreachable}", flush=True)
    digest = save_corpus(out / "planner_v001_corpus.pkl.gz", corpus)
    manifest = {"schema": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
                "metadata": metadata(), "corpus_sha256": digest, "games": games,
                "calls": len(corpus), "reachable": len(corpus) - unreachable, "unreachable": unreachable,
                "snapshot_note": "Trusted local gzip/pickle of jointly cloned live TetrisState, score=None; replay restores class score and np_random state. No timers from collection are official."}
    write_json(out / "planner_v001_corpus.json", manifest)
    return corpus, manifest


def replay(out, corpus, repeats):
    env = Tetris(gravity=True)
    env.reset(seed=42)
    rows, checks, baselines = [], [], []
    try:
        for entry in corpus[:10]:
            restore(env, entry["snapshot"])
            TacticalPlanner().plan(env, TargetPlacement(**entry["target"]))
        for run in range(1, repeats + 1):
            for entry in corpus:
                restore(env, entry["snapshot"])
                before = state_signature(env)
                target = TargetPlacement(**entry["target"])
                plan, elapsed = measured_plan(env, target)
                assert state_signature(env) == before
                record = result_record(plan)
                baseline = entry["baseline"]
                # Reachability/drop semantics are mandatory, exact actions are not.
                assert (record["status"], record["needs_hard_drop"]) == (
                    baseline["status"], baseline["needs_hard_drop"]), entry["id"]
                verified = verify(env, target, plan)
                if run == 1:
                    baselines.append({"id": entry["id"], "target": entry["target"],
                                      **record, "target_verified": verified})
                row = {"run": run, "id": entry["id"], "game_index": entry["game_index"],
                    "turn": entry["turn"], "attempt": entry["attempt"], "piece_type": entry["piece_type"],
                    **entry["target"], "status": record["status"], "needs_hard_drop": record["needs_hard_drop"],
                    "elapsed_ms": elapsed, "internal_elapsed_ms": plan.elapsed_seconds * 1000,
                    "expanded_states": plan.expanded_states, "visited_states": plan.visited_states,
                    "action_length": plan.path_length, "target_verified": verified,
                    **{k: entry[k] for k in ("board_occupied", "board_max_height", "board_holes", "board_mean_height")}}
                rows.append(row)
            summary = summarize([r for r in rows if r["run"] == run])
            checks.append(summary)
            print(f"replay {run}: {summary['overall']['elapsed_ms']}", flush=True)
    finally:
        env.close()
    write_csv(out / "planner_v001_calls.csv", rows)
    write_json(out / "planner_v001_correctness.json", baselines)
    return analyze_replay(rows, corpus, repeats)


def analyze_replay(rows, corpus, repeats):
    assert len(rows) == len(corpus) * repeats
    for run in range(repeats):
        assert [r["id"] for r in rows[run * len(corpus):(run + 1) * len(corpus)]] == [e["id"] for e in corpus]
    checks = [summarize([r for r in rows if r["run"] == run]) for run in range(1, repeats + 1)]
    # Average repeats for each input before association/progression analysis.
    unique = []
    for i, entry in enumerate(corpus):
        row = rows[i].copy()
        row["elapsed_ms"] = sum(rows[i + j * len(corpus)]["elapsed_ms"] for j in range(repeats)) / repeats
        unique.append(row)
    progression = {}
    for label, selector in {"early_turn_1_5": lambda r: r["turn"] <= 5,
                             "mid_turn_6_10": lambda r: 6 <= r["turn"] <= 10,
                             "late_turn_11_plus": lambda r: r["turn"] > 10}.items():
        progression[label] = summarize([r for r in unique if selector(r)])
    for group in ("overall", "REACHABLE", "UNREACHABLE"):
        selected = [r for r in unique if group == "overall" or r["status"] == group]
        progression[group + "_correlations"] = {k: {
            "vs_elapsed": correlation([r[k] for r in selected], [r["elapsed_ms"] for r in selected]),
            "vs_expanded": correlation([r[k] for r in selected], [r["expanded_states"] for r in selected])}
            for k in ("turn", "board_occupied", "board_max_height", "board_holes")}
    same_counts = defaultdict(list)
    for row in unique:
        same_counts[(row["status"], row["expanded_states"])].append(row["elapsed_ms"])
    count_variation = [{"status": key[0], "expanded_states": key[1],
                        "timing": stats(v), "max_min_ratio": max(v) / min(v)}
                       for key, v in same_counts.items() if len(v) >= 5]
    return {"runs": checks, "pooled_3_runs": summarize(rows),
            "unique_inputs_mean_timing": summarize(unique), "progression": progression,
            "same_expanded_groups_min_5": sorted(count_variation, key=lambda x: x["timing"]["count"], reverse=True)}


class TimedPlanner(TacticalPlanner):
    def __init__(self):
        self.times = []

    def plan(self, env, target):
        started = perf_counter()
        plan = super().plan(env, target)
        self.times.append((perf_counter() - started) * 1000)
        return plan


def e2e(out, corpus, manifest):
    rows = []
    for game_input in manifest["games"]:
        env = Tetris(gravity=True)
        timer = TimedPlanner()
        try:
            game = GameRunner(RandomDecisionAgent(game_input["decision_seed"]), planner=timer).run(
                env, env_seed=game_input["env_seed"], verbose=False)
            expected = [e for e in corpus if e["game_index"] == game_input["game_index"]]
            assert len(expected) == len(game.attempts)
            for e, a in zip(expected, game.attempts):
                assert e["target"] == {"x": int(a.decision.target.x), "y": int(a.decision.target.y),
                                       "orientation": int(a.decision.target.orientation)}
                assert e["baseline"]["status"] == a.plan.status.value
            assert game.executor_failure_count == 0
            planner_s = sum(timer.times) / 1000
            rows.append({**game_input, "game_total_s": game.game_duration, "planner_total_s": planner_s,
                         "planner_ratio": planner_s / game.game_duration, "executor_failures": 0})
        finally:
            env.close()
    write_csv(out / "planner_v001_e2e.csv", rows)
    return {"games": len(rows), "game_total_s": sum(r["game_total_s"] for r in rows),
            "planner_total_s": sum(r["planner_total_s"] for r in rows),
            "weighted_planner_ratio": sum(r["planner_total_s"] for r in rows) / sum(r["game_total_s"] for r in rows),
            "per_game_ratio": stats([r["planner_ratio"] for r in rows]), "executor_failures": 0}


def controlled(out):
    rows, baselines, summary = [], [], {}
    env = Tetris(gravity=True)
    env.reset(seed=42)
    try:
        for entry in controlled_cases():
            invariant = None
            for iteration in range(25):
                restore(env, entry["snapshot"])
                before = state_signature(env)
                target = TargetPlacement(**entry["target"])
                plan, elapsed = measured_plan(env, target)
                assert state_signature(env) == before
                record = result_record(plan)
                assert record["status"] == entry["expected_status"]
                assert record["needs_hard_drop"] == entry["expected_needs_hard_drop"]
                signature = tuple(record[k] for k in ("status", "needs_hard_drop", "action_length", "expanded_states", "visited_states"))
                if invariant is None:
                    invariant = signature
                assert signature == invariant
                verified = verify(env, target, plan)
                if iteration == 5:
                    baselines.append({"id": entry["id"], "target": entry["target"], **record,
                                      "target_verified": verified})
                if iteration >= 5:
                    rows.append({"id": entry["id"], "iteration": iteration - 4,
                                 "elapsed_ms": elapsed, "status": record["status"],
                                 "expanded_states": plan.expanded_states, "visited_states": plan.visited_states,
                                 "action_length": plan.path_length, "needs_hard_drop": plan.needs_hard_drop})
            summary[entry["id"]] = summarize([r for r in rows if r["id"] == entry["id"]])[entry["expected_status"]]
            print(f"controlled {entry['id']}: {summary[entry['id']]['elapsed_ms']}", flush=True)
    finally:
        env.close()
    write_csv(out / "planner_v001_controlled.csv", rows)
    write_json(out / "planner_v001_controlled_correctness.json", baselines)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("all", "collect", "replay", "finish"), default="all")
    parser.add_argument("--output", type=Path, default=ROOT / "_validation" / "planner_benchmark")
    parser.add_argument("--corpus", type=Path)
    parser.add_argument("--min-calls", type=int, default=1000)
    parser.add_argument("--min-unreachable", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 1 or args.min_calls < 1 or args.min_unreachable < 1:
        parser.error("counts must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    if args.mode in ("all", "collect"):
        corpus, manifest = collect(args.output, args.min_calls, args.min_unreachable)
    else:
        corpus, manifest = load_corpus(args.corpus or args.output / "planner_v001_corpus.pkl.gz")
    if args.mode != "collect":
        if args.mode == "finish":
            # Resume analysis of a completed saved replay without repeating timing.
            with (args.output / "planner_v001_calls.csv").open(encoding="utf-8", newline="") as stream:
                rows = list(csv.DictReader(stream))
            for row in rows:
                for key in ("run", "game_index", "turn", "attempt", "x", "y", "orientation",
                            "expanded_states", "visited_states", "action_length", "board_occupied",
                            "board_max_height", "board_holes"):
                    row[key] = int(row[key])
                for key in ("elapsed_ms", "internal_elapsed_ms", "board_mean_height"):
                    row[key] = float(row[key])
                for key in ("needs_hard_drop", "target_verified"):
                    row[key] = {"True": True, "False": False, "": None}[row[key]]
            replay_result = analyze_replay(rows, corpus, args.repeats)
        else:
            replay_result = replay(args.output, corpus, args.repeats)
        result = {"metadata": metadata(), "corpus_sha256": manifest["corpus_sha256"],
                  "workload": {k: manifest[k] for k in ("calls", "reachable", "unreachable")},
                  "seed_pairs": len(manifest["games"]), "repeats": args.repeats,
                  "benchmark": replay_result,
                  "controlled": controlled(args.output), "e2e": e2e(args.output, corpus, manifest)}
        write_json(args.output / "planner_v001_baseline.json", result)
        print(f"saved baseline to {args.output}", flush=True)


if __name__ == "__main__":
    main()
