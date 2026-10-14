"""Analysis-only utilities; no changes to planner/environment behavior.

Corpus snapshots are trusted, local pickle artifacts tied to dependency versions.
Live TetrisState fields are cloned together to retain queue/randomizer aliases and
deque maxlen (native get_state loses maxlen). The class-bound score method is not
serialized: replay removes the set_state score shadow, restoring native dispatch.
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

import copy
import csv
from dataclasses import asdict, fields
import gzip
import hashlib
import importlib.metadata
import json
import math
import pickle
import platform
import statistics
import subprocess
from time import perf_counter

import numpy as np
from tetris_gymnasium.envs.tetris import Tetris, TetrisState
from tetris_agent import TacticalPlanner, TargetPlacement, TetrisAdapter
from test_planner import select_piece, state_signature, assert_replay


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False,
                                    allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def metadata():
    has_git = (ROOT / '.git').exists()
    tracked = (subprocess.check_output(["git", "ls-files"], cwd=ROOT, text=True).splitlines()
               if has_git else [p.relative_to(ROOT).as_posix() for group in ('src','scripts')
                                for p in (ROOT/group).rglob('*.py')])
    return {
        "python": sys.version, "platform": platform.platform(),
        "processor": platform.processor(),
        "dependencies": {n: importlib.metadata.version(n) for n in
                         ("numpy", "gymnasium", "tetris-gymnasium", "pytest")},
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip() if has_git else None,
        "source_scope": "git tracked files" if has_git else "exported staging src/scripts",
        "tracked_sha256": {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest()
                           for p in tracked if (ROOT / p).is_file()},
        "clock": "time.perf_counter; outer plan() wall time includes finally restoration",
        "percentiles": "nearest rank; median averages the middle pair",
    }


def capture(env):
    assert "score" not in vars(env), "Corpus supports the native score method only"
    data = {f.name: getattr(env, f.name) for f in fields(TetrisState) if f.name != "score"}
    data["score"] = None
    return copy.deepcopy({
        "state": TetrisState(**data), "gravity": env.gravity_enabled,
        "np_random_state": env.np_random.bit_generator.state,
        "width": env.width, "height": env.height, "padding": env.padding,
    })


def restore(env, snapshot):
    assert (env.width, env.height, env.padding) == (
        snapshot["width"], snapshot["height"], snapshot["padding"])
    data = copy.deepcopy(snapshot)
    env.set_state(data["state"])
    vars(env).pop("score", None)
    env.gravity_enabled = data["gravity"]
    env.np_random.bit_generator.state = data["np_random_state"]
    assert env.queue.randomizer is env.randomizer


def measured_plan(env, target):
    # No instrumentation, hashing, copying, logging or correctness oracle in timer.
    started = perf_counter()
    result = TacticalPlanner().plan(env, target)
    return result, (perf_counter() - started) * 1000


def result_record(result):
    return {"status": result.status.value, "needs_hard_drop": result.needs_hard_drop,
            "action_length": result.path_length, "actions": list(result.actions),
            "expanded_states": result.expanded_states, "visited_states": result.visited_states}


def verify(env, target, result):
    if not result.reachable:
        return None  # No final path exists to verify; not a failed verification.
    assert_replay(env, target, result)  # Independent occupied-cell/pre-clear oracle.
    return True


def board_metrics(env):
    board = env.board[:env.height, env.padding:env.padding + env.width] != 0
    heights = [env.height - int(np.flatnonzero(board[:, x])[0])
               if board[:, x].any() else 0 for x in range(env.width)]
    holes = sum(int((~board[env.height - h:, x]).sum()) for x, h in enumerate(heights) if h)
    return {"board_occupied": int(board.sum()), "board_max_height": max(heights),
            "board_holes": holes, "board_mean_height": statistics.mean(heights)}


def stats(values):
    values = sorted(values)
    if not values:
        return {"count": 0, "total": 0, "avg": None, "median": None,
                "p95": None, "p99": None, "max": None}
    return {"count": len(values), "total": sum(values), "avg": statistics.mean(values),
            "median": statistics.median(values), "p95": values[math.ceil(.95 * len(values)) - 1],
            "p99": values[math.ceil(.99 * len(values)) - 1], "max": values[-1]}


def correlation(x, y):
    if len(x) < 2 or len(set(x)) < 2 or len(set(y)) < 2:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def ranks(values):
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranked = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        for i in order[start:end]:
            ranked[i] = (start + end - 1) / 2
        start = end
    return ranked


def summarize(rows):
    output = {}
    for group in ("overall", "REACHABLE", "UNREACHABLE"):
        selected = [r for r in rows if group == "overall" or r["status"] == group]
        times = [r["elapsed_ms"] for r in selected]
        expanded = [r["expanded_states"] for r in selected]
        visited = [r["visited_states"] for r in selected]
        pos = [r for r in selected if r["expanded_states"] > 0]
        output[group] = {
            "elapsed_ms": stats(times), "expanded_states": stats(expanded),
            "visited_states": stats(visited),
            "action_length": stats([r["action_length"] for r in selected]),
            "ms_per_expanded": stats([r["elapsed_ms"] / r["expanded_states"] for r in pos]),
            "zero_expanded_calls": len(selected) - len(pos),
            "weighted_ms_per_expanded": sum(r["elapsed_ms"] for r in pos) / sum(
                r["expanded_states"] for r in pos) if pos else None,
            "elapsed_vs_expanded_pearson": correlation(expanded, times),
            "elapsed_vs_expanded_spearman": correlation(ranks(expanded), ranks(times)),
            "elapsed_vs_visited_pearson": correlation(visited, times),
            "elapsed_vs_visited_spearman": correlation(ranks(visited), ranks(times)),
        }
        if len(set(expanded)) > 1:
            slope, intercept = np.polyfit(expanded, times, 1)
            residual = [t - (slope * e + intercept) for t, e in zip(times, expanded)]
            output[group]["linear_fit"] = {"ms_per_expanded_slope": float(slope),
                "intercept_ms": float(intercept),
                "residual_abs_ms": stats([abs(float(v)) for v in residual])}
    return output


def save_corpus(path, corpus):
    with gzip.open(path, "wb") as stream:
        pickle.dump(corpus, stream, protocol=5)
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_corpus(path):
    # NEVER load untrusted pickle. This file must be generated by collect locally.
    manifest = json.loads(Path(path).with_suffix("").with_suffix(".json").read_text(encoding="utf-8"))
    if hashlib.sha256(Path(path).read_bytes()).hexdigest() != manifest["corpus_sha256"]:
        raise ValueError("Corpus hash mismatch")
    for name, version in manifest["metadata"]["dependencies"].items():
        if name != "pytest" and importlib.metadata.version(name) != version:
            raise ValueError(f"Dependency mismatch: {name}")
    with gzip.open(path, "rb") as stream:
        return pickle.load(stream), manifest


def controlled_cases():
    # Exact fixture setups from test_planner.py and test_planner_cases.py.
    specifications = [
        ("A O direct drop", 1, (8, 18, 0)), ("B T rotation", 2, (8, 17, 1)),
        ("C O left edge", 1, (4, 18, 0)), ("C O right edge", 1, (12, 18, 0)),
        ("D I left vertical", 0, (4, 16, 1)), ("D I right vertical", 0, (13, 16, 1)),
        ("E O auto-lock", 1, (8, 18, 0)), ("F O tuck-under", 1, (8, 18, 0)),
        ("G sealed O cavity", 1, (7, 18, 0)), ("H sealed T cavity", 2, (7, 18, 0)),
    ]
    cases = []
    for name, index, coords in specifications:
        env = Tetris(gravity=True)
        try:
            env.reset(seed=42)
            select_piece(env, index)
            if name.startswith(("G", "H")):
                env.board[17, 4:14] = 2
            elif name.startswith("E"):
                env.x, env.y = 7, 18
            elif name.startswith("F"):
                env.board[16, 8:10] = 2
            cases.append({"id": name, "target": dict(zip(("x", "y", "orientation"), coords)),
                          "snapshot": capture(env),
                          "expected_needs_hard_drop": not name.startswith(("E", "F", "G", "H")),
                          "expected_status": "UNREACHABLE" if name.startswith(("G", "H")) else "REACHABLE"})
        finally:
            env.close()
    return cases
