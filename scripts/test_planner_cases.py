"""Small, headless Planner v0.01 case/metrics demonstration (no executor)."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tetris_gymnasium.envs.tetris import Tetris

from tetris_agent import TacticalPlanner, TargetPlacement


def main() -> None:
    cases = (
        ("O right", 1, TargetPlacement(10, 18, 0)),
        ("T rotation", 2, TargetPlacement(8, 17, 1)),
        ("T right+rotation", 2, TargetPlacement(10, 17, 1)),
        ("O direct drop", 1, TargetPlacement(8, 18, 0)),
        ("O left edge", 1, TargetPlacement(4, 18, 0)),
        ("O right edge", 1, TargetPlacement(12, 18, 0)),
        ("I left vertical", 0, TargetPlacement(4, 16, 1)),
        ("I right vertical", 0, TargetPlacement(13, 16, 1)),
        ("sealed O cavity", 1, TargetPlacement(7, 18, 0)),
        ("sealed O no gravity", 1, TargetPlacement(7, 18, 0)),
        ("sealed T cavity", 2, TargetPlacement(7, 18, 0)),
        ("O auto-lock RIGHT", 1, TargetPlacement(8, 18, 0)),
        ("I auto-lock clear", 0, TargetPlacement(7, 17, 0)),
        ("O tuck under roof", 1, TargetPlacement(8, 18, 0)),
    )
    print("case | status | expanded | visited | length | elapsed ms | final HARD_DROP | actions")
    for name, index, target in cases:
        env = Tetris(gravity=True)
        try:
            env.reset(seed=42)
            env.active_tetromino = env.tetrominoes[index]
            env.reset_tetromino_position()
            if name.startswith("sealed"):
                env.board[17, 4:14] = 2
                env.gravity_enabled = name != "sealed O no gravity"
            elif name == "O auto-lock RIGHT":
                env.x, env.y = 7, 18
            elif name == "I auto-lock clear":
                env.active_tetromino = env.rotate(env.active_tetromino)
                env.x, env.y = 7, 16
                env.board[17, 4:14] = 3
                env.board[17, 7:11] = 0
                env.board[18, 7] = 3
                env.queue.queue[0] = 0
            elif name == "O tuck under roof":
                env.board[16, 8:10] = 2
            labels = {
                env.actions.move_left: "LEFT", env.actions.move_right: "RIGHT",
                env.actions.move_down: "DOWN", env.actions.rotate_clockwise: "CW",
                env.actions.rotate_counterclockwise: "CCW", env.actions.no_op: "NO_OP",
            }
            result = TacticalPlanner().plan(env, target)
            sequence = " ".join(labels[a] for a in result.actions) or "[]"
            print(f"{name} | {result.status.value} | {result.expanded_states} | "
                  f"{result.visited_states} | {result.path_length} | "
                  f"{result.elapsed_seconds * 1000:.3f} | {result.needs_hard_drop} | {sequence}")
        finally:
            env.close()


if __name__ == "__main__":
    main()
