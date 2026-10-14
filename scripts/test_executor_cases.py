"""Small headless Planner -> Executor demonstration; no Decision Agent."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tetris_gymnasium.envs.tetris import Tetris

from tetris_agent import Executor, TacticalPlanner, TargetPlacement


def main() -> None:
    cases = (
        ("O right", 1, TargetPlacement(10, 18, 0)),
        ("T rotation", 2, TargetPlacement(8, 17, 1)),
        ("O direct drop", 1, TargetPlacement(8, 18, 0)),
        ("O auto-lock RIGHT", 1, TargetPlacement(8, 18, 0)),
        ("I auto-lock clear", 0, TargetPlacement(7, 17, 0)),
        ("O lock then game over", 1, TargetPlacement(8, 0, 0)),
        ("sealed O cavity", 1, TargetPlacement(7, 18, 0)),
    )
    print("case | planner actions | needs drop | executed actions | success | "
          "target verified | raw reward | terminated | truncated | failure")
    for name, index, target in cases:
        env = Tetris(gravity=True)
        try:
            env.reset(seed=42)
            env.active_tetromino = env.tetrominoes[index]
            env.reset_tetromino_position()
            if name == "O auto-lock RIGHT":
                env.x, env.y = 7, 18
                env.queue.queue[0] = 1
            elif name == "I auto-lock clear":
                env.active_tetromino = env.rotate(env.active_tetromino)
                env.x, env.y = 7, 16
                env.board[17, 4:14] = 3
                env.board[17, 7:11] = 0
                env.board[18, 7] = 3
                env.queue.queue[0] = 0
            elif name == "O lock then game over":
                env.board[2, 4:14] = 2
            elif name == "sealed O cavity":
                env.board[17, 4:14] = 2
            labels = {
                env.actions.move_left: "LEFT", env.actions.move_right: "RIGHT",
                env.actions.move_down: "DOWN", env.actions.rotate_clockwise: "CW",
                env.actions.rotate_counterclockwise: "CCW", env.actions.no_op: "NO_OP",
                env.actions.hard_drop: "HARD_DROP",
            }

            def sequence(actions):
                return " ".join(labels[a] for a in actions) or "[]"

            plan = TacticalPlanner().plan(env, target)
            result = Executor().execute(env, target, plan)
            failure = result.failure_reason.value if result.failure_reason else "-"
            print(f"{name} | {sequence(plan.actions)} | {plan.needs_hard_drop} | "
                  f"{sequence(result.executed_actions)} | {result.success} | "
                  f"{result.target_verified} | {result.raw_reward_sum:g} | "
                  f"{result.terminated} | {result.truncated} | {failure}")
        finally:
            env.close()


if __name__ == "__main__":
    main()
