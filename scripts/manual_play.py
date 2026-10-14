"""Original key-by-key manual play, moved out of .venv."""

import argparse

import cv2
from tetris_gymnasium.envs.tetris import Tetris


CONTROLS = """
조작법
----------------
A : 왼쪽
D : 오른쪽
S : 아래
E : 시계방향 회전
W : 반시계방향 회전
SPACE : 하드드롭
Q : HOLD / SWAP
X : NOOP
R : 게임 리셋
ESC : 종료
----------------
"""


def main(smoke_test: bool = False) -> None:
    env = Tetris(render_mode="human", gravity=True)
    try:
        env.reset(seed=42)
        print(CONTROLS)
        actions = {
            ord("a"): env.actions.move_left,
            ord("d"): env.actions.move_right,
            ord("s"): env.actions.move_down,
            ord("e"): env.actions.rotate_clockwise,
            ord("w"): env.actions.rotate_counterclockwise,
            ord(" "): env.actions.hard_drop,
            ord("q"): env.actions.swap,
            ord("x"): env.actions.no_op,
        }
        smoke_keys = iter([*map(ord, "adsew qxr"), 27]) if smoke_test else None
        terminated = truncated = False
        steps = resets = renders = 0
        while not (terminated or truncated):
            env.render()
            renders += 1
            if smoke_keys is None:
                # Preserve the original behavior: wait for each key, with
                # environment gravity applied once per primitive action.
                key = cv2.waitKey(0) & 0xFF
            else:
                cv2.waitKey(1)
                key = next(smoke_keys)
            if key == 27:
                break
            if key == ord("r"):
                env.reset()
                terminated = truncated = False
                resets += 1
                continue
            if key not in actions:
                continue
            _, reward, terminated, truncated, info = env.step(actions[key])
            steps += 1
            print(f"x={env.x}, y={env.y}, reward={reward}, lines={info['lines_cleared']}")
            if terminated:
                print("Game Over!")
        if smoke_test:
            if (steps, resets, renders) != (8, 1, 10):
                raise RuntimeError("Manual-play smoke test ended before exercising all keys.")
            print(f"Smoke test passed: {steps} actions, {resets} reset, {renders} renders, ESC exit.")
    finally:
        env.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Play Tetris with the original keyboard controls.")
    parser.add_argument("--smoke-test", action="store_true", help="Render and exercise every control, then exit.")
    main(smoke_test=parser.parse_args().smoke_test)
