# Formal Random 기준선

`FormalRandomPolicy`는 전체 geometric-valid ID에서 한 번 균일 선택한다. RLRunner는 Planner의 confirmed U에 Executor를 호출하지 않고 원 self-loop를 유지한다. 실패 target을 제거하고 재선택하는 `run_random_game.py`의 end-to-end smoke와 다르다.

Network·trainer·optimizer·replay 저장·checkpoint는 만들지 않는다. Episode metrics와 transient transition 계약은 유지한다. Natural death는 TERMINATED, decision cap은 TRUNCATED다. Planner uncertainty·실행 error는 U 성공적인 라벨이 아니며 diagnostic으로 중단한다. Seed별 환경과 policy RNG를 명시해 새 출력에 실행한다.

CLI의 전체 옵션은 `python -B scripts/run_formal_random.py --help`로 확인한다. [학습 설계](rl_pretraining.md) · [실험 해석](experiments.md)
