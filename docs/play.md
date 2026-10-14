# 연구 추론과 관전 정책

`scripts/play_agent.py`는 명시한 checkpoint의 online weights를 `weights_only=True`로 로딩하고 frozen ε=0 rollout을 수행한다. 새 Smoke checkpoint에 이 CLI를 사용할 수 있다. Trainer·optimizer·replay update는 없다. CLI 직접 입력이 Registry 지원 승인이나 모델 provenance 검증을 대신하지 않으므로 Native 검증을 먼저 실행한다.

Geometric `--mode research`는 top1 U self-loop를 유지한다. `--mode play`는 Q 순위에서 confirmed U만 건너뛰는 Planner fallback이다. Oracle `--reachability-mask oracle`은 mode와 관계없이 기존 single-BFS mask/argmax/witness 계약이다. 관전용 fallback이 잘 실행되는 결과를 Decision top1의 학습 성공이라고 말하지 않는다.

`scripts/infer_agent.py`는 기존 모델용 Registry/FrozenInference façade다. 정확한 experiment ID, role/step, mode, 파일 SHA256, Native profile, CPU 지원을 검증한다. 선택 파일이 없거나 pending이면 실패한다. 4개 지원 experiments/9 checkpoints의 배포 권한은 아직 미확정이며 binary는 공개 후보에 없다.

| 연구 family | Research mode | 관전 mode |
| --- | --- | --- |
| Geometric | `research_geometric_top1` | `main_geometric_play_fallback` |
| Oracle | `research_oracle_greedy` | `main_oracle_play_witness` |

새 출력 디렉터리만 사용하고 모델은 재저장하지 않는다. [명령 예시](WSL_TRAINING.md) · [아키텍처](architecture.md) · [GUI 관계](gui.md)
