# 아키텍처와 실행 계약

Decision → TargetPlacement → Tactical Planner / Oracle → Executor → Native의 책임을 나눈다. 직접 좌우 키를 고르는 RL이 아니라 목표 배치를 고르는 RL이다. 구현 식별은 `src/tetris_agent`를 기준으로 한다.

| 계층 | 책임 | 실패 처리 |
| --- | --- | --- |
| Native / Adapter | 20×10 보드, 조각, 실제 primitive 중력·회전·충돌, canonical spawn 상태 | 상태 계약 불일치 시 중단 |
| Decision | State235에서 Q800을 계산하고 주어진 mask 내 목표 선택 | 기하 mask는 실제 도달 가능성을 보증하지 않음 |
| Tactical Planner | Native 전이를 따르는 BFS, 목표 도달 경로·복원 검증 | confirmed U와 UNKNOWN/timeout/error 구분 |
| Executor | 검증된 primitive 입력 실행, lock·목표·보드 결과 검증 | 불확실한 실행을 학습 transition으로 저장하지 않음 |
| Oracle | 경계별 full BFS 1회로 도달 가능 mask와 순서가 있는 witness 생성 | uncertainty는 replay 전에 abort |

State235=잠긴 20×10 binary board 200 + 현재 1개/보이는 next 4개×7 one-hot 35다. 숨은 RNG, 현재 물리 위치, Hold는 인코딩하지 않는다. Action800=(y×10+x)×4+rotation의 고정 슬롯이며 occupied bounding box 좌표와 중복 회전 mask 계약을 사용한다. 중력은 Decision 단위가 아니라 Native primitive 입력마다 적용된다.

Geometric Research의 confirmed UNREACHABLE은 Executor를 호출하지 않고 semantic state를 그대로 두며 −0.05(또는 해당 실험 penalty)를 준다. 이 self-loop에서 고정 greedy 정책은 같은 ID를 다시 고를 수 있다. 성공할 때까지 후보를 자동 제거하면 원 연구의 실패 측정이 달라진다.

| 명시 모드 | 실행 의미 |
| --- | --- |
| `research_geometric_top1` | geometric mask의 Q top1, U self-loop 보존 |
| `main_geometric_play_fallback` | Q 순위별 target Planner; confirmed U만 건너뛰고 첫 R의 plan 실행 |
| `research_oracle_greedy` | full BFS mask 안의 Q argmax와 해당 witness 실행 |
| `main_oracle_play_witness` | 같은 Oracle 계약을 frozen façade로 제공 |

Frozen inference는 trainer/optimizer/replay를 생성하지 않는다. Registry의 ID+role/step+파일명+SHA256+Native profile을 함께 확인하며 missing/pending/unsupported에 latest나 legacy fallback이 없다. Oracle은 미래 숨은 RNG 정보를 Decision input에 추가하는 기능이 아니라 현재 Native 상태의 실행 가능성 조건이다.

Oracle N-decision capped nonterminal episode는 bootstrap용 next boundary까지 N+1 BFS, 자연 종료는 N BFS다. Perf-v2는 mask·ordered witness·복원·replay ownership을 보존하는 최적화이며 물리를 바꾸지 않았다. [DEVLOG](devlog.md)의 48·55절, [학습 설계](rl_pretraining.md), [관전 정책](play.md)을 참고한다.
