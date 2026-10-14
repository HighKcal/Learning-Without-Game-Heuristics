# 강화학습 구조와 설정

MLP는 235→256→256→800, hidden ReLU다. Vanilla DQN은 target network의 최대 Q로 bootstrap한다. Double DQN은 online network로 다음 행동을 고르고 target network로 그 행동을 평가한다. 구현은 `rl/dqn.py`, 주요 후속 연구는 Double DQN이다.

| 항목 | production 기본 / 역사 조건 |
| --- | --- |
| 알고리즘·보상 | Double DQN / minimal (Vanilla·shaped 구현도 보존) |
| γ / Adam lr | 0.99 / 1e−4 |
| Batch / Replay / warmup | 128 / 100000 / 5000 stored experiences |
| Target sync / gradient clip | 5000 optimizer updates / 10 |
| Epsilon | warmup 1.0, 이후 0.05까지 선형 감소 |
| 100K·장기 비교 decay | 38000 stored experiences; 장기 budget에서는 명시 override |
| 기본 U penalty | −0.05; ablation −0.10 / −0.20 |
| Episode cap | 10000 decisions, 자연 사망과 truncation 구분 |

Minimal reward는 실제 lines − U penalty 비용 − 실제 game-over 1이다. Watchdog truncation은 death penalty를 더하지 않는다. Terminated는 bootstrap을 끄고, nonterminal truncation은 next state에서 bootstrap한다. Replay는 상태·mask의 소유권을 보장한다. Oracle next mask는 실제 경계에서 계산해 acting과 replay에 재사용한다.

기본 decay는 `ceil(.4*(budget−warmup))`이므로 장기 실험의 역사 조건을 맞추려면 `--epsilon-decay-decisions 38000`을 명시해야 한다. CLI의 단순 budget 변경만으로 같은 실험이 되지 않는다.

일반 training snapshot은 online/target/Adam/counters를 포함하지만 replay/RNG/Native episode orchestration 전체를 저장하지 않아 **정확한 resume를 지원하지 않는다**. 기존 exact-resume와 S1~S4 구현은 별도 비공개 연구 archive에 보존됐다. 이 공개 production 코드에 그 개입을 새로 통합하지 않았다. 관련 실패 결과는 [실험 문서](experiments.md)와 [DEVLOG](devlog.md)에 남긴다.

Smoke는 32 decisions / warmup4 / batch4 / replay64 / sync8updates / cap16의 기능 검사다. Production 기본과 장기 성능 평가를 대신하지 않는다. [실행 가이드](WSL_TRAINING.md)
