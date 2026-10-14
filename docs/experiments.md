# 실험 목록·관측·해석

Registry는 [experiments/index.yaml](../experiments/index.yaml)이며 81 records=80 run/context+logical S2 sequence 1이다. 로컬 존재, CPU 추론 지원, 모델 배포 승인은 별도 상태다. S1~S4의 구현·raw run은 비공개 archive에 보존하고 production 소스에는 통합하지 않았다.

아래는 과거 결과를 정리한 표다. 신규 학습 성과가 아니다. U는 confirmed physical UNREACHABLE이며 UNKNOWN/error를 분자에 넣지 않는다. Training occurrence, fixed-state greedy, frozen gameplay, fallback Play는 분모가 다르다.

| 실험 / 가설 | 변경 변수·조건 | 실제 결과 | 해석·한계 |
| --- | --- | --- | --- |
| 최초 100K / DDQN이 물리적 U를 피할 수 있는가 | geometric, seed42/123, penalty−.05 | LATE U 91.660 / 91.265% | 전략 network만으로 실행 가능성 안정 학습 미해결 |
| Penalty ablation / action gap을 키우면 U 감소하는가 | 100K 두 seed, −.05/−.10/−.20; EARLY(10K,30K], LATE(80K,100K] | −.20 LATE U 68.365 / 67.635%, 감소23.295 / 23.630pp | −.10 mixed, 잔존 U; 동시 CUDA의 bitwise pairing 미보장 |
| 회복 / 장기 학습 또는 replay reset 영향 | seed42 −.20 warm-start150K vs fresh500K | 추가50K U55.792%; fresh 마지막50K U63.630% | Warm-start는 exact continuation 아님; 회복 후 재악화 |
| Frozen trap / exploration 없이 회피하는가 | 4개 100K final, training-exposed dev50 seeds, first-U prefix | baseline50/50 trap, penalty42/50·43/50 | 1000-decision projection을 새 rollout으로 표시하지 않음 |
| S1 / auxiliary head가 선택 일반화를 돕는가 | exact6000 late branches, BCE λ.01; veto 없음 | seed42 U80.1→81.7667%, max U292→771 | Replay fitting은 높지만 online 선택 개선 미입증 |
| S2 / 관측 U의 상대 Q를 낮추면 탈출하는가 | observed-U margin.05, λ.01, geometric policy 유지 | seed42 U80.100→74.033%; seed12386.050→84.717%, max U343→658 | seed replication과 극단 tail 불균일 |
| S3 / known-U coverage·comparator가 원인인가 | seed123 exact branch, S2/S3-A/S3-B 각6000 | U84.7167/81.7833/80.5167%, max U658/386/249 | 최장 greedy-ID persistence131과 U-streak249는 다른 단위 |
| S4 / stale target self-loop bootstrap 영향 | 양쪽 S3-B, confirmed U에만 fresh online bootstrap | U80.5167→77.7167%, max U249→128 | median streak 악화·task tradeoff; 한 branch의 local 효과 |
| 결합500K / local 효과가 지속되는가 | fresh S2+S3-B+S4 vs 복구 S2 canonical | 마지막50K 결합 U93.38%, 2.417 lines/1000P | 후기 collapse; system-level 비교로 S4 단독 인과 아님 |
| Oracle / 외부 feasibility를 주면 전략을 학습하는가 | paired seed42 100K, geometric vs physical mask | Oracle selected U0, placement exposure 증가 | 외부 mask의 효과, Decision 자체 feasibility 해결 아님 |
| Oracle perf-v2 / 의미 보존으로 비용 감소 가능한가 | paired 반복, 동일 mask·witness·복원·weights/Adam gate | gameplay BFS97.878→38.456ms, CPU7.302→17.565 decisions/s | 작은 fixture 처리량; 장기 성능 비교 아님 |
| Oracle seed123500K / 더 긴 전략 학습 | DDQN/minimal/decay38000 | periodic500K50-seed54.938 lines/1000P; local450K56.950 | local winner와 paired 불확실성 구분 |
| Oracle seed422M / 개선·정체·고보드 한계 | 별도 fresh run, 공통100 seeds+별도 final50 seeds | final holdout132.778 lines/1000P; 2M−1M paired CI[−4.25,27.24] | 후기 확정 우위 미입증, 50/50 deaths, competence frontier inconclusive |

Oracle500K와2M은 동시 시작한 별도 seed/run이다. 실행일·commit일, 당시 가설·관측·사후 해석은 [DEVLOG 35~55와 부록](devlog.md)에 구분했다. Penalty100K의 약68%와 fresh500K 후기63.630%를 같은 결과로 대체하지 않는다.

선별된 Oracle 과거 평가 CSV는 [results/README.md](results/README.md)에 있다. 개발 PC 경로 열만 제거하고 수치·protocol은 보존했다. 이것은 전체 raw Canonical 공개나 보존된 모델 전체의 평가 인증이 아니다. 지원9개 중 evaluation 기록이 없는 artifact는 [models.md](models.md)에 `미확인`으로 남긴다.
