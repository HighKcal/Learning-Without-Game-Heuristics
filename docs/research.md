# 연구 범위·한계와 후속 질문

원래 질문인 Decision의 안정적인 물리적 실행 가능성 학습은 해결되지 않았다. Geometric-valid target과 Native로 reachable한 target의 간극 때문에 학습 실패와 실행 search를 구분해야 했다. Penalty와 S1~S4는 local 효과가 달랐고 장기 collapse도 남았다.

Oracle에서는 외부 BFS가 feasibility를 제공해 전략 개선을 관찰했다. 2M final의 모든 평가 episode가 사망 종료했고, 1M 이후 공통 greedy 평가 차이의 CI는0을 포함했다. High-board frontier는 높이·holes·방문 상태가 함께 달라 inconclusive다. 과거 누적 training-state exposure가 없어 노출 부족이 원인이라고 확정하지 않는다.

Historical checkpoints는 정확한 resume payload가 아니다. Archive의 exact-resume 연구와 현재 production snapshot 계약을 혼합하지 않는다. CUDA forward/action 일치와 32-decision 실제 학습은 장기 bitwise 재현성·다른 GPU 설치·수렴 증거가 아니다. Public fixture는 선택된256 states이며 전체 상태 공간이나 독립 gameplay 성능을 대표하지 않는다.

후속 계획으로 matched empty-start vs 완전 Native snapshot curriculum, 공통 probe bank, n-step ablation을 검토할 수 있다. 실행된 연구 결과는 발견되지 않았다. Semantic board trace만으로 완전 Native snapshot bank를 만들었다고 간주하지 않는다.

상세한 실패·불일치·미저장 자료는 [DEVLOG 부록 C](devlog.md#부록-c-자료-불일치증거-누락과-주장-상한), [근거 상태](evidence.md), [실험 비교](experiments.md)에 있다. 비공개 raw 근거가 미포함이므로 독자는 공개 집계와 공개 코드로 새로운 run을 만들 수 있지만 모든 과거 수치를 독립 재현할 자료가 제공된 것은 아니다.
