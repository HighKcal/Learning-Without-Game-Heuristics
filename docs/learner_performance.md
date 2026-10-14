# 성능 측정의 범위

초기 학습에서 작은 MLP보다 단일상태 평가·Replay/검증·Planner 비용이 병목일 수 있었다. 후속 최적화는 의미론과 owned data를 보존하며 batch validation, piece geometry cache, search-local primitives, 재사용 branch workspace를 적용했다.

Oracle perf-v2의 paired fixture 측정은 gameplay boundary97.878→38.456ms(2.545×), corpus106.047→41.666ms였다. 같은 device의 transition·online/target/Adam hash 일치를 확인한 pipeline은 CPU7.302→17.565, CUDA8.250→19.276 decisions/s였다. 192-decision pass3회 결과로 장기500K/2M 처리량을 인증하지 않는다. Oracle 비용은 최종 wall share79~82%로 남았다.

CPU7.324/CUDA8.070의 앞선 다른 trajectory 표본은 순수 device 차이가 아니다. 최적화로 연구 성능이 좋아졌다는 인과 주장도 하지 않는다. 상세 시점·가설·측정은 [DEVLOG48](devlog.md#48-oracle-profiling--perf-v2--의미론을-보존하며-장기-실행-비용-감소), 과거 raw profile은 비공개 근거로 보존됐다.
