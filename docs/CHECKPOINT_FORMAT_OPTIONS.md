# 원본 Checkpoint와 추론 전용 파생 모델

원본9개는 파일당 약5.32MB, 합계47,896,785bytes다. 각각 online/target weights와 Adam step·exp_avg·exp_avg_sq, config·epsilon·counters·metrics·timestamps를 포함한다. Replay·RNG·완전한 환경 episode 상태는 없어 정확resume를 지원하지 않는다. 기존 Registry/FrozenInference는 원filename/hash와 schema v1 training_snapshot 계약을 요구한다.

| 형식 | 장점 | 제한과 추가 검증 |
| --- | --- | --- |
| 원본 `.pt`9개 | 원연구 bytes/hash·role·metadata 보존, 기존 loader와 호환 | optimizer/target까지 전달, pickle 계열 포맷이므로 출처 확인 필요 |
| online-only `.pt` | 불필요 target/Adam 제거, 작은 state_dict | 여전히 pickle 계열, 현재 schema 변경시 기존 loader 거부; 별도 loader 필요 |
| online-only safetensors | weights 중심의 제한된 직렬화, metadata 최소화 | 새 dependency/loader·manifest·hash·권리·배포 승인 필요; downstream 위험이0인 것은 아님 |

Online weights는331,808 float32 parameters=1,327,232 raw bytes다. 파생파일을 실제 저장한 크기가 아니다. 이번에는 변환을 실행하지 않았고 원본 대체를 결정하지 않았다. Optimizer 전달을 피하려면 원본과 별개의 새 파생 artifact에 대해 승인을 받아야 한다.

향후 승인된 변환은 원online tensor의 key/dtype/shape/raw hash를 exact 대조한다. 같은256-state fixture의 Q·mode별 mask/greedy action·tie rule을 비교하고 Native19 hash와 같은 CPU 환경을 고정한다. 같은 seed의 Research/Play rollout에서 시간 외 trajectory field/digest, Oracle witness 및 Geometric U/fallback 계약을 검증한다. 새 artifact는 독립 ID/hash로 등록하며 원본9개·periodic/final을 합치거나 덮어쓰지 않는다.
