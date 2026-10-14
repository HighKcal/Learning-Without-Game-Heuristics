# 고정 상태 평가

`evaluation/fixed_eval_final_v1`은 4개 seed×64=256개 frozen SemanticState와 geometric-valid / Planner-labeled reachable action을 가진 선별 fixture다. 전체 Canonical은 미포함이다. 동일 상태의 greedy 첫 선택을 비교하는 offline exam이며 실제 gameplay 성능이나 Formal Random episode와 다르다.

`evaluate_policy.py`는 untrained seed 또는 schema v1 snapshot의 online weights를 읽는다. Q는 expected return이며 도달 가능성 확률이 아니다. Oracle 평가의 selected U0은 mask 효과다. Fixed-state Q ranking, training U-rate, frozen episode survival을 혼합하지 않는다.

`states.jsonl`·`metadata.json`·`freeze_gate.json`은 검증된 소스 바이트를 유지했다. 과거 `untrained_seed42.json`의 source.dataset PC 경로만 상대 경로로 바꿨고 Q/평가 수치는 그대로다. 원본은 비공개 소스에 보존했다. Hash 차이는 공개 manifest에 표시한다.

새 fixture는 `scripts/build_fixed_eval.py`로 새 출력에 생성할 수 있다. 원 fixture에 overwrite하지 않는다. 공개 fixture 배포 권한은 프로젝트 소스·문서 권리 확인에 포함된다. [평가 명령](WSL_TRAINING.md)
