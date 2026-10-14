# WSL 재현성 근거와 한계

Phase3(2026-10-13)의 검증된 보고서를 공개용으로 정리했다. 원본 보고서와 raw JSON/XML은 비공개 보존한다. 이 문서는 Phase4의 새 장기 학습 성과가 아니다.

| 기존 Phase3 항목 | 확인된 결과 |
| --- | --- |
| 환경 | Ubuntu24.04 x86_64 / Python3.12.10 / Torch2.14.1 CPU·cu132 |
| 의존성 | CPU30 / CUDA49 wheel URL·version·SHA256 lock, provenance |
| Native | Python19개 raw SHA256 일치, 원 MIT 고지, render dtype1줄 수정 |
| CUDA-build suite | 808 PASS / 0 FAIL / 0 SKIP |
| CPU-build suite | 803 PASS / 0 FAIL / 5 SKIP |
| Historical diagnostic | 기존 checkpoint 재업데이트34case 제외 |
| 모델 기능 | 9개×2 modes18 checks, 9×256×2=4608 CPU/CUDA action 비교 |
| CPU/CUDA Q | allclose rtol=atol1e−5, 최대 차5.7220459e−6; bitwise 동일 아님 |
| 보존 Oracle rollout | seed123periodic500K2 episodes의16개 비시간 field·trajectory digest 일치 |
| Fresh Smoke4조건 | Geometric/Oracle×CPU/CUDA32 transitions/29 updates/3 sync, 실제 weight변화·3 checkpoints reload |

Smoke32/warmup4/batch4/replay64/sync8은 production5000/128/100000/5000과 다르다. Oracle selected U0은 mask 구조의 효과다. 장기500K/2M 수렴·성능/처리량·CUDA bitwise 학습·exact resume·다른 머신이나 GPU는 미인증이다. 당시 설치된 환경의 lock dry-run과 새 제3머신 설치도 구분했다.

Phase3 보고서는 lock cache가 한때 원본 경로에 생성된 뒤 자체 파일만 외부로 옮기고 보호 검사를 재수행한 오류·복구를 기록했다. 같은 시기 GUI 외부 변경도 관찰했으며 GUI 전체 무변경으로 판정하지 않았다. 공개 사본은 개인정보 경로와 private 증거 링크만 제거하며 이 한계를 숨기지 않는다.

Phase4에서는 공개 후보 소스만 Linux 새 디렉터리에 복사하고 **새 CPU venv에 lock을 실제 설치**한 뒤 Native·tests·fresh Smoke·평가·추론을 검증한다. 세부 결과는 [Phase4 보고서](PHASE4_GITHUB_PUBLIC_PREPARATION_REPORT.md)의 측정값으로 구분한다. 기존 interpreter/WSL host를 사용하므로 OS부터 독립적인 제3머신 인증은 아니다. [실행 가이드](WSL_TRAINING.md)
