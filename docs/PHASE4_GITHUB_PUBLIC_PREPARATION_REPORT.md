# 테멍 Phase 4 — GitHub 공개 준비 결과

> 역사적 준비 보고서: 아래 승인 대기·미게시 판정은 Phase4/4.5 작성 시점의 기록이다. Phase5에서 사용자가source/document/fixture/model9개MIT와게시를승인했다. 현재조건은 [라이선스 안내](licensing.md)와 [승인 기록](RELEASE_APPROVALS.md)을 따른다.


작성일: 2026-10-14 (Asia/Seoul). **최종 판정: GO WITH CONDITIONS — 로컬 공개 후보 준비 완료, 게시 승인 대기.** 소스 공개는 권리·LICENSE 승인까지 HOLD, 모델은 HOLD, Windows GUI/EXE 공개 Beta는 NO-GO다.

## 1. 작업 전후 상태

원본 main HEAD는 `0a07550bdd5a9a154486b48359ac00cf1344725e`다. 수정된 DEVLOG와 미추적 연구 자료가 있어 강제로 전환하지 않았다. Phase 2·Phase 3·DEVLOG 통합 후 검증 기준은 `codex/research-staging-integration`의 `3801e53c236262b8196880b1fb7392559e91b13a`(2026-10-13 23:03:03 KST)다. 통합 checkout의 기존 미추적 DEVLOG 초안도 보존했다. main·통합·GUI HEAD와 기존 status는 검사 시점에 일치했다. 원본 docs에는 요청받은 내부용 결과 보고서만 새로 추가했다.

원 연구 origin은 `HighKcal/Tetris_Agent_Lab`, GUI origin은 사용자 지정 Private `HighKcal/Temeong`이다. GitHub connector 조회는 존재/접근 확인 실패(422)여서 실제 remote HEAD·visibility는 확인하지 못했다. 기존 remote와 원격 저장소는 변경하지 않았다.

## 2. 공개 후보 스냅샷

로컬 후보 디렉터리명은 **`Temeong_Public_Candidate_20261014`**다. 실제 절대 위치는 내부용 보고서에 기록한다. 원본 연구·Staging·GUI·비공개 감사 폴더와 별도이며, `.git`이 없는 **192개 파일 / 약 2.6MB**의 스냅샷이다. 기존 Git/LFS 이력은 포함하지 않는다.

먼저 160개 소스 파일의 명시 목록을 기록한 뒤 파일마다 복사했다. 최종 [PUBLIC_ALLOWLIST.txt](../PUBLIC_ALLOWLIST.txt)는 생성한 문서·metadata를 포함한 전체 목록, [PUBLIC_MANIFEST.json](../PUBLIC_MANIFEST.json)은 파일별 size·SHA256·원본 대비 변경 여부를 제공한다. Manifest는 자기 hash만 제외하며 그 파일의 존재는 allowlist에 기록한다.

## 3. 작성·수정한 문서

한국어 README와 `architecture`, `rl_pretraining`, `experiments`, `research`, `models`, `play`, `gui`, `fixed_evaluation`, `formal_random`, `learner_performance`, `WSL_TRAINING`, `WSL_REPRODUCIBILITY_REPORT`, `publication`, `licensing`, `evidence`, `PUBLIC_SANITIZATION`을 작성·정리했다. 기존 대응 문서명을 활용해 중복 내용을 줄였다. `devlog`는 원본을 보존하고 공개 사본을 만들었다. 역사적인 영문 제목은 원 기록의 식별을 위해 유지한다.

모델 Release manifest, dependency license inventory, 과거 Oracle 평가 CSV 2개와 출처 hash, 신규 WSL 기능 검증 JSON, Native의 Viet Nguyen MIT 원문과 출처도 추가했다. README와 vendor 안내를 포함해 한국어를 기본으로 작성했다.

## 4. 공개 대상 소스 목록

| 구분 | 최초 선별 파일 수 | 내용 |
| --- | ---: | --- |
| `src/tetris_agent` | 34 | RL·Planner·Executor·Oracle·추론·Resolver |
| `scripts` | 30 | 기존 설치·학습·평가·추론·진단 CLI |
| `tests` | 55 | 테스트 및 pre-perf-v2 독립 reference fixture |
| `experiments` | 6 | 81 records Registry와 Native profile 5개 |
| `requirements` | 4 | CPU/CUDA lock와 provenance |
| `vendor` | 23 | Native Python 19개·manifest·patch·LICENSE·안내 |
| `evaluation` | 4 | 선별된 256-state 고정 평가 fixture |
| 루트 설정 | 4 | `.gitattributes`, pytest 설정, requirements 2개 |

Python code·Registry/profile·Native source·lock은 기준 소스와 byte 동일하다. 기존 160개 중 사본 변경은 vendor README의 한국어 설명과 과거 `untrained_seed42.json`의 PC 경로 치환뿐이다. 평가 수치와 연구 알고리즘은 변경하지 않았다.

## 5. 제외한 비공개 자료

전체 Canonical·raw runs·과거 analysis/benchmarks, 외부 비공개 backup, venv/runtime/cache, archive/worktree 사본, `_migration`/`_validation`, Git objects/refs/reflog/config/hooks/bundles/LFS, 모든 checkpoint·모델·아이콘·이미지·EXE/DLL/ZIP, GUI 소스는 제외했다. 기존 자료를 삭제하거나 이동하지 않았다. 새 Smoke 모델과 raw 실행 로그도 공개 후보에 포함하지 않았다.

실제 PC 경로와 DEVLOG 근거 대응표, 모델 정적 검사, 보존 검사, 설치·테스트 로그는 비공개 감사 폴더에 있다. 공개 후보에 그 감사 폴더를 포함하는 절차는 없다.

## 6. README 및 WSL 실행 검증

공개 후보만 Linux 새 소스 디렉터리에 복사하고 Python 3.12.10으로 **새 CPU venv**를 만들었다. Windows 소스·개인 runtime·Canonical 모델을 실행 입력으로 사용하지 않았다. CPU lock 30개 wheel을 캐시 없이 실제 다운로드·설치했고, pip 설치 report의 30개 SHA256이 provenance와 정확히 일치했다. pip check도 통과했다. 기존 WSL host와 interpreter를 사용했으므로 OS부터 독립적인 제3머신 인증은 아니다.

| 검증 | 새 Phase 4 결과 |
| --- | --- |
| Native 설치·검증·runtime·import | PASS, 19개 source SHA256/profile 일치 |
| 테스트 | **803 PASS / 5 SKIP / 0 FAIL**, 전체 808 cases |
| 제외된 진단 | 기존 모델을 재업데이트하는 2개 모듈의 34 cases |
| Geometric Smoke | 32 transitions / 29 updates / 3 syncs, finite loss·weight 변화·3 checkpoint reload PASS |
| Oracle Smoke | 32 transitions / 29 updates / 3 syncs, finite loss·weight 변화·3 checkpoint reload PASS |
| 고정 상태 평가 | 새 모델 각각 256 states, 해당 geometric/oracle 조건으로 PASS |
| Frozen 추론 | 각 모델 6-decision 상한 rollout PASS, optimizer/replay updates 0, 모델·weights 불변 |
| 미배포 Registry 모델 | `MODEL_NOT_INSTALLED`, 예상 exit 2; 자동 fallback 없음 |
| 최종 후보와 검증 소스 | Python·Registry·locks 등 150개 파일 byte 일치 |
| 공개 문서 정적 검사 | 로컬 파일/anchor 링크 112개, 깨진 링크 0, fence 오류 0 |

실제 기능 결과와 새 checkpoint hash는 [PHASE4_VALIDATION.json](results/PHASE4_VALIDATION.json)에 있다. 32-decision Smoke는 기능 검증이며 500K/2M 성능·수렴·exact resume·CUDA 장기 재현성을 인증하지 않는다. 새 CUDA 학습은 수행하지 않았고 Phase 3의 CPU/CUDA 기능 근거는 별도 문서에 구분했다. 일반 6000-decision 학습 명령은 실행 예시로 제공했으며 이번에는 실행하지 않았다.

권한 미확정 이미지가 없어 README는 Mermaid 아키텍처를 사용한다. 공식 외부 링크 8개를 조회했다. GitHub 게시 전이므로 실제 GitHub 렌더·remote clone·Release 다운로드는 검증하지 않았다. 외부 페이지의 미래 가용성을 보장하지 않는다.

## 7. DEVLOG 최종 상태

Section 0~55의 연속 번호와 중복 부재, Takeaway, 부록 A~D, 실패·한계와 Phase 3 반영을 확인했다. 내부 원본 SHA256은 `d0333d3ef925729585ba8f51280c0d7a440975b926b1a6a5b06ed2c638cd0804`로 유지했다. 공개 사본의 Section 1~34+Takeaway 텍스트는 원본과 일치한다. Section 35~55의 가설·조건·관측·사후 해석 및 실행 날짜와 Commit 날짜 구분도 보존했다.

PC/WSL 절대 경로는 공개 사본에서 별칭으로 바꿨다. E01~E36의 원본 자료명·hash·비공개 상태는 [근거 안내](evidence.md)에 연결했다. 비공개 파일을 실제 제공하는 링크처럼 표현하지 않는다. 원본과 실제 경로 대응표는 비공개 감사에 남겼다. [사본 처리 내역](PUBLIC_SANITIZATION.md)

## 8. 모델 공개 계획

지원된 **4개 실험 / 9개 Checkpoint**의 원 파일 SHA256·size가 Registry와 일치했다. ZIP `data.pkl`의 pickle opcode와 문자열을 정적으로 조사했고 PC 경로·email·token 패턴은 0건이었다. Pickle 실행이나 모델 재저장은 하지 않았다. Tensor storage의 의미나 모든 개인정보 부재를 보증하는 검사는 아니다. 외부 config.json의 output_dir에 있는 개발 PC 경로는 배포 metadata에서 제외했다.

[MODEL_RELEASE_MANIFEST.json](../MODEL_RELEASE_MANIFEST.json)과 [모델 문서](models.md)에 seed, budget, checkpoint step/role, 학습 설정, 실제 평가 기록, SHA256, Native·소스 의존성, 권리 상태를 artifact별로 정리했다. Gameplay 기록이 확인되지 않은 artifact는 미확인으로 남겼고 periodic/final 결과를 자동 합치지 않았다. 현재 revision은 검증 소스 의존성이지 미저장된 역사 launch SHA를 확정한 값이 아니다.

권리자·모델 license 확인 후 허용된 artifact만 원 bytes 그대로 별도 Release Asset으로 준비한다. 현재는 모두 **HOLD**, download URL 없음, binary 미포함이다. CPU 추론 지원과 배포 허가는 별개다.

## 9. GUI 통합 계획

별도 공개 연구 저장소 → 후속 승인된 GUI 소스 저장소 → 허용된 모델 assets → Windows EXE Release로 연결한다. Monorepo를 선택한다면 후속 allowlist 감사·승인 후 GUI 소스를 새로운 commit으로 추가한다. 기존 private Git/LFS를 가져오지 않는다. `FrozenInference`의 ID/role/mode/hash/CPU 계약으로 연동하며 이번에는 GUI 코드를 수정하지 않았다.

GUI V2.4.3은 기존 빌드·Frozen 추론·자동 GUI 검증 PASS와 별개로 **공개 Beta NO-GO**다. 소스·모델·아이콘·Native 수정분의 권리, Qt 대응 소스·고지와 실제 무상 전달 위치, MSVC entitlement, Intel EULA/REDIST 조건, 실제 third-party coverage, 최종 Windows/SAC 검사와 게시·서명 판단이 남아 있다. 이번 연구 소스 검증으로 EXE를 공개 가능 상태로 승격하지 않았다. [GUI 계획](gui.md)

## 10. 라이선스·개인정보·보존 검사

프로젝트 LICENSE는 권리자 승인 전까지 작성하지 않았다. [라이선스 검토](licensing.md)에서 MIT/Apache-2.0 후보와 공식 조건을 비교했다. Native MIT 원 고지, Maximilian Weichart·Viet Nguyen attribution, 수정 사실, source hash를 보존하고 Viet Nguyen의 MIT 원문도 추가했다.

CPU 30개/CUDA 49개 설치본의 license metadata와 관련 고지 파일 162개/180개 inventory를 조사했다. [제3자 고지](../THIRD_PARTY_NOTICES.md)와 [dependency_licenses.json](../dependency_licenses.json)은 그 범위의 증거이며 wheel/EXE 재배포 승인이나 완전한 binary SBOM이 아니다.

공개 후보 전체에서 개인 PC 경로·email·API key/token·private key 패턴의 **미승인 탐지 0건**이다. 합성 path 거부 테스트의 2개 위치만 명시 예외로 유지했다. 이 결과는 모든 secret의 부재를 보증하지 않는다. 최대 파일은 Registry 766,593 bytes이며 100MB급 파일은 없다. 모델·이미지·binary·Git/LFS 미포함도 확인했다.

기존 보호 파일 **475개**의 SHA256·size·mtime, 기존 checkpoint **2,700 file occurrences**의 size·mtime를 대조해 변화 0건이었다. 지원 모델 9개는 재해시 9/9 일치했다. 전체 Canonical payload의 이번 전수 재해시나 전체 backup 복구 시험을 추가 수행했다고 주장하지 않는다. 보고서 신규 추가는 보호 검사와 구분해 기록했다.

## 11. 공개 전 남은 차단 항목

| 대상 | 필요한 결정·증빙 | 현재 상태 |
| --- | --- | --- |
| 소스·문서·fixture·Native 수정분 | 권리자·기여자 허락, 프로젝트 LICENSE, DEVLOG/집계/Registry 공개 범위 승인 | 게시 HOLD |
| GitHub 게시 | 별도 repository 이름·visibility·공개 author 정보·최종 파일 목록과 Push 승인 | 미게시 |
| 모델 9개 | 저작권·학습 소스 조건·license·개별 asset 범위·Release 승인 | HOLD |
| GUI 소스·아이콘 | 별도 allowlist·권리·license·interface 검증 | 미포함 |
| Windows EXE | V2.4.3의 source/binary 재배포 조건, 최종 Windows 검사, 게시 결정 | NO-GO |
| 재현성 범위 | 다른 host/GPU, Python build/OS 설치, 장기 성능·bitwise/exact resume | 이번 인증 범위 밖 |

## 12. 최종 판정

**GO WITH CONDITIONS (Phase 4 로컬 준비)**. 별도 스냅샷·한국어 문서·명시 allowlist·source/Native hash·새 WSL CPU 실행 검증·모델/GUI/라이선스 차단 항목·새 공개 root 이력의 게시 방법을 준비했다. 소스·모델 게시 허가는 HOLD, Windows Beta는 NO-GO다.

Private 저장소 Public 전환, Push, Release 게시, 기존 Git 이력 rewrite, 장기 GPU 학습, 기존 Checkpoint 재저장, 연구 알고리즘/GUI 변경, 무단 LICENSE 확정은 수행하지 않았다. [공개 절차](publication.md)에 따라 남은 권리 결정과 구체적인 게시 범위에 대한 사용자 승인을 기다린다.
