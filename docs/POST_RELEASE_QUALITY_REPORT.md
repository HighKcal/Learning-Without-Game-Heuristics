# research-v1.0.1 준비: Post-Release Quality Pass

검증일: 2026-10-14 (Asia/Seoul). 연구 알고리즘·성능 변경 없이 공개 저장소의 설치·테스트·CI·문서 사용성을 개선했다. 새 로컬 clone의 `codex/post-release-polish` 브랜치에서 작업했으며 원격 Push·설정 변경·Release 게시를 수행하지 않았다.

## 기준과 Release 보존

| 항목 | 확인 값 |
| --- | --- |
| 저장소 | [HighKcal/Learning-Without-Game-Heuristics](https://github.com/HighKcal/Learning-Without-Game-Heuristics) |
| 작업 기준 main | `130f0020e020673adc66184d73be4306912a3f8c` |
| 기존 research-v1.0.0 tag commit | `a9d24728e45fe2bf773d60a22d27635c8969e473` |
| 기존 Release | [research-v1.0.0](https://github.com/HighKcal/Learning-Without-Game-Heuristics/releases/tag/research-v1.0.0), ID `412235349` |
| 기존 Asset | 원본 모델 9개 + companion 4개 = 13개 |
| 기존 모델 합계 | 47,896,785 bytes |

기존 tag는 최초 공개 commit에 남아 있다. tag 이후 main의 세 commit은 다음과 같다. 연구 소스·Registry·Native·Fixture 변경은 없었다.

| Commit | 반영 내용 |
| --- | --- |
| `d2fd625` | 공개 checkpoint Release·다운로드 경로 문서화 |
| `53c574c` | 실제 게시 Asset의 canonical URL 사용 |
| `130f002` | Windows/Linux checkout 사이 공개 metadata checksum·줄바꿈 보존 |

`git diff research-v1.0.0..130f002`는 `.gitattributes`, `CODE_PROVENANCE.json`, `PUBLIC_MANIFEST.json`, `README.md`, `docs/CHECKPOINT_SECURITY.md`, `docs/MODEL_RELEASE_GUIDE.md`, `docs/evidence.md`, `docs/publication.md`의 8개 파일이다. 이 후속 개선은 기존 tag에 포함되지 않는다. 기존 tag를 이동하지 않고 다음 소스 패치 Release로 묶는 것이 적절하다.

GitHub API에서 기존 Release/Asset의 ID·이름·size·digest·updated_at를 읽어 작업 전후 비교했고 13개 모두 동일했다. 모델 manifest의 이름·size·SHA256도 API digest와 9개 모두 일치했다. 원격 main/tag도 작업 기준과 동일했다. 이번 작업은 모델 binary를 내려받아 재평가하거나 재저장하지 않았다. 기존 Release의 다운로드 검증과 이번 metadata 보존 검증은 별개다.

## 변경 파일

| 파일 | 변경 목적 |
| --- | --- |
| [tests/conftest.py](../tests/conftest.py), [pytest.ini](../pytest.ini) | 역사 입력 의존 case marker와 `--run-historical-diagnostics` opt-in |
| [.github/workflows/research-cpu.yml](../.github/workflows/research-cpu.yml) | Ubuntu 24.04 CPU 공개 테스트·두 Smoke CI |
| [scripts/verify_public_integrity.py](../scripts/verify_public_integrity.py) | Git/allowlist/manifest/provenance 경로·bytes·SHA256 읽기 전용 검사 |
| [.gitattributes](../.gitattributes) | 새 Workflow의 LF checkout 지정 |
| [README.md](../README.md) | 기본 테스트, CPU CI 결과 링크, 일반 학습 예시, 기존 Smoke 기능 결과, 패치 계획 |
| [WSL_TRAINING.md](WSL_TRAINING.md) | 수동 `--ignore` 제거, 역사 진단 실행 방법·입력 계약, 현재 공개 상태 |
| [gui.md](gui.md), [models.md](models.md), [publication.md](publication.md) | 소스·모델 공개 완료와 GUI/EXE 미공개 구분 |
| [experiments.md](experiments.md) | 기존 연구 결과와 새 기능 검증 구분 |
| [devlog.md](devlog.md) | 상단 현재 공개 상태 안내만 추가; 역사 본문·부록 보존 |
| [PUBLIC_ALLOWLIST.txt](../PUBLIC_ALLOWLIST.txt), [PUBLIC_MANIFEST.json](../PUBLIC_MANIFEST.json), [CODE_PROVENANCE.json](../CODE_PROVENANCE.json) | 새 파일 목록과 현행 bytes/hash 갱신 |
| 이 보고서 | 새 검증·한계·패치 Release 계획·설정 추천 기록 |

공개 파일은 207개에서 211개로 늘어난다. 새 파일은 conftest·Workflow·무결성 검사·보고서 4개다. 기존 manifest의 `.gitattributes`는 candidate hash와 현재 hash가 다른데도 동일성 flag가 true였으므로 실제 비교 결과 false로 바로잡는다. 과거 candidate hash·연구 provenance는 보존한다. Manifest 자체는 자기 hash에서 제외하고 나머지 210개 파일을 검사한다. Provenance는 control 3개를 제외한 208개 파일을 분류한다.

## 공개 테스트와 역사 진단

기존 두 진단 파일과 진단 스크립트를 수정하지 않았다. 저장된 역사 manifest가 그 implementation hash를 검증하기 때문이다. conftest에서 fixture 의존성 및 직접 과거 파일을 읽는 case를 구분한다.

| 범위 | Cases | 실행 정책 |
| --- | --- | --- |
| 기존 두 진단 모듈의 공개 단위 검사 | 12 | 기본 실행 |
| 역사 입력을 요구하는 진단 | 22 | 기본 skip, 명시 opt-in |
| 전체 수집 | 842 | 공개 820 + 역사 22 |

```bash
pytest -q
pytest -q -m "not historical_diagnostic"
pytest -q --run-historical-diagnostics -m historical_diagnostic
```

역사 입력의 checkpoint·trap·trace·state·episode·schedule·RNG·manifest는 공개되지 않았으며 공개 모델 9개로 대체할 수 없다. 자료를 가진 보관자의 격리 checkout용 실행 방법과 경로 계약은 [WSL 가이드](WSL_TRAINING.md#4-공개-테스트와-역사-진단)에 있다. 이 작업에서 비공개 자료는 읽거나 복사하지 않았다.

기본 `pytest -q`는 **815 passed, 27 skipped in 361.85s**, exit 0이었다. Skip은 역사 진단 22개와 CPU에서 실행할 수 없는 기존 CUDA 검사 5개다. 공개 테스트 PASS를 전체 역사 진단 PASS로 표현하지 않는다.

`-m historical_diagnostic` 기본 실행은 **22 skipped, 820 deselected**. `--run-historical-diagnostics -m historical_diagnostic --maxfail=1`은 누락된 `analysis/geometric/feedback_timing/manifest.json`에서 exit 1, **1 error, 820 deselected**였다. Feedback timing 모듈만 opt-in하면 기존 loader 경로의 seed42 100K checkpoint 누락에서 exit 1, **1 error, 2 deselected**였다. 이는 두 모듈 모두 opt-in이 실제 실행을 활성화하고 자료 누락을 오류로 드러내는 검사이며 역사 진단 PASS가 아니다.

## 새 로컬 검증과 CI

새 Ubuntu 24.04.4 WSL2 venv / CPython 3.12.10 / pip 25.0.1 / Torch 2.14.1+cpu에서 공개 lock의 30개 패키지를 hash 검증으로 설치했다. 기존 연구 runtime을 복사하지 않았다. CPU tensor 합계 검사 3680.0, CUDA runtime 없음. Windows의 새 clone을 WSL에서 읽어 검증했다. 새 OS 설치나 GitHub hosted runner 실행을 인증하는 결과는 아니다.

| 검사 | 결과 |
| --- | --- |
| CPU hash-lock 새 venv 설치 / pip check | PASS / No broken requirements |
| 수정 Native 설치 / profile 검사 | PASS / 19개 source SHA256 일치 |
| 기본 `pytest -q` | PASS / 815 passed, 27 skipped / exit 0 |
| Geometric 새 Smoke | PASS / 32 decisions·32 transitions·29 updates·3 sync |
| Oracle 새 Smoke | PASS / 32 decisions·32 transitions·29 updates·3 sync |
| 실제 optimizer | 두 모드 모두 weight 변화 > 0, finite losses 29개, 6 parameter의 Adam step 모두 29 |
| 새 checkpoint 재로딩 | 두 모드 각각 periodic16·periodic32·final 3개 모두 exact weights reload |
| Workflow 정적 문법 | actionlint 1.7.12 PASS; exit 0 |
| Workflow run 단계 | bash 문법 전부 PASS; 마지막 요약 step을 두 실제 Smoke JSON으로 실행하여 PASS |
| GitHub Actions 원격 실행 | 미실행: 로컬 브랜치를 Push하지 않았음 |
| 공개 inventory | PASS / Git·allowlist 211개·manifest hash 210개·provenance 208개 일치 |
| README·지정 문서 로컬 링크/anchor | PASS / 74개; 공개 모델 URL 9개는 실제 Release Asset 이름과 일치 |

Smoke는 기존 `scripts/smoke_train.py`를 그대로 실행했다. Production learning_starts=5000을 유지하며 Smoke에만 warmup4/batch4/replay64/sync8을 사용한다. 새 출력만 만들었으며 원 checkpoint를 입력으로 사용하지 않았다. 설치·실제 update·저장·재로딩 기능 검증이다. 장기500K/2M 성능, 수렴, CUDA bitwise 재현, exact resume, Decision의 feasibility 학습을 인증하지 않는다. Oracle의 U=0은 외부 physical mask의 효과라는 기존 해석을 유지한다.

Workflow는 `push`(main/작업 브랜치), `pull_request`(main), 수동 실행을 지원한다. `contents: read`만 허용하고 checkout credentials를 저장하지 않으며 timeout은 30분이다. actions/checkout v7과 setup-python v7은 upstream tag를 조회한 full commit SHA로 고정했다. Python 3.12.10을 확인한 후 별도 venv를 만들고 hash-lock·Native·테스트·Smoke를 순서대로 실행한다. 마지막 step은 두 `smoke_evidence.json`의 PASS·32/29/3·3 checkpoint reload를 검사하고 GitHub 실행 요약에 기록한다. 모델·Canonical 다운로드, tag/Release 조작, 기존 checkpoint 저장 단계는 없다.

설계 기준: [checkout 공식 문서](https://github.com/actions/checkout), [setup-python 공식 문서](https://github.com/actions/setup-python), [GitHub Workflow 문법](https://docs.github.com/en/actions/reference/workflows-and-actions/workflow-syntax), [actionlint](https://github.com/rhysd/actionlint). README에는 [실제 실행 결과 링크](https://github.com/HighKcal/Learning-Without-Game-Heuristics/actions/workflows/research-cpu.yml)를 준비했고 원격 성공 배지를 미리 표시하지 않았다. 첫 원격 실행 성공 여부는 승인 후 게시할 소스에 대해 확인해야 한다.

로컬 raw 로그·JSON과 새 Smoke checkpoint는 ignored `_validation/` 및 `outputs/`에만 남기며 공개 목록에 넣지 않는다. 로그의 개발 PC 경로도 공개 보고서에 복사하지 않는다.

## 연구 무결성

기준 main에서 `git show`로 읽은 각 원본 blob과 작업 파일 raw bytes를 비교했다. 아래 hash는 정렬된 `path + NUL + 파일 SHA256 + LF`를 연결한 값의 SHA256이다. 기존 source hash의 별도 감사 값이며 새 파일을 포함하지 않는다.

| 기존 공개 범위 | 파일 수 | 비교 | 집계 SHA256 |
| --- | --- | --- | --- |
| src/ | 34 | byte 동일 | `108a3639b4a750eb5c4e03c8d6fc296c4a97ce0a5bb4dee86fd3560fd84a4a49` |
| scripts/ | 31 | byte 동일 | `c3c56454deaf63b6d98232556e69024a9ca9c05e651d12a981a485327a41c0bf` |
| tests/ (기존 두 진단·reference fixture 포함) | 55 | byte 동일 | `ef13935c434b3696a9b12ecac824b7056fd609168a30761cf268b9f4f73a6008` |
| experiments/ (Registry·profiles) | 6 | byte 동일 | `6e1f57cc26d8ff13beea415ced93a10255fb26aeea9cf6ce84ca28a2b1fd3ceb` |
| vendor/ (Native 19개·고지 포함) | 25 | byte 동일 | `7d604d44f830bc98c09ce98ad38c3556a016c3ef9d5e19daed27a4532858e162` |
| evaluation/ (frozen fixture) | 4 | byte 동일 | `451c86a236b9bad8a432addf74eb99859d6045a2537a83f9fc7d3fa7db335dfc` |
| requirements/ | 4 | byte 동일 | `9726423ce59aaf6dcf6aefc1854e238f492e19ac2ef7ccc1a685654c02ed66cd` |

DEVLOG에서 이번 상단 안내 1문단을 제거하면 기존 파일과 내용이 정확히 같다. 기존 연구 원본·Private 저장소·Canonical·Staging·checkpoint 9개·GUI/EXE는 이번 작업 대상이 아니며 변경하지 않았다. 알고리즘·reward·Planner·Executor·Oracle·Registry/profile support 계약을 유지했다.

## GitHub Description / Topics 추천

설정을 변경하지 않았다. 다음은 사용자 검토용 추천이다.

- Description: **수작업 게임 전략 휴리스틱 없이 DDQN이 배치 전략과 실행 가능성을 학습하는지 탐구하는 연구. BFS Planner·Oracle 비교, 재현 가능한 CPU 학습·테스트, 공개 모델 9개를 제공합니다.**
- Topics: `reinforcement-learning`, `double-dqn`, `ai-agent`, `reinforcement-learning-research`, `path-planning`, `tetris`, `pytorch`, `reproducible-research`.

README는 프로젝트 소개·핵심 질문/결과·한계·설치·일반 학습·Smoke·모델 다운로드/검증/추론·연구 문서·기본 테스트·CI 링크를 제공한다. 기존 Smoke 기능 증거의 32/29/3 JSON 예시만 추가했고 측정하지 않은 성능 수치를 만들지 않았다.

## research-v1.0.1 계획과 남은 gate

판정: **research-v1.0.1 소스 패치 검토·게시 승인 준비 가능**. 로컬 공개 CPU 테스트·Smoke·무결성 및 Workflow 정적 검증은 통과했다. 실제 GitHub hosted CI 성공은 아직 주장할 수 없다. 아래 절차는 계획이며 게시 명령을 실행하지 않았다.

1. 사용자가 로컬 diff·보고서를 검토한다. 승인된 브랜치 공개/PR 절차로 변경을 전달하고 main Push는 별도 승인 범위에서만 수행한다.
2. 정확한 최종 commit에서 GitHub Actions의 첫 hosted 실행을 확인한다. 실패하면 설치/설정 범위에서 해결하고 연구 코드·lock을 임의로 바꾸지 않는다.
3. 검토·CI 통과한 소스 commit에 새 `research-v1.0.1` tag를 만든다. 기존 `research-v1.0.0` tag/Asset은 이동·삭제·덮어쓰기하지 않는다. Tag 생성·Release 게시는 사용자 승인 후 수행한다.
4. 패치 Release에는 소스·문서·테스트 분리·CPU CI 개선과 검증 한계를 적는다. 모델 9개는 기존 [v1.0.0 모델 Release](https://github.com/HighKcal/Learning-Without-Game-Heuristics/releases/tag/research-v1.0.0), [모델 manifest](../MODEL_RELEASE_MANIFEST.json), [MIT 모델 LICENSE](../MODEL_LICENSE.txt), [model cards](MODEL_CARDS.md)를 참조한다. 새 모델 업로드·변환·재저장은 불필요하다.

제안 Release 제목: **research-v1.0.1 — 공개 테스트·CPU CI·문서 개선**.

제안 Release 설명: “연구 알고리즘과 공개 모델 bytes는 유지했습니다. 기본 pytest는 공개 검사와 비공개 역사 진단을 구분하며, Ubuntu 24.04 / Python 3.12.10 CPU Workflow가 hash-lock 설치·Native 19개 해시·공개 테스트·두 32-decision Smoke의 실제 optimizer update와 checkpoint 재로딩을 검사합니다. 소스·모델 공개 상태와 실행 안내를 정리했습니다. 모델 9개는 research-v1.0.0 Release에서 받으세요. 이 패치는 역사 진단 전체 통과, 장기 학습 성능·수렴·exact resume를 인증하지 않습니다.” 원격 CI 결과 문구는 실제 실행 후에만 확정한다.

남은 항목은 첫 hosted CI 확인, 사용자 게시 승인, 자료를 가진 보관자의 역사 진단 재실행(선택), 별도 GUI/EXE 공개 gate다. 연구 질문은 완전히 해결되지 않았다. 이 Quality Pass는 새 연구 성과나 GUI 공개 승격이 아니다.
