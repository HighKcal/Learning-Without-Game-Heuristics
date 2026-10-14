# 연구 코드와 모델의 권리·라이선스 감사

2026-10-14 기준이다. **사용자가 공개 권한을 보유한 연구 코드·문서·Fixture와 직접 학습한 Checkpoint9개에 MIT를 적용했다.** 2026-10-14 Phase5 최종 승인을 반영했다. 코드 기술 검증, 사용자 진술, 약관상 권리, 법률상 저작권 성립, 실제 배포 승인을 구분한다.

## 1. 사용자가 확인한 출처

사용자는 연구를 기획했고 DDQN 구조·학습 코드를 Codex로 개발했다. 사전학습 모델이나 fine-tuning 없이 무작위 초기 가중치에서 개인 PC로 강화학습했으며 replay는 게임 환경 상호작용으로 생성했다고 밝혔다. 이 진술을 감사 입력으로 기록한다. 현재 코드의 `QNetwork` 생성, `DQNTrainer`의 초기 target 복사, `train()`의 fresh-run 검사와 환경 transition 저장 구조가 이 설명에 부합한다. 다만 각 과거 launch의 최초 weights/replay/RNG/정확한 source SHA가 checkpoint에 없으므로 payload만으로 모든 역사 출처를 입증한 것은 아니다.

## 2. Codex 생성 코드

[OpenAI Terms of Use](https://openai.com/policies/terms-of-use/)의 Content 절은 법이 허용하는 범위에서 사용자와 OpenAI 사이의 Output 권리를 사용자에게 배정한다. 사용자 Input의 권리와 제3자 권리 준수 책임도 다루며, 다른 사용자·제3자 Output을 그 배정에 포함하지 않는다. 계약상 배정이 모든 AI 생성 부분의 저작권 성립·독점성·제3자 비침해를 보증하지는 않는다.

[Service Terms §4 Codex and Code Generation](https://openai.com/policies/service-terms/)는 Codex 출력에 제3자·오픈소스 라이선스가 적용될 수 있음을 명시한다. 따라서 Codex 사용 자체 때문에 프로젝트를 공개할 수 없다는 결론도, Codex 출력 전부가 자동 MIT라는 결론도 성립하지 않는다. AI 도움을 사용했다고 명시하고 식별된 상류 고지는 유지한다.

개인용 Terms(2026-01-01)와 Service Terms(2026-09-29)를 확인했다. API/조직 사용에는 [Services Agreement §4.1](https://openai.com/policies/services-agreement/) 등 해당 계약이 적용된다. 본 감사는 실제 계정의 계약·결제·별도 기관 약정을 조사하지 않았다. 생성 시점에도 2026년 약관이 적용됐는지와 조직을 대신한 개발 여부는 최종 권리 확인 범위다. 보상·면책 조항이 개인 계정에 자동 적용된다고 주장하지 않는다.

약관의 OpenAI 경쟁 모델 개발 제한도 확인했다. 이 프로젝트는 OpenAI weights/outputs를 학습 데이터로 수집하는 LLM이 아니라 게임 상태·환경 보상으로 학습하는 235→800 DDQN이다. 현재 증거에서는 경쟁 서비스 모델이라는 근거가 발견되지 않았다는 **프로젝트 범위의 해석**이며, 모든 모델 개발을 약관이 허용한다고 일반화하지 않는다.

## 3. 파일별 권리 범위

| 분류 | 대상 | 확인 결과와 남은 범위 |
| --- | --- | --- |
| 사용자 기획·Codex 도움을 받은 프로젝트 코드 | `src/tetris_agent`, 기존 scripts/tests | 사용자 출처 진술과 구현·이력 기준 일치; 사람/AI 작성분의 정확한 줄별 구분은 미입증 |
| 프로젝트 문서·실험 결과 | README, 한국어 docs, 선별 집계 | 수치·근거 보존, 개인경로 치환; 공개 범위·MIT 적용 사용자 승인 완료 |
| 내부 이전 코드 평가 fixture | `tests/fixtures/oracle_pre_perf_v2` | source commit/file hash와 byte 일치; 전체 Git 이력 없이 선별, 프로젝트 권리 범위의 MIT 적용 |
| 선별 상태 평가 fixture | `evaluation/fixed_eval_final_v1` | 환경에서 생성한 상태/label; 개인정보·외부 pretrained data 미발견; 전체 Canonical 미포함 |
| 수정 Native | `vendor/native_tetris_031` | 원 MIT 두 저작권자 고지 보존; render dtype 한 줄만 기존 수정 |
| 외부 설치 라이브러리 | Torch/NumPy/Gymnasium/OpenCV 등 | lock·metadata 고지 inventory, 각 상류 라이선스 유지; wheel/binary 미배포 |
| 학습 모델 | 직접 학습한 9개 snapshot | 외부 pretrained weights/data 미발견; payload와 사용자 진술 부합, 모델9개 별도 MIT 승인 완료 |

자세한 파일 분류·hash는 [CODE_PROVENANCE.json](../CODE_PROVENANCE.json)이다. 알려진 외부 코드·고지 참조를 검색하고 Native 원 wheel과 비교했다. 인터넷 전체 유사 코드 검색이나 모든 과거 프롬프트의 법적 provenance를 증명한 감사는 아니다. 미고지 차용·공동저작·기관 소유가 있다면 공개 전 분류와 권한을 갱신해야 한다.

## 4. 상류 코드·라이브러리

Native Copyright 2024 Maximilian Weichart / 일부 game logic Copyright 2020 Viet Nguyen의 MIT 원문·attribution을 보존한다. 상류 README/주석의 Wiki·Stack Overflow 링크는 참고로 유지하며 그 사이트의 모든 콘텐츠를 프로젝트 MIT로 재허가하지 않는다. 판정은 Native 배포본의 명시 MIT와 확인된 원 소스에 근거한다. render 수정·줄바꿈 hash 계약을 바꾸지 않았다.

패키지를 import하는 연구 코드와 패키지 wheel/바이너리의 재배포를 구분했다. 현 source-only/weights-only 구성에서 식별된 상류 MIT 고지와 프로젝트 MIT 후보의 충돌은 발견되지 않았다. 이 결과는 GUI/Qt/MSVC/Intel/EXE 재배포 조건에 적용하지 않는다. [제3자 고지](../THIRD_PARTY_NOTICES.md)·[dependency metadata](../dependency_licenses.json)

## 5. 적용한 MIT와 범위

사용자 최종 결정은 source/document/fixture MIT, 직접 학습한 원본 Checkpoint9개도 별도 MIT다. 권리자 표시는 Copyright (c) 2026 HighKcal이다. [프로젝트 LICENSE](../LICENSE)와 [모델 LICENSE](../MODEL_LICENSE.txt)를 제공하고 Release에도 model license를 동봉한다.

MIT는 수정·재배포·상업적 이용을 허용하며 해당 copyright·permission 고지를 유지해야 한다. 비상업 제한이나 학술 인용 의무를 추가하지 않는다. 원 Native의 두 저작권자 MIT, dependency의 상류 고지는 사용자 grant와 별도로 보존한다. [MIT 원문](https://opensource.org/license/mit)

이용약관상의 배정이나 사용자 승인이 모든 AI 생성 부분의 법률상 저작권 성립·독점성·비침해를 증명하지는 않는다. 미확인 역사 launch·전체 유사 코드 provenance의 한계는 위 감사 기록을 유지한다. [모델 사용 조건](MODEL_LICENSE_OPTIONS.md)·[Phase5 승인 기록](RELEASE_APPROVALS.md)
