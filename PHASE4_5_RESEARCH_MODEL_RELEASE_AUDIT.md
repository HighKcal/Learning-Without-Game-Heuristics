# 테멍 Phase 4.5 — 연구 코드 및 Checkpoint 9개 공개 최종 감사

> 역사적 준비 보고서: 아래 승인 대기·미게시 판정은 Phase4/4.5 작성 시점의 기록이다. Phase5에서 사용자가source/document/fixture/model9개MIT와게시를승인했다. 현재조건은 [라이선스 안내](docs/licensing.md)와 [승인 기록](docs/RELEASE_APPROVALS.md)을 따른다.


작성일: 2026-10-14 (Asia/Seoul). **최종 판정: GO WITH CONDITIONS — 연구 소스와 원본 모델9개의 로컬 배포 준비·기술 감사 통과, 권리·라이선스·게시 승인 대기.** 프로젝트·모델 LICENSE는 확정하지 않았고 원격 게시도 하지 않았다. 확인된 라이선스 충돌이나 특정 모델만 기술적으로 제외해야 할 사유는 발견되지 않았다. 미입증 출처와 승인 항목은 해결됐다고 표시하지 않는다.

## 1. 기준·경로·보존

Phase4 기준 `Temeong_Public_Candidate_20261014`의192개 파일과 allowlist/manifest hash를 재검사한 뒤 **별도 `Temeong_Final_Public_Candidate_20261014`** 작업 사본을 만들었다. 검증 소스는 `codex/research-staging-integration` HEAD `3801e53c236262b8196880b1fb7392559e91b13a`다. Phase2·3 보고서와 Phase4 실행 증거, 모델 manifest, source/fixture/Native 계약을 함께 확인했다. 기존 main HEAD는 `0a07550bdd5a9a154486b48359ac00cf1344725e`이며 강제 전환하지 않았다.

모델은 소스 폴더 밖의 **`Temeong_Model_Release_Candidate_20261014`**에서9개 Asset으로 준비했다. 실제 PC 절대 경로·원본별 위치와 raw log는 내부용 감사 보고서에 기록한다. Public source에는 `.pt`가 없고 별도 Release candidate에만 있다. 전체 Canonical, raw research data, `.git`/LFS/private history, GUI/EXE는 포함하지 않는다.

Phase4 기존192개, 선택한 기존 연구·Staging·GUI 보호파일476개는 SHA256/size/mtime 일치다. 원 checkpoint9개도 SHA256/size/mtime가 변하지 않았다. main·통합·GUI의 기존 HEAD/status는 그대로다. 이 보존 검사는 전체 Canonical payload 재해시·전체 backup 복구 시험을 추가 수행한 인증은 아니다.

## 2. 연구 코드 저작권·라이선스 판정

사용자는 직접 기획하고 Codex로 DDQN 구조·학습 코드를 개발했으며 개인 PC에서 random initialization과 환경 생성 replay로 학습했다고 진술했다. 현재 `DQNTrainer`는 새 `QNetwork`를 만들고 target을 초기 복사하며 `train()`은 fresh-run만 허용한다. 코드·payload 조건은 진술에 부합한다. 사전학습 weights나 외부 데이터셋을 다운로드하는 학습 경로는 발견되지 않았다.

프로젝트 코드, 과거 프로젝트 reference fixture, 문서·선별 집계, 환경 생성256-state fixture, Native 상류 코드, dependency metadata를 분류했다. [CODE_PROVENANCE.json](CODE_PROVENANCE.json)은 파일별 분류·hash를 제공한다. 사람/AI 작성 부분을 줄 단위로 입증한 기록은 아니며 인터넷 전체 유사 코드·모든 과거 프롬프트 provenance의 전수 감사도 아니다. 사용자 보유 부분의 MIT 적용은 후보로 적합하지만 권리자·공동기여·미고지 차용/기관 계약 여부의 최종 확인과 grant 승인이 남는다.

현재 소스·weights만 분리 제공하는 구성에서 식별된 상류 MIT와 프로젝트 MIT 후보 사이의 충돌은 발견되지 않았다. 사용자에게 없는 제3자 권리를 임의로 재허가하지 않는다. 소스 import와 dependency wheel/EXE 재배포 조건을 구분했다.

## 3. Codex 생성 코드 공개 권리 검토

[OpenAI Terms of Use](https://openai.com/policies/terms-of-use/)는 법이 허용하는 범위에서 사용자와 OpenAI 사이의 Output 권리를 사용자에게 배정하며 제3자 권리 준수를 요구한다. [Service Terms §4](https://openai.com/policies/service-terms/)는 Codex 생성 코드에도 제3자·오픈소스 라이선스가 적용될 수 있다고 명시한다. 생성 코드가 자동 MIT가 되거나 독점적 저작권·비침해가 보장되는 것은 아니다.

2026-01-01 개인 Terms와 2026-09-29 Service Terms를 확인했다. API/조직 계정은 [Services Agreement](https://openai.com/policies/services-agreement/) 등 실제 사용 계약을 적용해야 한다. 계정별 계약·기관 약정의 확인을 별도로 수행하지 않았으며 개인 계정의 면책을 가정하지 않는다. 이 작은 테트리스 DDQN이 OpenAI 경쟁 모델이라는 근거는 현재 범위에서 발견되지 않았다는 해석을 기록한다. OpenAI weights/output을 증류한 모델로 취급하지 않는다. [상세 권리 감사](docs/licensing.md)

## 4. 수정 Native·제3자 소스·Fixture

공식 `tetris-gymnasium0.3.1` wheel을 다운로드해 manifest의 SHA256을 확인하고19개 source와 원 LICENSE를 재비교했다. LICENSE는 byte동일하고, 의미 차이는 **기존 render dtype `np.integer→int` 한 줄뿐**이다. Physics 변경은 없다. Copyright2024 Maximilian Weichart / game logic Copyright2020 Viet Nguyen의 MIT 원 고지·attribution, Viet Nguyen의 MIT 원문, 변경 patch를 유지했다.

Reference fixture24개를 PROVENANCE의 commit `f78017e0a1ac831c2fcd0bc74dcc0d546f19822e` Git object와 각각 hash 비교해 일치했다. Git 이력 자체는 복사하지 않았다. Native의 참고 Wiki/Stack Overflow 링크는 유지하며 그 사이트 전체 콘텐츠를 프로젝트 MIT로 재허가한 것은 아니다. [소스 provenance 검사](docs/results/PHASE4_5_SOURCE_PROVENANCE_CHECK.json)

Phase4의 CPU30/CUDA49 dependency metadata·license-file inventory를 유지했다. Package wheel·Python runtime·바이너리를 배포하지 않는다. GUI/Qt/MSVC/Intel 등의 EXE 조건은 이번 범위에서 닫지 않았다. [제3자 고지](THIRD_PARTY_NOTICES.md)

## 5. Checkpoint9개 개별 감사

| Experiment / artifact | Seed / budget / step | Role | Bytes | SHA256 |
| --- | --- | --- | --- | --- |
| `geometric.baseline.seed123.100k` / `final` | 123 / 100000 / 100000 | final | 5321161 | `c882fa1bd96f7493cb1bf9ef0149b3b126d524d777ac22396bcf59b8184321b9` |
| `geometric.penalty020.seed42.500k` / `final` | 42 / 500000 / 500000 | final | 5321225 | `8124cb9a9ebd54461967213c1b40744f399d8f30ca4521470a3f4fc563b4d0ff` |
| `oracle_mask.seed123.500k` / `step_500000` | 123 / 500000 / 500000 | periodic | 5322057 | `f27eb283f952a10f80473458d01eb341f051d0358733fe2a7739205b62cade76` |
| `oracle_mask.seed123.500k` / `final` | 123 / 500000 / 500000 | final | 5322057 | `cd4d5d9423a986804acecbe7a7b48ae440db6ada5598d931e82bb680c656f146` |
| `oracle_mask.seed42.2m` / `step_300000` | 42 / 2000000 / 300000 | periodic | 5322057 | `8fb9e7e69a2fc9cbe7b2c38a8e5babff72c4c9387ca5fb6757a5fd963cf6e337` |
| `oracle_mask.seed42.2m` / `step_500000` | 42 / 2000000 / 500000 | periodic | 5322057 | `f0de04cab322f7670512f8004505c36c5fe98839f79438b8002d1b6131d41ed9` |
| `oracle_mask.seed42.2m` / `step_1000000` | 42 / 2000000 / 1000000 | periodic | 5322057 | `742a963a7e24a14f61c5159bad256c414c283d87af01f844517a792e56e48f8c` |
| `oracle_mask.seed42.2m` / `step_2000000` | 42 / 2000000 / 2000000 | periodic | 5322057 | `7933ccfc4dbfee1064b3d3c13e34ed4eeb4f544919a682642f43fcd6eb9e4e34` |
| `oracle_mask.seed42.2m` / `final` | 42 / 2000000 / 2000000 | final | 5322057 | `ada68fbe3e5c6e68775212f906e60ab031aa0d27948c0d833643bffafe457ab2` |


| Experiment / artifact | 원본 Registry 상대 경로 | Release Asset |
| --- | --- | --- |
| `geometric.baseline.seed123.100k` / `final` | `runs/geometric/baseline/seed_123/100k/checkpoint_final.pt` | `Temeong-geometric-baseline-seed123-100k-final.pt` |
| `geometric.penalty020.seed42.500k` / `final` | `runs/geometric/penalty_020/seed_42/500k/checkpoint_final.pt` | `Temeong-geometric-penalty020-seed42-500k-final.pt` |
| `oracle_mask.seed123.500k` / `step_500000` | `runs/oracle_mask/seed_123/500k/checkpoint_000500000.pt` | `Temeong-oracle_mask-seed123-500k-periodic-500000.pt` |
| `oracle_mask.seed123.500k` / `final` | `runs/oracle_mask/seed_123/500k/checkpoint_final.pt` | `Temeong-oracle_mask-seed123-500k-final.pt` |
| `oracle_mask.seed42.2m` / `step_300000` | `runs/oracle_mask/seed_42/2m/checkpoint_000300000.pt` | `Temeong-oracle_mask-seed42-2m-periodic-300000.pt` |
| `oracle_mask.seed42.2m` / `step_500000` | `runs/oracle_mask/seed_42/2m/checkpoint_000500000.pt` | `Temeong-oracle_mask-seed42-2m-periodic-500000.pt` |
| `oracle_mask.seed42.2m` / `step_1000000` | `runs/oracle_mask/seed_42/2m/checkpoint_001000000.pt` | `Temeong-oracle_mask-seed42-2m-periodic-1000000.pt` |
| `oracle_mask.seed42.2m` / `step_2000000` | `runs/oracle_mask/seed_42/2m/checkpoint_002000000.pt` | `Temeong-oracle_mask-seed42-2m-periodic-2000000.pt` |
| `oracle_mask.seed42.2m` / `final` | `runs/oracle_mask/seed_42/2m/checkpoint_final.pt` | `Temeong-oracle_mask-seed42-2m-final.pt` |


원파일은 private Staging의 위 Registry 상대 경로에서 읽었다. 전체 원본 절대 위치는 내부용 보고서에만 있다. 이름 충돌을 피하는 Release Asset명을 부여하고 **byte복사**했으며 torch.save/원본재저장/변환은 수행하지 않았다. 합계47,896,785bytes이고 누락0개다.

각파일은 online331,808 float32 parameters의6개 tensor, target6개, Adam step/exp_avg/exp_avg_sq18개로 총30개finite tensor를 가진다. Geometric은16개, Oracle은22개 top-level key이며 config/epsilon/counters/metrics/timestamps가 있다. Dict/OrderedDict/list/tuple/scalar/Tensor 외 임의 사용자 객체를 발견하지 못했다. Replay·RNG·완전한 Native episode 상태는 없고 정확resume를 지원하지 않는다.

검사 순서는 원본·Registry SHA256/size 확인 → ZIP inventory·pickle opcode/GLOBAL·문자열 정적 검사 → `get_unsafe_globals_in_checkpoint` → 신뢰된 원bytes의 `weights_only=True, map_location=cpu, mmap=True` 제한 로딩 → type·shape·finite·config/seed/budget/step·metadata 문자열 검사다. 미허용 GLOBAL0, 개인path/email/token pattern0건이다. 일반pickle실행·weights_only=False·추가safe_globals 허용은 사용하지 않았다. Static scan과 제한 로딩은 모든 DoS/downstream 문제나 tensor 내 개인정보의 부재를 입증하지 않는다. [보안 범위](docs/CHECKPOINT_SECURITY.md)

사용자 진술은 from-scratch 학습이며 현재 구현과 일치한다. 각 historical launch의 초기weights·완전source SHA·replay/RNG는 저장되지 않아 그 전부를 파일로 증명하지는 못한다. 제3자 pretrained weights·외부 dataset은 발견되지 않았다. 모델9개 권리 상태는 **READY_PENDING_APPROVAL**, MIT 우선 후보/미확정이다. [개별 model cards](docs/MODEL_CARDS.md)와 [manifest](MODEL_RELEASE_MANIFEST.json)에 실제 학습 설정·과거/이번 평가·사용조건·객체 inventory를 기록했다.

## 6. 원본 배포와 추론 전용 변환

원본은 기존 Registry/schema/hash·periodic/final·연구 metadata를 보존해 즉시 호환된다. Online-only `.pt`나 safetensors는 target/Adam을 줄이고 후자의 포맷을 제한할 수 있지만 새 loader/manifest/hash·승인과 inference 동등성 검증이 필요하다. Online raw weights1,327,232bytes는 계산값이며 새 파생파일을 저장한 크기가 아니다.

변환은 실행하지 않았다. 향후 승인시 tensor key/dtype/shape/raw hash,256-state Q/mask/greedy action/tie rule, 같은Native/CPU/seed rollout의 비시간 의미론·trajectory digest·witness/fallback을 비교한다. 원본9개를 대체하거나 같은 step의 periodic/final을 합치지 않는다. [형식 비교](docs/CHECKPOINT_FORMAT_OPTIONS.md)

## 7. MIT 및 모델 라이선스 후보

우선 제안은 사용자 보유 범위의 코드·문서·fixture MIT와 모델9개의 별도 명시 MIT다. MIT는 수정·재배포·상업적 사용을 허용하고 copyright/permission 고지를 유지해야 한다. 모델 Apache-2.0 또는 CC BY4.0, 특정 파일만 보류도 비교했다. Apache의 특허/고지와 CC BY의 attribution·변경표시 등 선택한 조건을 model card와 Release companion license에 명시해야 한다. [공식 MIT](https://opensource.org/license/mit)·[공식 Apache](https://www.apache.org/licenses/LICENSE-2.0)·[공식 CC BY](https://creativecommons.org/licenses/by/4.0/)

학술인용은 MIT의 추가강제조건으로 붙이지 않는다. 비상업 조건을 원한다면 MIT/Apache/CC BY가 맞는 선택은 아니다. Root LICENSE와 실제 MODEL_LICENSE는 아직 없으며 검토초안은 효력없음으로 표시했다. [모델license선택](docs/MODEL_LICENSE_OPTIONS.md)·[사용자승인목록](docs/RELEASE_APPROVALS.md)

## 8. GitHub Release Asset 구성

소스에는 모델 binary를 넣지 않고, 고유 이름의 개별 `.pt`9개를 별도 Release 후보에 준비했다. Companion은 `SHA256SUMS`, `MODEL_RELEASE_MANIFEST.json`, 한국어 `MODEL_CARDS.md`, 승인 후 작성할 `MODEL_LICENSE.txt`다. 현재는 미확정 상태를 설명하는 `MODEL_LICENSE_STATUS.md`만 있다. 실제 Release URL/tag가 없어 다운로드 링크를 만들지 않았다.

기존 Private 이력을 보존하는 새 공개 저장소의 root commit에서 소스를 게시하고, 승인된 tag에9개 Asset을 연결하는 설계다. `install_model_assets.py`는9개 목록·role/path/size/hash를 기존 Registry와 대조한 뒤 새 artifact root에 배치한다. 모델이 없어도 기존 Smoke와 새 학습은 가능하다. [배포 가이드](docs/MODEL_RELEASE_GUIDE.md)

## 9. 실제 로컬 배포 모의 검증

9개 원본 → Windows 별도 Asset 폴더 → WSL 다운로드 모사 폴더 → 새 Registry artifact root로 복사했다. 모든 단계에서 SHA256/size가 일치했다. 최종 후보 소스만 WSL 새 작업 폴더에 놓고 Phase4에서 lock으로 설치한 CPU venv를 재사용했다. Ubuntu24.04 / Python3.12.10 / Torch2.14.1+cpu / 1 thread 기준이며 venv를 OS 보안 sandbox로 취급하지 않는다.

| 검사 | 실제 결과 |
| --- | --- |
| Asset 복사·배치 | 9/9 hash/size 일치, periodic/final 분리 |
| Native | 19개 source/profile hash 검증 PASS |
| 모델 평가 | 9개×256 states=2304 first choices, 모두 실행 성공 |
| Registry frozen 추론 | 9개×2 지원 mode=18회 PASS, seed100000/limit6 |
| 모델 보존 | file·weights 불변, gradients 없음, optimizer/replay updates0 |
| 부정 검사 | 기존 root·변조 Asset·manifest/Registry 불일치 거부 PASS |
| 집중 계약 suite | **75 PASS / 0 SKIP / 0 FAIL** |
| Fresh 학습 | 모델 입력 없이 geometric/oracle 각32 transitions/29 updates/3 syncs·reload PASS |

고정 상태의 선택 U는 baseline48/256(18.75%), penalty22/256(8.59375%), Oracle7개0/256이다. 이 비율은 training U-rate/gameplay trap 비율이 아니다. Oracle0은 physical mask 효과이며 Decision feasibility 학습 성공이 아니다. 6-decision/placement 추론은 성능 대규모 평가가 아니다. 장기 학습·다른 OS/GPU·bitwise/exact resume는 미인증이다. [실제 검증 결과](docs/results/PHASE4_5_RELEASE_VALIDATION.json)

기존 full suite803PASS/5SKIP은 Phase4 결과다. Phase4.5에는 연구 알고리즘 byte 변경이 없어 위 집중 suite와 실제9개 모델 실행을 추가했다. 새 full suite를 실행했다고 표현하지 않는다.

## 10. README·보안·최종 후보 검사

README를 한국어로 유지하고 Geometric의 목표 선택 후 target Planner 경로와 Oracle의 physical mask 선택 후 witness 재사용 경로를 별도 subgraph로 표시했다. From-scratch 사용자 진술과 미저장 history 한계,9개 모델 목록·배치 helper·추론/평가 명령·미확정 model license·GUI 제외/후속 계획을 추가했다. 모델 없는 새 학습 안내를 유지했고 가짜 다운로드 URL은 없다.

최종 소스 후보는 **206개 파일 / 약3MB**다. 기존 핵심 소스34개와 기존 scripts/tests·Registry/profile·Native hash/고지는 byte 동일하다. 새 helper는 배치용이며 학습 알고리즘·FrozenInference·Registry 계약을 변경하지 않는다. Allowlist가 전체 파일과 정확히 일치하고, 자기참조 Manifest를 제외한205개 파일 hash 및203개 provenance records를 검증했다.

문서의 로컬 파일/anchor 링크150개에서 깨진 링크0, fence 오류0이다. 개인 path/email/API key/token/private key 패턴의 미승인 탐지0건이며, path 거부 테스트의 합성 Windows 경로2개 위치만 예외로 명시했다. 모델·금지 binary·Git/LFS/private history가 소스 후보에 없는 것도 확인했다. 결과는 검사 패턴과 선별된 보호 범위에 한정한다. Tensor 의미에서 개인정보가 추출될 수 없다는 증명을 수행한 것은 아니다.

GitHub 렌더·실제 remote 다운로드·remote Actions는 게시 전이므로 미실행이다. GUI/EXE 소스 수정·배포 조건 해결은 이번 범위 밖이다.

## 11. 남은 차단 항목과 사용자 결정

1. 권리자 표시(제안 `2026 HighKcal`)와 코드/문서/fixture·Native 수정분의 보유 공개 권리 범위를 확정한다. 공동저작·기관/고용 계약·미고지 차용이 있다면 해당 조건을 반영한다.
2. 코드·문서·fixture MIT와 모델9개의 MIT/별도 license/파일 보류 선택을 승인한다. 약관의 권리 배정만으로 법률상 저작권 성립·비침해를 인증하지 않는다.
3. 원 training snapshot의 target/Adam·config/counters/timestamps 전달과 공개 DEVLOG/집계/Registry 범위를 승인한다. 원본 대체 변환은 별도 승인이다.
4. 새 repository 이름/visibility·공개 author 정보·Release tag·최종 source/model license를 확정하고 실제 Push/Release 게시를 별도로 승인한다. 기존 Private Public 전환과 이력 복사는 설계에 없다.

현재 기술적 변조·미고지 pretrained data·상류 license 충돌이 발견돼 특정 모델만 NO-GO인 상태는 없다. 역사 launch와 줄별 origin의 미입증 범위를 문서화했으며, 실제 알려진 권리 문제가 있다면 해당 파일을 보류한다. 보유 권리와 license 선택을 확정하기 전 grant를 발행하지 않는다. [구체적인 한국어 승인표](docs/RELEASE_APPROVALS.md)

## 12. 최종 판정

**GO WITH CONDITIONS — 로컬 공개 준비·모델9개 기술 감사/모의 배포 통과, 권리·라이선스·게시 승인 대기.** Phase4 원본 후보와 연구/Staging/Canonical·GUI·원 Checkpoint를 보존했고9개 모두 별도 Asset으로 준비했다. 소스·모델 LICENSE는 승인 후만 확정한다.

Push/Public 전환/Release 게시/Git 이력 rewrite/모델 변환·재저장/연구 알고리즘 변경/GUI 변경은 수행하지 않았다. 승인 후 실제 LICENSE·모델 license/권리자 표시를 적용하고 manifest/companion 고지의 hash를 갱신한 뒤, 승인된 대상과 tag에만 게시할 수 있는 준비 상태다.
