# Windows GUI 테멍과 연구 저장소

연구 코드는 학습·평가·Planner/Executor·frozen inference를 제공한다. GUI는 별도 프로젝트에서 frozen inference worker와 사용자 표시를 관리한다. 이번 Phase 4에서는 GUI/EXE 소스를 수정하거나 복사하지 않았다.

현재 GUI V2.4.3의 기존 배포 보고서 판정은 **빌드·Frozen 추론·자동 GUI 검증 PASS / 공개 Beta NO-GO**다. 패키지 버전2.4.3과 앱 화면의 기존 표시2.4.0은 다르다. 공식 version/source 조건과 권한 확정 전에는 EXE 공개 가능으로 승격하지 않는다.

| 공개 구성 | 계획 | 현 상태 |
| --- | --- | --- |
| 연구·학습 코드와 문서 | 현재 후보에서 새로운 Git root 이력으로 독립 공개 | 준비·승인 대기 |
| GUI 소스 | 별도 allowlist/라이선스/개인정보 감사 후 별도 저장소; 검증된 research revision/hash와 interface 명시 | 이번 후보 미포함 |
| 모델 | 승인된9개 이내 개별 Release Asset+SHA256+model card | 권한 미확정·binary 미포함 |
| Windows EXE | GUI 저장소 Release; app/Qt source/고지/SBOM/모델 연동 | NO-GO |

Private 저장소와 전체 Git/LFS 이력은 그대로 보존한다. Public 연구→향후 Public GUI의 링크를 명시하고, `FrozenInference`의 ID/role/mode/hash 및 CPU support 계약으로 연결한다. GUI 소스를 monorepo에 넣고 싶다면 후속 감사·승인 후 선별해 새 commit으로 추가한다. Private repo를 submodule이나 mirror로 공개 프로젝트에 끌어오지 않는다.

EXE에는 소스 LICENSE와 다른 재배포 gate가 있다. 남은 항목은 소스·모델·아이콘·수정 Native 권리, Qt 대응 소스 revision/patch/build recipe와 실제 무상 전달 위치, MSVC 배포자 entitlement, Intel OpenMP EULA/REDIST·수령자 조건, 실제 바이너리 third-party coverage, 독립 새 Windows 및 Windows11 SAC/Explorer/작업표시줄/DPI 검사다. 서명·게시 판단도 별도다. 연구 CPU/WSL 검사로 이 조건을 닫지 않았다.

기존 GUI 보고서 원본은 비공개 자료로 보존하고 이 문서에는 기능 판정과 차단 조건만 정리했다. [공개 절차](publication.md)
