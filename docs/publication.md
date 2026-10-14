# 최초 공개 절차와 범위

2026-10-14 사용자는 HighKcal/Learning-Without-Game-Heuristics의main Push와research-v1.0.0 Release를승인했다. Source/document/fixture MIT와원본model9개별도MIT를적용했다. 저작권표시는Copyright (c) 2026 HighKcal이다. [승인 기록](RELEASE_APPROVALS.md)

최초 공개는 기존 Private Git/LFS 이력을 복사하지 않는 새 root commit으로 완료했다. Source는명시allowlist에있는파일만추적하고`.pt`모델은별도ReleaseAsset으로배포한다. 모델은originalbytes/SHA256/periodic-final구분을유지한다. Native의기존MIT두저작권자고지는그대로보존했다.

원연구·Staging·Canonical·기존후보·GUI·원Checkpoint는변경하지않는다. GUI/EXE/DLL·아이콘/이미지·venv/cache/backup/rawresearchdata는게시하지않는다. Git author는사용자승인HighKcal+noreply주소다.

research-v1.0.0 Release에는 원본 9개와 SHA256SUMS·MODEL_RELEASE_MANIFEST.json·MODEL_LICENSE.txt·한국어 MODEL_CARDS.md를 게시했다. Source공개후실제모델업로드·해시검증이완료된Release만downloadlink로안내한다. 실패한업로드를삭제·강제이력변경으로숨기지않고상태를기록한다.

사용방법은 [모델배포가이드](MODEL_RELEASE_GUIDE.md)·[보안안내](CHECKPOINT_SECURITY.md)·[WSL실행](WSL_TRAINING.md)에있다. 안정적인실행가능성학습은미해결이며Oraclemask효과와Decision학습성과를구분한다. [GitHubRelease공식안내](https://docs.github.com/en/repositories/releasing-projects-on-github/about-releases)

공개Repository: [Learning Without Game Heuristics](https://github.com/HighKcal/Learning-Without-Game-Heuristics). 실제Release: [research-v1.0.0](https://github.com/HighKcal/Learning-Without-Game-Heuristics/releases/tag/research-v1.0.0), 원본9개+companion4개게시완료. 기존 원격 다운로드·Clone 검증과 이번 패치의 새 검증은 구분한다. 패치 검증·main과 tag의 차이·research-v1.0.1 계획은 [Quality Pass 보고서](POST_RELEASE_QUALITY_REPORT.md)에 있다. v1.0.0 tag/Asset을 유지하며 새 main Push·Release 게시에는 사용자 승인이 필요하다.
