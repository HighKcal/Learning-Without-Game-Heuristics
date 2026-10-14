# 공개 파일 선별·비공개 정보 처리

기존 Phase4 후보192개는 그대로 보존하고 별도 최종작업사본을 만들었다. Phase4에서 integration revision `3801e53c`의160개 source/config/test/fixture/vendor 파일을 명시선별한 근거를 유지한다. Phase4.5에서는배치helper1개와한국어권리·보안·모델카드/감사문서·선별기능검증metadata를추가했다.

핵심 Python source와기존scripts/tests·Registry/profile·Native19개·requirementslock은Phase4bytes와같다. 새 `scripts/install_model_assets.py`는9개Asset/Registry의size/hash/role/path를검증해없는새artifactroot에byte복사한다. 연구학습·FrozenInference계약은바꾸지않는다. 최종파일은 [allowlist](../PUBLIC_ALLOWLIST.txt)·[manifest](../PUBLIC_MANIFEST.json)에있다.

Source에는`.pt`가없다. 모델9개는별도Release후보에서원bytes/hash를유지한다. `MODEL_RELEASE_MANIFEST.json`에공개필요metadata만넣고개발PC원본절대path·raw문자열검사표·실행log는비공개감사에남겼다. `docs/results/PHASE4_5_RELEASE_VALIDATION.json`도PCpath와개별state전체trace를제외했다.

DEVLOG 내부원본과Section1~55·Takeaway·부록은보존했다. Public사본의PC/WSLpath는별칭이고E01~E36비공개근거의이름/hash/제공상태는[evidence](evidence.md)에남겼다. 원비공개근거를실제공개한링크처럼표현하지않는다. 과거Oracle평가CSV2개는PCpath열/BOM만제거하고수치문자열/row순서를유지했다.

개인path/email/APIkey/token/privatekeypattern을최종source/doc/metadata전체에검사한다. Path거부test의합성Windows절대path2위치는예외로명시하고보존한다. 검사패턴에탐지없음이모든비밀정보/개인정보의부재를보증하지는않는다. Native와공개상류copyright이름·출처URL은attribution이므로유지한다.

전체Canonical·rawruns/analysis/benchmarks/archive·외부privatebackup·venv/runtime/cache·과거worktree·Git/LFS/historybundles·이미지/icon·GUI/EXE/DLL은제외했다. 원본자료를삭제·이동하거나model을재저장하지않았다. Phase5승인으로project/model MIT가적용됐고root LICENSE·MODEL_LICENSE.txt를제공한다. 검토초안은게시사본에서제외했다.
