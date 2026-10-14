# 모델9개 목록과 개별 감사

4개 실험의9개 원 training snapshot을 모두 준비했다. 합계47,896,785bytes이며 source snapshot에는 binary가 없다. Release 후보 폴더에만 byte복사했고 원본을 수정·재저장하지 않았다. 각 지원모드와 Native hash·Registry contract는 유지했다.

모든 파일은331,808개의 online float32 parameter, 6개 online tensor·6개 target tensor·18개 Adam tensor(총30개)를 포함한다. 저장 객체는 dict/OrderedDict/list/tuple/scalar/Tensor이며 architecture class·replay·RNG는 저장하지 않는다. Geometric payload는16개 top-level key, Oracle은22개다. Global 참조는 `collections.OrderedDict`, `torch.FloatStorage`, `torch._utils._rebuild_tensor_v2`이고 추가 허용이 필요한 GLOBAL은0건이다. 제한 로딩·모든tensor finite·shape·seed/budget/step 검사와 metadata 문자열에서 개인path/email/key 패턴0건을 확인했다. 가중치가 개인정보를전혀내포하지않는다는통계적추출증명은아니다.

사용자 진술과 현재 코드로는random initialization/self-generated replay가부합한다. 각역사launch의initial weights/replay/source SHA가완전히저장되지는않아이를파일내용만으로증명하지않는다. 제3자pretrained weights/data는발견되지않았다. 권리범위·modellicense는 [승인 목록](RELEASE_APPROVALS.md)을결정한후확정한다.

| Experiment / artifact | Seed / budget / 저장 step | Role | Bytes | 이번 고정256-state 선택U | SHA256 |
| --- | --- | --- | --- | --- | --- |
| `geometric.baseline.seed123.100k` / `final` | 123 / 100000 / 100000 | final | 5321161 | 48/256 (18.7500%) | `c882fa1bd96f7493cb1bf9ef0149b3b126d524d777ac22396bcf59b8184321b9` |
| `geometric.penalty020.seed42.500k` / `final` | 42 / 500000 / 500000 | final | 5321225 | 22/256 (8.5938%) | `8124cb9a9ebd54461967213c1b40744f399d8f30ca4521470a3f4fc563b4d0ff` |
| `oracle_mask.seed123.500k` / `step_500000` | 123 / 500000 / 500000 | periodic | 5322057 | 0/256 (0.0000%) | `f27eb283f952a10f80473458d01eb341f051d0358733fe2a7739205b62cade76` |
| `oracle_mask.seed123.500k` / `final` | 123 / 500000 / 500000 | final | 5322057 | 0/256 (0.0000%) | `cd4d5d9423a986804acecbe7a7b48ae440db6ada5598d931e82bb680c656f146` |
| `oracle_mask.seed42.2m` / `step_300000` | 42 / 2000000 / 300000 | periodic | 5322057 | 0/256 (0.0000%) | `8fb9e7e69a2fc9cbe7b2c38a8e5babff72c4c9387ca5fb6757a5fd963cf6e337` |
| `oracle_mask.seed42.2m` / `step_500000` | 42 / 2000000 / 500000 | periodic | 5322057 | 0/256 (0.0000%) | `f0de04cab322f7670512f8004505c36c5fe98839f79438b8002d1b6131d41ed9` |
| `oracle_mask.seed42.2m` / `step_1000000` | 42 / 2000000 / 1000000 | periodic | 5322057 | 0/256 (0.0000%) | `742a963a7e24a14f61c5159bad256c414c283d87af01f844517a792e56e48f8c` |
| `oracle_mask.seed42.2m` / `step_2000000` | 42 / 2000000 / 2000000 | periodic | 5322057 | 0/256 (0.0000%) | `7933ccfc4dbfee1064b3d3c13e34ed4eeb4f544919a682642f43fcd6eb9e4e34` |
| `oracle_mask.seed42.2m` / `final` | 42 / 2000000 / 2000000 | final | 5322057 | 0/256 (0.0000%) | `ada68fbe3e5c6e68775212f906e60ab031aa0d27948c0d833643bffafe457ab2` |

Baseline48/256·penalty22/256은고정상태의greedy첫선택으로trainingU-rate나gameplaytrap비율과다르다. Oracle7개0/256은externalphysicalmask의효과이며Decision의실행가능성학습성공을뜻하지않는다. 이번추론은model별2modes의6decision또는placement상한smoke다. 전체성능재평가가아니다.

각파일의Asset명·실제원본의Registry상대경로·정확한config·객체/tensorinventory·과거및이번평가·공개상태는 [MODEL_RELEASE_MANIFEST.json](../MODEL_RELEASE_MANIFEST.json)과 [MODEL_CARDS.md](MODEL_CARDS.md)에있다. 원PC절대경로는비공개audit에만기록했다. 과거gameplay미확인artifact는그대로미확인으로남겼다. 같은step의periodic/final은tensor가같아도hash가달라분리한다.

공개상태는 **APPROVED_FOR_PUBLIC_RELEASE** 이고model license는MIT이다. 기술통과가권리grant가아니다. [배포방법](MODEL_RELEASE_GUIDE.md)·[원본/변환비교](CHECKPOINT_FORMAT_OPTIONS.md)·[보안](CHECKPOINT_SECURITY.md)
