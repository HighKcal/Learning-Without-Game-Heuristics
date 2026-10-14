# Checkpoint 신뢰 출처와 로딩 보안

공식HighKcal/Learning-Without-Game-Heuristics의모델Release9개와동일한manifest·SHA256목록을사용한다. Source공개후모델Release를업로드·검증중이며게시완료후실제링크를제공한다. 파일 이름·크기가 같다는 이유만으로 신뢰하지 않는다. SHA256 일치는 파일 동일성 검사이며 manifest까지 공격자가 바꿀 수 있으면 발행자 인증을 제공하지 않는다. 최종 public commit/tag·권리자 공개 목록에서 hash를 교차 확인한다.

이 모델9개는 Registry와 기존 원본의 hash를 확인한 후 ZIP inventory·pickle opcode/GLOBAL/문자열을 정적으로 검사했다. `get_unsafe_globals_in_checkpoint`의 미허용 GLOBAL은0이었다. 다음으로 신뢰된 원본의 bytes에만 `torch.load(..., weights_only=True, map_location='cpu', mmap=True)` 제한 로딩을 적용하고 plain dict/list/tuple/scalar/Tensor 구조·shape·finite·문자열을 검사했다. 일반 `pickle.load`, `weights_only=False`, custom `safe_globals` 추가는 사용하지 않았다. Registry와 기존 평가/추론 loader도 `weights_only=True`를 유지한다.

이 검사는 악성코드·개인정보의 완전한 부재 증명이 아니다. 정적 GLOBAL 검사는 동적으로 생성되는 모든 type을 찾지 못할 수 있고 `weights_only`도 자원 고갈과 모든 downstream memory 위험을 막지 않는다. 알 수 없는 출처·hash 불일치 파일은 로딩하지 않는다. 사용자 모델이더라도 공유받는 파일은 출처를 먼저 확인한다. 실제 감사 범위는 `MODEL_RELEASE_MANIFEST.json`에 있다.

다운로드 폴더를 준비한 후 코드 root에서 실행한다.

```bash
python -B scripts/install_model_assets.py --asset-dir ../downloaded-models --artifact-root ../model-artifacts
python -B scripts/validate_native_profile.py
python -B scripts/infer_agent.py --artifact-root ../model-artifacts --experiment oracle_mask.seed42.2m --role final --mode main_oracle_play_witness --device cpu --seed 100000 --limit 6 --output outputs/release_inference_new
```

`model-artifacts`는 없는 새 디렉터리여야 한다. 설치기는9개 Asset·manifest·Registry·role/path/size/hash를 모두 검사한 후 byte복사하고 재검증한다. Missing/변조/기존root/Registry불일치는 오류로 거부한다. 이 helper는 모델 provenance의 법적 보증이나 게시 승인을 대신하지 않는다. OS 접근권한·실행 격리는 host 환경의 책임이며 venv 자체를 보안 sandbox라고 표현하지 않는다.

공식 근거: [PyTorch serialization](https://docs.pytorch.org/docs/2.14/notes/serialization.html), [PyTorch security policy](https://github.com/pytorch/pytorch/security/policy).
