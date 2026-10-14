# 모델 Release Asset 구성과 사용 방법

Source repository와research-v1.0.0게시를사용자가승인했다.9개원본모델은별도ReleaseAsset으로배포한다. 실제공개Release는 [research-v1.0.0](https://github.com/HighKcal/Learning-Without-Game-Heuristics/releases/tag/research-v1.0.0)다.9개모델과4개companion파일이게시됐다. 기존 private Git 이력이나 source Git에 모델 binary를 추가하지 않는다.

## 1. 구성

원 bytes의9개 모델 Asset은 이름에 experiment·seed·budget·periodic step/final을 구분한다. [SHA256 목록](MODEL_ASSET_SHA256SUMS.txt)·[manifest](../MODEL_RELEASE_MANIFEST.json)에 원filename, size, hash, Registry 상대 설치 경로, 학습/평가 조건을 기록한다. 합계47,896,785bytes다.

Companion Assets는 `SHA256SUMS`, `MODEL_RELEASE_MANIFEST.json`, 한국어 `MODEL_CARDS.md`, 별도 MIT `MODEL_LICENSE.txt`다. 저작권 표시는Copyright (c) 2026 HighKcal이다.9개모델에MIT를적용하고원Native고지는보존했다. GUI/EXE assets는 없다.

## 2. 다운로드와 검증

승인된 공식 Release에서9개 `.pt`를 다운로드 폴더에 놓는다. 공식 source commit의manifest와모델hash를교차확인한다. SHA256은발행자의정체를단독보증하지않는다. Source/model코드는독립배포하며불명확한mirror나개인메시지파일을그대로load하지않는다.

Asset폴더에서선택한SHA256목록을수동검사할수있다.

```bash
sha256sum -c SHA256SUMS
```

Sourceroot에서는모든9개를기존Registry계층에자동배치한다. Source의manifest를사용하므로내려받은manifest가바뀌어도Registry와일치해야한다.

```bash
python -B scripts/install_model_assets.py --asset-dir ../downloaded-models --artifact-root ../model-artifacts
python -B scripts/validate_native_profile.py
```

`model-artifacts`는없는새디렉터리다. 기존root에는덮어쓰지않는다. `--manifest`를지정한다면그파일의발행자/상태도확인한다. helper는missing·파일변조·목록/Registry불일치·symlink/junction경로를거부하며native/source학습코드를바꾸지않는다.

## 3. 평가·추론

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -B scripts/infer_agent.py --artifact-root ../model-artifacts --experiment oracle_mask.seed123.500k --role periodic --step 500000 --mode research_oracle_greedy --device cpu --seed 100000 --limit 6 --output outputs/oracle_periodic_inference_new
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -B scripts/infer_agent.py --artifact-root ../model-artifacts --experiment oracle_mask.seed42.2m --role final --mode main_oracle_play_witness --device cpu --seed 100000 --limit 6 --output outputs/oracle_final_inference_new
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -B scripts/infer_agent.py --artifact-root ../model-artifacts --experiment geometric.baseline.seed123.100k --role final --mode research_geometric_top1 --device cpu --seed 100000 --limit 6 --output outputs/geometric_research_new
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -B scripts/infer_agent.py --artifact-root ../model-artifacts --experiment geometric.penalty020.seed42.500k --role final --mode main_geometric_play_fallback --device cpu --seed 100000 --limit 6 --output outputs/geometric_play_new
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -B scripts/evaluate_policy.py --dataset evaluation/fixed_eval_final_v1 --checkpoint ../model-artifacts/runs/oracle_mask/seed_42/2m/checkpoint_final.pt --device cpu --reachability-mask oracle --output outputs/oracle_release_eval_new.json
```

Geometric checkpoint의평가는`--reachability-mask geometric`을사용한다. Researchtop1U·Playfallback과Oraclemask/witness를혼합하지않는다. Fixed256-state결과는gameplay성능이아니다. 모델license가확정된후에도strategy/feasibility학습이해결됐다고과장하지않는다.

## 4. 실제 로컬 모의 배포

9개원본→고유Asset이름의별도폴더→WSL다운로드모사폴더→helper의새Registryroot순으로byte복사했다. 각단계SHA256/size가일치했다. 후보code로9개×256상태평가와9개×2지원모드추론,hash불변·optimizer/replayupdate0을확인했다. 변조파일·기존root·manifest/Registry불일치거부검사도통과했다. Native19hash계약은유지했다.

Rawlogs와내부실제경로는비공개감사에있고공개결과는 [PHASE4_5_RELEASE_VALIDATION.json](results/PHASE4_5_RELEASE_VALIDATION.json)이다. 같은WSLhost와Phase4의lock설치CPUvenv를재사용했으며새OS/장기훈련/모든seed평가를인증하지않는다. [보안](CHECKPOINT_SECURITY.md)·[모델카드](MODEL_CARDS.md)·[승인목록](RELEASE_APPROVALS.md)


## 실제 Release에서 내려받기

아래명령은현재게시된13개Asset URL 중9개모델과SHA256SUMS만다운로드한다. 기존폴더를덮어쓰지않도록새디렉터리를사용한다. Release의MODEL_LICENSE.txt·manifest·MODEL_CARDS.md도함께읽는다.

```bash
mkdir ../downloaded-models
curl --fail --location 'https://github.com/HighKcal/Learning-Without-Game-Heuristics/releases/download/research-v1.0.0/Temeong-geometric-baseline-seed123-100k-final.pt' --output '../downloaded-models/Temeong-geometric-baseline-seed123-100k-final.pt'
curl --fail --location 'https://github.com/HighKcal/Learning-Without-Game-Heuristics/releases/download/research-v1.0.0/Temeong-geometric-penalty020-seed42-500k-final.pt' --output '../downloaded-models/Temeong-geometric-penalty020-seed42-500k-final.pt'
curl --fail --location 'https://github.com/HighKcal/Learning-Without-Game-Heuristics/releases/download/research-v1.0.0/Temeong-oracle_mask-seed123-500k-final.pt' --output '../downloaded-models/Temeong-oracle_mask-seed123-500k-final.pt'
curl --fail --location 'https://github.com/HighKcal/Learning-Without-Game-Heuristics/releases/download/research-v1.0.0/Temeong-oracle_mask-seed123-500k-periodic-500000.pt' --output '../downloaded-models/Temeong-oracle_mask-seed123-500k-periodic-500000.pt'
curl --fail --location 'https://github.com/HighKcal/Learning-Without-Game-Heuristics/releases/download/research-v1.0.0/Temeong-oracle_mask-seed42-2m-final.pt' --output '../downloaded-models/Temeong-oracle_mask-seed42-2m-final.pt'
curl --fail --location 'https://github.com/HighKcal/Learning-Without-Game-Heuristics/releases/download/research-v1.0.0/Temeong-oracle_mask-seed42-2m-periodic-1000000.pt' --output '../downloaded-models/Temeong-oracle_mask-seed42-2m-periodic-1000000.pt'
curl --fail --location 'https://github.com/HighKcal/Learning-Without-Game-Heuristics/releases/download/research-v1.0.0/Temeong-oracle_mask-seed42-2m-periodic-2000000.pt' --output '../downloaded-models/Temeong-oracle_mask-seed42-2m-periodic-2000000.pt'
curl --fail --location 'https://github.com/HighKcal/Learning-Without-Game-Heuristics/releases/download/research-v1.0.0/Temeong-oracle_mask-seed42-2m-periodic-300000.pt' --output '../downloaded-models/Temeong-oracle_mask-seed42-2m-periodic-300000.pt'
curl --fail --location 'https://github.com/HighKcal/Learning-Without-Game-Heuristics/releases/download/research-v1.0.0/Temeong-oracle_mask-seed42-2m-periodic-500000.pt' --output '../downloaded-models/Temeong-oracle_mask-seed42-2m-periodic-500000.pt'
curl --fail --location 'https://github.com/HighKcal/Learning-Without-Game-Heuristics/releases/download/research-v1.0.0/SHA256SUMS' --output '../downloaded-models/SHA256SUMS'
(cd ../downloaded-models && sha256sum -c SHA256SUMS)
python -B scripts/install_model_assets.py --asset-dir ../downloaded-models --artifact-root ../model-artifacts
```
