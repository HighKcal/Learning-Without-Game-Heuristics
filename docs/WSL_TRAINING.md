# WSL2에서 설치·학습·평가·추론하기

Ubuntu24.04 x86_64 / CPython **3.12.10** 기준이다. 소스는 Linux home의 새 디렉터리, venv·출력도 새 경로를 사용한다. 검증 source는 `codex/research-staging-integration`의 `3801e53c`에서 선별한 공개 후보다. Private Git bundle·이력·Windows venv를 설치 입력으로 사용하지 않는다. 공개 source는 GitHub main에서 Clone한다. 모델은 별도 Release Asset이며 전체 private 데이터가 필요하지 않다.

## 1. Python과 소스 준비

Ubuntu24.04 WSL2와 정확한 Python3.12.10은 사전 조건이다. 일반 Ubuntu `python3.12`는 patch version이 다를 수 있으므로 검사한다. 소스 디렉터리에서 시작한다.

```bash
git clone https://github.com/HighKcal/Learning-Without-Game-Heuristics.git ~/Learning-Without-Game-Heuristics
cd ~/Learning-Without-Game-Heuristics
PYTHON=python3.12
"$PYTHON" -c 'import sys; assert sys.version_info[:3] == (3,12,10), sys.version'
"$PYTHON" -m venv .venv
source .venv/bin/activate
python -m pip --version
```

사용자 pyenv가 이미 설치됐다면 정확한 interpreter를 선택하는 대안이다. 앞의 venv 생성과 둘 중 하나만 수행한다.

```bash
pyenv install -s 3.12.10
PYTHON="$(pyenv root)/versions/3.12.10/bin/python"
"$PYTHON" -c 'import sys; assert sys.version_info[:3] == (3,12,10)'
"$PYTHON" -m venv .venv
source .venv/bin/activate
```

Python source build의 compiler/SSL/zlib/ffi/readline/sqlite/lzma 준비는 [pyenv 설치](https://github.com/pyenv/pyenv#installation)·[빌드 환경](https://github.com/pyenv/pyenv/wiki#suggested-build-environment)을 따른다. Phase4에서는 기존 interpreter로 새 venv를 만들었고 Python 자체의 새 source build·새 WSL 설치를 검증하지 않았다.

## 2. CPU 또는 CUDA 의존성

CPU30 wheels/CUDA49 wheels의 URL·SHA256·version을 lock에 고정했다. CPython3.12/Linux x86_64 전용이다. pip25.0.1에서 검증했다. 하나의 venv에서 한 종류만 선택한다.

```bash
# CPU
python -m pip install --require-hashes -r requirements/linux-cpu.lock
# CUDA를 선택했다면 위 대신 별도 새 venv에서 실행:
# python -m pip install --require-hashes -r requirements/linux-cu132.lock
python -m pip check
```

핵심 의존성은 Torch2.14.1+cpu 또는2.14.1+cu132, NumPy2.5.3, Gymnasium1.3.0, OpenCV4.14.0.94, PyYAML6.0.3, pytest9.1.1이다. Torch는 공식 `download.pytorch.org`, NVIDIA 일부는 `pypi.nvidia.com`, 나머지는 PyPI 고정 wheel URL이다. `requirements.txt`만으로 Torch·Native source hash 재현을 보장하지 않는다.

CUDA는 WSL에서 GPU가 노출되는지 `/usr/lib/wsl/lib/nvidia-smi`로 확인한다. WSL은 Windows NVIDIA driver를 사용하며 Linux display driver를 별도로 설치하지 않는다. 프로젝트 venv 설치와 driver·OS 라이브러리 설치를 구분한다. [NVIDIA WSL 안내](https://docs.nvidia.com/cuda/wsl-user-guide/index.html)·[PyTorch 설치](https://pytorch.org/get-started/locally/). Phase4 새 검증은 CPU이며 CUDA의32-decision 기능 근거는 Phase3 보고서다. 다른 GPU/CUDA 환경을 새 인증한 것이 아니다.

OpenCV import에 필요한 OS 라이브러리는 wheel과 별도다. 현 Ubuntu 환경에서는 추가 apt 없이 통과했지만 headless/vanilla wheel로 조용히 교체하지 않는다.

## 3. 수정 Native 설치·19개 SHA256 확인

```bash
python -B scripts/install_native.py --report outputs/native_install_new.json
python -B scripts/validate_native_profile.py
python -B scripts/verify_wsl_runtime.py --device cpu --output outputs/runtime_cpu_new.json
# CUDA venv이면 마지막 명령의 --device cuda
```

Native package version0.3.1만으로 충분하지 않다. Vendor19 Python source의 raw bytes/hash가 profile과 일치해야 한다. Installer는 새 venv 내부의 확인된 upstream 또는 정확한 수정본에만 적용한다. 유일한 수정은 render dtype `.astype(np.integer)`→`.astype(int)`다. Physics를 바꾸지 않았고 CRLF/LF 원 바이트가 hash 계약에 포함된다.

Train/smoke/evaluate CLI는 Native preflight를 실행한다. `play_agent.py` 직접 checkpoint CLI는 Registry façade가 아니므로 **위 Native 검증을 먼저 실행**한다. Registry `infer_agent.py`는 profile/hash/지원 CPU 계약을 자체 확인한다.

## 4. 전체 테스트

```bash
unset TETRIS_ARTIFACT_ROOT PYTHONPATH
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -B -m pytest -q -p no:cacheprovider --basetemp="$(mktemp -d)/pytest" --ignore=tests/test_u_competing_actions_diagnostic.py --ignore=tests/test_u_feedback_timing_diagnostic.py
```

제외된 두 모듈34 historical diagnostic cases는 비공개 trained checkpoints를 복원해 optimizer update하는 과거 진단이다. 공개 snapshot의 모델 없이 실행 가능한 suite와 구분한다. 나머지 tests는 합성·임시 모델을 사용한다. Oracle pre-perf-v2 reference fixture는 검증용 선별 Python 소스이며 과거 worktree/Git history 사본이 아니다.

## 5. 새 Geometric / Oracle Smoke 학습

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -B scripts/smoke_train.py --device cpu --reachability-mask geometric --seed 123 --output-dir outputs/geometric_smoke_new
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -B scripts/smoke_train.py --device cpu --reachability-mask oracle --seed 123 --output-dir outputs/oracle_smoke_new
# CUDA 선택시 --device cuda, 충돌 없는 새 output-dir 지정
```

Canonical이나 기존 model input이 필요 없다. Smoke32 decisions/warmup4/batch4/replay64/sync8updates/cap16에서 기대값은32 replay transitions/29 optimizer updates/3 target sync다. 실제 weight 변화·finite loss·새 periodic16/32/final checkpoint 저장·reload가 `smoke_evidence.json`에 기록된다. Snapshot은 exact resume를 지원하지 않는다. 이32 decisions를 장기 학습 성능 재현으로 표시하지 않는다.

## 6. 일반 새 학습 (실행 예시, Phase4에서 미실행)

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -B scripts/train_rl.py --algorithm double --reward minimal --decision-budget 6000 --seed 123 --device cpu --reachability-mask geometric --unreachable-penalty -0.05 --epsilon-decay-decisions 38000 --log-every 100 --checkpoint-every 1000 --max-decisions-per-episode 10000 --output-dir outputs/geometric_seed123_new
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -B scripts/train_rl.py --algorithm double --reward minimal --decision-budget 6000 --seed 123 --device cpu --reachability-mask oracle --unreachable-penalty -0.05 --epsilon-decay-decisions 38000 --log-every 100 --checkpoint-every 1000 --max-decisions-per-episode 10000 --output-dir outputs/oracle_seed123_new
```

Production warmup5000/batch128/replay100000/sync5000을 유지한다. 6000 budget은 실행 예시이지 수렴/성능 검증이 아니다. 새 run은 Registry에 자동 등록되거나 지원 승격되지 않는다. 오래된 run 폴더를 output으로 지정하지 않는다.

## 7. 평가와 새 Smoke 모델 추론

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -B scripts/evaluate_policy.py --dataset evaluation/fixed_eval_final_v1 --checkpoint outputs/oracle_smoke_new/checkpoint_final.pt --device cpu --reachability-mask oracle --output outputs/oracle_eval_new.json
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -B scripts/play_agent.py --checkpoint outputs/oracle_smoke_new/checkpoint_final.pt --mode research --reachability-mask oracle --episodes 1 --max-decisions 6 --device cpu --output outputs/oracle_inference_new
```

Geometric 모델은 checkpoint와 mask를 함께 geometric으로 바꾼다. `--mode play --reachability-mask geometric`은 Planner fallback이므로 research top1 결과와 다르다. Fixed256 상태 시험과 gameplay6 decisions도 별도 검사다.

## 8. 별도 배포 모델 추론

아래는9개 모델 Release Asset을 내려받아 배치한 뒤의 절차다. Registry-relative hierarchy를 유지한 사용자 artifact directory를 사용한다. 기본 source checkout에 private 원본을 자동 찾지 않는다.

```bash
export TETRIS_ARTIFACT_ROOT="$(realpath ../approved-models)"
python -B scripts/infer_agent.py --experiment oracle_mask.seed42.2m --role final --mode main_oracle_play_witness --device cpu --seed 100000 --limit 6 --output outputs/approved_oracle_play_new
```

Missing/hash mismatch/pending/unsupported는 오류로 종료한다. 지원4 experiments/9 checkpoints이며 final/periodic을 합치지 않는다. Registry validated device는 **CPU**다. CUDA forward·fresh CUDA 학습 가능성이 Registry CUDA support 승격을 의미하지 않는다.

검증 범위는 [WSL 보고서](WSL_REPRODUCIBILITY_REPORT.md), 새 Phase4 결과는 [준비 보고서](PHASE4_GITHUB_PUBLIC_PREPARATION_REPORT.md)에 있다. [모델 계획](models.md)·[license gate](licensing.md)

## 9. Phase4.5의9개모델 배치

Phase4.5의모델은코드와별도ReleaseAsset후보다. 승인된9개를받으면`install_model_assets.py`로size/hash/Registry를검사해새root에배치한다. [모델배포가이드](MODEL_RELEASE_GUIDE.md)와[보안주의](CHECKPOINT_SECURITY.md)를따른다. 위1~7절의모델없는신규학습·평가경로는유지한다.
