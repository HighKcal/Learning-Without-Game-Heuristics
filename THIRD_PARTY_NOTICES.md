# 제3자 출처와 고지

공개 후보에 포함한 third-party 코드는 Native Python19개와 그 원 고지다. wheel·site-packages·EXE/DLL은 포함하지 않는다.

- `vendor/native_tetris_031/LICENSE`: 원문 그대로, Copyright(c)2024 Maximilian Weichart, MIT. 일부 game logic의 Copyright(c)2020 Viet Nguyen attribution 포함.
- `vendor/native_tetris_031/third_party/uvipen/LICENSE`: [상류 MIT 원문](https://raw.githubusercontent.com/uvipen/Tetris-deep-Q-learning-pytorch/master/LICENSE), Copyright(c)2020 Viet Nguyen. 수집 URL/SHA256은 같은 폴더의 provenance에 있다.
- 수정 사실: `envs/tetris.py` render dtype `.astype(np.integer)`→`.astype(int)` 한 줄. 원 Physics는 유지. 공식 wheel URL·hash, 변경 patch,19개 source hash는 vendor manifest/patch에 보존한다.
- `tests/fixtures/oracle_pre_perf_v2`는 이 프로젝트의 이전 소스를 추출한 독립 reference fixture다. commit·파일hash는 PROVENANCE.json에 보존한다. 별도 외부 라이브러리의 code를 검증됐다고 오인하지 않는다.

설치 의존성의 실제 metadata를 CPU 신규 설치본30개와 기존 CUDA49개에서 조사했다. CPU162 / CUDA180 license 관련 파일의 경로·size·SHA256을 기록했다(중복되는 패키지/파일 포함). [dependency_licenses.json](dependency_licenses.json)의 SPDX expression·License/Classifiers·project URL·고지 inventory를 참고한다. 아래 license는 설치본 metadata의 식별값이며 상류 전체 바이너리 재배포 권리의 최종 판정이 아니다.

| 환경 | 패키지 | 버전 | metadata license |
| --- | --- | --- | --- |
| CPU | absl-py | 2.5.1 | Apache-2.0 |
| CPU | chex | 0.1.90 | Apache 2.0 |
| CPU | cloudpickle | 3.1.2 | BSD-3-Clause |
| CPU | Farama-Notifications | 0.0.6 | License :: OSI Approved :: MIT License |
| CPU | filelock | 3.32.3 | MIT |
| CPU | fsspec | 2026.7.0 | BSD-3-Clause |
| CPU | gymnasium | 1.3.0 | MIT License |
| CPU | iniconfig | 2.3.1 | MIT |
| CPU | jax | 0.5.3 | Apache-2.0 |
| CPU | jaxlib | 0.5.3 | Apache-2.0 |
| CPU | Jinja2 | 3.1.6 | License :: OSI Approved :: BSD License |
| CPU | MarkupSafe | 3.0.3 | BSD-3-Clause |
| CPU | ml_dtypes | 0.6.0 | Apache-2.0 |
| CPU | mpmath | 1.3.0 | BSD |
| CPU | networkx | 3.6.1 | BSD-3-Clause |
| CPU | numpy | 2.5.3 | BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0 |
| CPU | opencv-python | 4.14.0.94 | Apache 2.0 |
| CPU | opt_einsum | 3.4.0 | MIT |
| CPU | packaging | 26.3 | Apache-2.0 OR BSD-2-Clause |
| CPU | pluggy | 1.6.0 | MIT |
| CPU | Pygments | 2.21.0 | BSD-2-Clause |
| CPU | pytest | 9.1.1 | MIT |
| CPU | PyYAML | 6.0.3 | MIT |
| CPU | scipy | 1.18.1 | Copyright (c) 2001-2002 Enthought, Inc. 2003, SciPy Developers. |
| CPU | setuptools | 78.1.0 | License :: OSI Approved :: MIT License |
| CPU | sympy | 1.14.0 | BSD |
| CPU | tetris-gymnasium | 0.3.1 | metadata 미확인 |
| CPU | toolz | 1.2.0 | BSD-3-Clause |
| CPU | torch | 2.14.1+cpu | Apache-2.0 AND Apache-2.0 WITH LLVM-exception AND BSD-2-Clause AND BSD-3-Clause AND BSL-1.0 AND MIT |
| CPU | typing_extensions | 4.16.0 | PSF-2.0 |
| CUDA | absl-py | 2.5.1 | Apache-2.0 |
| CUDA | chex | 0.1.90 | Apache 2.0 |
| CUDA | cloudpickle | 3.1.2 | BSD-3-Clause |
| CUDA | cuda-bindings | 13.3.1 | LicenseRef-NVIDIA-SOFTWARE-LICENSE |
| CUDA | cuda-pathfinder | 1.6.0 | Apache-2.0 |
| CUDA | cuda-toolkit | 13.2.2 | metadata 미확인 |
| CUDA | Farama-Notifications | 0.0.6 | License :: OSI Approved :: MIT License |
| CUDA | filelock | 3.32.3 | MIT |
| CUDA | fsspec | 2026.7.0 | BSD-3-Clause |
| CUDA | gymnasium | 1.3.0 | MIT License |
| CUDA | iniconfig | 2.3.1 | MIT |
| CUDA | jax | 0.5.3 | Apache-2.0 |
| CUDA | jaxlib | 0.5.3 | Apache-2.0 |
| CUDA | Jinja2 | 3.1.6 | License :: OSI Approved :: BSD License |
| CUDA | MarkupSafe | 3.0.3 | BSD-3-Clause |
| CUDA | ml_dtypes | 0.6.0 | Apache-2.0 |
| CUDA | mpmath | 1.3.0 | BSD |
| CUDA | networkx | 3.6.1 | BSD-3-Clause |
| CUDA | numpy | 2.5.3 | BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0 |
| CUDA | nvidia-cublas | 13.4.1.3 | LicenseRef-NVIDIA-Proprietary |
| CUDA | nvidia-cuda-cupti | 13.2.86 | LicenseRef-NVIDIA-Proprietary |
| CUDA | nvidia-cuda-nvrtc | 13.2.86 | LicenseRef-NVIDIA-Proprietary |
| CUDA | nvidia-cuda-runtime | 13.2.86 | LicenseRef-NVIDIA-Proprietary |
| CUDA | nvidia-cudnn-cu13 | 9.24.0.43 | LicenseRef-NVIDIA-Proprietary |
| CUDA | nvidia-cufft | 12.2.0.57 | LicenseRef-NVIDIA-Proprietary |
| CUDA | nvidia-cufile | 1.17.1.22 | LicenseRef-NVIDIA-Proprietary |
| CUDA | nvidia-curand | 10.4.2.66 | LicenseRef-NVIDIA-Proprietary |
| CUDA | nvidia-cusolver | 12.2.0.11 | LicenseRef-NVIDIA-Proprietary |
| CUDA | nvidia-cusparse | 12.7.10.12 | LicenseRef-NVIDIA-Proprietary |
| CUDA | nvidia-cusparselt-cu13 | 0.8.1 | NVIDIA Proprietary Software |
| CUDA | nvidia-nccl-cu13 | 2.30.7 | LicenseRef-NVIDIA-Proprietary |
| CUDA | nvidia-nvjitlink | 13.3.33 | LicenseRef-NVIDIA-Proprietary |
| CUDA | nvidia-nvshmem-cu13 | 3.4.5 | LicenseRef-NVIDIA-Proprietary |
| CUDA | nvidia-nvtx | 13.2.86 | LicenseRef-NVIDIA-Proprietary |
| CUDA | opencv-python | 4.14.0.94 | Apache 2.0 |
| CUDA | opt_einsum | 3.4.0 | MIT |
| CUDA | packaging | 26.3 | Apache-2.0 OR BSD-2-Clause |
| CUDA | pluggy | 1.6.0 | MIT |
| CUDA | Pygments | 2.21.0 | BSD-2-Clause |
| CUDA | pytest | 9.1.1 | MIT |
| CUDA | PyYAML | 6.0.3 | MIT |
| CUDA | scipy | 1.18.1 | Copyright (c) 2001-2002 Enthought, Inc. 2003, SciPy Developers. |
| CUDA | setuptools | 78.1.0 | License :: OSI Approved :: MIT License |
| CUDA | sympy | 1.14.0 | BSD |
| CUDA | tetris-gymnasium | 0.3.1 | metadata 미확인 |
| CUDA | toolz | 1.2.0 | BSD-3-Clause |
| CUDA | torch | 2.14.1+cu132 | Apache-2.0 AND Apache-2.0 WITH LLVM-exception AND BSD-2-Clause AND BSD-3-Clause AND BSL-1.0 AND MIT |
| CUDA | triton | 3.8.0 | MIT |
| CUDA | typing_extensions | 4.16.0 | PSF-2.0 |

프로젝트의 MIT/Apache 후보는 위 dependency license를 덮어쓰지 않는다. GPL/LGPL·NVIDIA proprietary 등 metadata가 있거나 license expression이 누락된 패키지의 실제 terms는 별도로 확인해야 한다. Source-only lock URL 제공과 EXE·wheel bundling은 구분한다. 특히 GUI의 Qt/PySide/shiboken, MSVC entitlement, Intel OpenMP, OpenSSL 및 실제 포함된 NumPy/Torch third parties의 조건은 GUI gate에 남아 있다. 이 inventory를 최종 binary SBOM/대응 source offer로 취급하지 않는다.
