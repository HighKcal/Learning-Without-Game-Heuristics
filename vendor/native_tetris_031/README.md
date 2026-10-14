# 검증된 수정 Native 0.3.1

Native Python 소스19개와 raw SHA256, 공식 upstream wheel URL/hash를 보존한다. Upstream 설치 후 새 venv에서 `python -B scripts/install_native.py`를 실행한다. 설치기는 알 수 없는 source inventory·hash·venv 밖 수정을 거부한다.

수정은 `envs/tetris.py` render dtype의 `.astype(np.integer)`→`.astype(int)` 한 줄이다. 물리·보상 변경이 아니다. 원 줄바꿈을 보존해야 raw hash가 일치한다. 상류 MIT LICENSE 원문은 수정하지 않았다. Copyright2024 Maximilian Weichart, 일부 game logic의 Copyright2020 Viet Nguyen attribution을 유지한다.

Viet Nguyen의 대응 MIT 원문도 [third_party/uvipen/LICENSE](third_party/uvipen/LICENSE)에 보존한다. 원 출처와 수집 SHA256은 [third_party/uvipen/provenance.json](third_party/uvipen/provenance.json)에 있다. Windows 바이너리·site-packages 사본은 포함하지 않는다. 프로젝트 수정분 권리자 승인은 별도 남는다.
