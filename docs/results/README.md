# 선별 공개 집계

두 CSV는 과거 Oracle gameplay 평가 결과에서 개발 PC `path` 열만 제거한 사본이다. 나머지 열 이름·수치 문자열·row 순서는 원본과 같다. 출처와 source/public SHA256은 [provenance.json](provenance.json)에 있다. 전체 raw trace·Canonical은 포함하지 않는다.

- [seed123500K](oracle_seed123_500k_gameplay.csv): periodic50K grid10개×50 seeds100000~100049, ε=0, CPU1-thread, episode cap10000. periodic500K와 final의 평가를 자동 합치지 않는다.
- [seed422M](oracle_seed42_2m_gameplay.csv): 8 checkpoints×공통100 seeds100000~100099 selection, final2M별도50 seeds100200~100249 holdout, seed123periodic500K reference50. 각 phase를 분리한다.

Oracle의 selected U0은 외부 mask 계약이다. 지표는 `lines_per_1000_placements`이며 decisions 분모가 아니다. Seed 간 비교·local winner 선택·holdout과 training 효율은 서로 다른 평가 조건이다. 공통100-seed final2M144.4538과 별도50-seed132.7783을 혼동하지 않는다. 정확한 모든 과거 상태를 공개한 것처럼 표현하지 않는다.
