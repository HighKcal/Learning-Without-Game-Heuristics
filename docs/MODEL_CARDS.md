# 모델별 학습·평가·사용 조건

모든카드는원artifact를식별한다. Public원본경로는Registry-relative이고실제PC절대경로는비공개감사표에보존한다. License는MIT,권리자표시는Copyright (c) 2026 HighKcal다. 상업적사용/수정/재배포조건은최종선택한modellicense가정하며현재permissiongrant를자동추정하지않는다.

공통조건: DDQN/minimal, γ.99, Adam1e−4, batch128, replay100000, warmup5000, targetsync5000updates, clip10, epsilon1→.05/decay38000, episodecap10000. 235→256→256→800MLP. CPU 추론 profile과 Native19 raw hash가 필요하다. CUDA 학습과 CPU 검증 지원을 구분한다.

## 1. geometric.baseline.seed123.100k / final

- Release Asset: `Temeong-geometric-baseline-seed123-100k-final.pt`
- Registry 원filename/배치위치: `runs/geometric/baseline/seed_123/100k/checkpoint_final.pt`
- Bytes / SHA256: 5321161 / `c882fa1bd96f7493cb1bf9ef0149b3b126d524d777ac22396bcf59b8184321b9`
- Seed / 학습budget / 저장decisions / role: 123 / 100000 / 100000 / final
- 정책조건: geometric, U penalty -0.05, DDQN/minimal.
- 학습설정: `{"decision_budget": 100000, "algorithm": "double", "reward_mode": "minimal", "seed": 123, "device": "cuda", "learning_rate": 0.0001, "gamma": 0.99, "replay_capacity": 100000, "batch_size": 128, "learning_starts": 5000, "target_sync_updates": 5000, "gradient_clip": 10.0, "unreachable_penalty": -0.05, "reachability_mask_mode": "geometric", "epsilon_decay_override": 38000}`
- 실제payload: top-level16keys,30finite tensors, online/target/Adam 포함, replay/RNG없음. Stored experiences 100000, gradient updates 95001, target sync 19.
- 저장시각(UTC)/status: 2026-10-04T11:08:07.293399+00:00 / completed (Commit일과구분).
- 의존성: source revision`3801e53c236262b8196880b1fb7392559e91b13a`, profile`native_tetris_031_geometric`, Native0.3.1의19SHA256, Torch2.14.1 CPU runtime. 역사launch SHA는미확인.
- 이번평가: fixed256states, ε=0/geometric, first-choice reachable 81.2500%, U 48/256. Gameplay우수성판정아님.
- 이번추론: `research_geometric_top1, main_geometric_play_fallback`, seed100000/limit6, optimizer/replayupdates0, 원file·weights불변.
- 과거 artifact별 gameplay: 미확인. 다른periodic/final의결과로대체하지않음.
- 초기화/데이터출처: 사용자진술random initialization·개인PC학습·환경생성replay. 현재코드/payload와부합하나각launch초기weights의증명은아님.
- 권리/사용조건: 사용자직접학습진술에근거한공개후보, model MIT 적용/사용자 최종승인. MIT 고지와license를동봉한다. 사용자가권리자표시HighKcal을최종승인했다.
- 한계: 정확resume불가, 장기성능/feasibility학습해결보장없음. OracleU0은mask효과. 원본만배포준비했으며변환없음.

## 2. geometric.penalty020.seed42.500k / final

- Release Asset: `Temeong-geometric-penalty020-seed42-500k-final.pt`
- Registry 원filename/배치위치: `runs/geometric/penalty_020/seed_42/500k/checkpoint_final.pt`
- Bytes / SHA256: 5321225 / `8124cb9a9ebd54461967213c1b40744f399d8f30ca4521470a3f4fc563b4d0ff`
- Seed / 학습budget / 저장decisions / role: 42 / 500000 / 500000 / final
- 정책조건: geometric, U penalty -0.2, DDQN/minimal.
- 학습설정: `{"decision_budget": 500000, "algorithm": "double", "reward_mode": "minimal", "seed": 42, "device": "cuda", "learning_rate": 0.0001, "gamma": 0.99, "replay_capacity": 100000, "batch_size": 128, "learning_starts": 5000, "target_sync_updates": 5000, "gradient_clip": 10.0, "unreachable_penalty": -0.2, "epsilon_decay_override": 38000, "reachability_mask_mode": "geometric"}`
- 실제payload: top-level16keys,30finite tensors, online/target/Adam 포함, replay/RNG없음. Stored experiences 500000, gradient updates 495001, target sync 99.
- 저장시각(UTC)/status: 2026-10-04T22:24:15.831595+00:00 / completed (Commit일과구분).
- 의존성: source revision`3801e53c236262b8196880b1fb7392559e91b13a`, profile`native_tetris_031_geometric`, Native0.3.1의19SHA256, Torch2.14.1 CPU runtime. 역사launch SHA는미확인.
- 이번평가: fixed256states, ε=0/geometric, first-choice reachable 91.4062%, U 22/256. Gameplay우수성판정아님.
- 이번추론: `research_geometric_top1, main_geometric_play_fallback`, seed100000/limit6, optimizer/replayupdates0, 원file·weights불변.
- 과거 artifact별 gameplay: 미확인. 다른periodic/final의결과로대체하지않음.
- 초기화/데이터출처: 사용자진술random initialization·개인PC학습·환경생성replay. 현재코드/payload와부합하나각launch초기weights의증명은아님.
- 권리/사용조건: 사용자직접학습진술에근거한공개후보, model MIT 적용/사용자 최종승인. MIT 고지와license를동봉한다. 사용자가권리자표시HighKcal을최종승인했다.
- 한계: 정확resume불가, 장기성능/feasibility학습해결보장없음. OracleU0은mask효과. 원본만배포준비했으며변환없음.

## 3. oracle_mask.seed123.500k / step_500000

- Release Asset: `Temeong-oracle_mask-seed123-500k-periodic-500000.pt`
- Registry 원filename/배치위치: `runs/oracle_mask/seed_123/500k/checkpoint_000500000.pt`
- Bytes / SHA256: 5322057 / `f27eb283f952a10f80473458d01eb341f051d0358733fe2a7739205b62cade76`
- Seed / 학습budget / 저장decisions / role: 123 / 500000 / 500000 / periodic
- 정책조건: oracle, U penalty -0.05, DDQN/minimal.
- 학습설정: `{"decision_budget": 500000, "algorithm": "double", "reward_mode": "minimal", "seed": 123, "device": "cuda", "learning_rate": 0.0001, "gamma": 0.99, "replay_capacity": 100000, "batch_size": 128, "learning_starts": 5000, "target_sync_updates": 5000, "gradient_clip": 10.0, "unreachable_penalty": -0.05, "epsilon_decay_override": 38000, "reachability_mask_mode": "oracle"}`
- 실제payload: top-level22keys,30finite tensors, online/target/Adam 포함, replay/RNG없음. Stored experiences 500000, gradient updates 495001, target sync 99.
- 저장시각(UTC)/status: 2026-10-06T19:10:53.643246+00:00 / running (Commit일과구분).
- 의존성: source revision`3801e53c236262b8196880b1fb7392559e91b13a`, profile`native_tetris_031_oracle_perf_v2`, Native0.3.1의19SHA256, Torch2.14.1 CPU runtime. 역사launch SHA는미확인.
- 이번평가: fixed256states, ε=0/oracle, first-choice reachable 100.0000%, U 0/256. Gameplay우수성판정아님.
- 이번추론: `research_oracle_greedy, main_oracle_play_witness`, seed100000/limit6, optimizer/replayupdates0, 원file·weights불변.
- 과거gameplay: historical_periodic50K_grid, 50episodes, seeds100000부터, ε0, 54.938272lines/1000placements, 1.78lines/episode, deaths50. Raw protocol/집계는 `docs/results` 사본 참고.
- 초기화/데이터출처: 사용자진술random initialization·개인PC학습·환경생성replay. 현재코드/payload와부합하나각launch초기weights의증명은아님.
- 권리/사용조건: 사용자직접학습진술에근거한공개후보, model MIT 적용/사용자 최종승인. MIT 고지와license를동봉한다. 사용자가권리자표시HighKcal을최종승인했다.
- 한계: 정확resume불가, 장기성능/feasibility학습해결보장없음. OracleU0은mask효과. 원본만배포준비했으며변환없음.

## 4. oracle_mask.seed123.500k / final

- Release Asset: `Temeong-oracle_mask-seed123-500k-final.pt`
- Registry 원filename/배치위치: `runs/oracle_mask/seed_123/500k/checkpoint_final.pt`
- Bytes / SHA256: 5322057 / `cd4d5d9423a986804acecbe7a7b48ae440db6ada5598d931e82bb680c656f146`
- Seed / 학습budget / 저장decisions / role: 123 / 500000 / 500000 / final
- 정책조건: oracle, U penalty -0.05, DDQN/minimal.
- 학습설정: `{"decision_budget": 500000, "algorithm": "double", "reward_mode": "minimal", "seed": 123, "device": "cuda", "learning_rate": 0.0001, "gamma": 0.99, "replay_capacity": 100000, "batch_size": 128, "learning_starts": 5000, "target_sync_updates": 5000, "gradient_clip": 10.0, "unreachable_penalty": -0.05, "epsilon_decay_override": 38000, "reachability_mask_mode": "oracle"}`
- 실제payload: top-level22keys,30finite tensors, online/target/Adam 포함, replay/RNG없음. Stored experiences 500000, gradient updates 495001, target sync 99.
- 저장시각(UTC)/status: 2026-10-06T19:10:53.655246+00:00 / completed (Commit일과구분).
- 의존성: source revision`3801e53c236262b8196880b1fb7392559e91b13a`, profile`native_tetris_031_oracle_perf_v2`, Native0.3.1의19SHA256, Torch2.14.1 CPU runtime. 역사launch SHA는미확인.
- 이번평가: fixed256states, ε=0/oracle, first-choice reachable 100.0000%, U 0/256. Gameplay우수성판정아님.
- 이번추론: `research_oracle_greedy, main_oracle_play_witness`, seed100000/limit6, optimizer/replayupdates0, 원file·weights불변.
- 과거 artifact별 gameplay: 미확인. 다른periodic/final의결과로대체하지않음.
- 초기화/데이터출처: 사용자진술random initialization·개인PC학습·환경생성replay. 현재코드/payload와부합하나각launch초기weights의증명은아님.
- 권리/사용조건: 사용자직접학습진술에근거한공개후보, model MIT 적용/사용자 최종승인. MIT 고지와license를동봉한다. 사용자가권리자표시HighKcal을최종승인했다.
- 한계: 정확resume불가, 장기성능/feasibility학습해결보장없음. OracleU0은mask효과. 원본만배포준비했으며변환없음.

## 5. oracle_mask.seed42.2m / step_300000

- Release Asset: `Temeong-oracle_mask-seed42-2m-periodic-300000.pt`
- Registry 원filename/배치위치: `runs/oracle_mask/seed_42/2m/checkpoint_000300000.pt`
- Bytes / SHA256: 5322057 / `8fb9e7e69a2fc9cbe7b2c38a8e5babff72c4c9387ca5fb6757a5fd963cf6e337`
- Seed / 학습budget / 저장decisions / role: 42 / 2000000 / 300000 / periodic
- 정책조건: oracle, U penalty -0.05, DDQN/minimal.
- 학습설정: `{"decision_budget": 2000000, "algorithm": "double", "reward_mode": "minimal", "seed": 42, "device": "cuda", "learning_rate": 0.0001, "gamma": 0.99, "replay_capacity": 100000, "batch_size": 128, "learning_starts": 5000, "target_sync_updates": 5000, "gradient_clip": 10.0, "unreachable_penalty": -0.05, "epsilon_decay_override": 38000, "reachability_mask_mode": "oracle"}`
- 실제payload: top-level22keys,30finite tensors, online/target/Adam 포함, replay/RNG없음. Stored experiences 300000, gradient updates 295001, target sync 59.
- 저장시각(UTC)/status: 2026-10-06T15:59:29.412304+00:00 / running (Commit일과구분).
- 의존성: source revision`3801e53c236262b8196880b1fb7392559e91b13a`, profile`native_tetris_031_oracle_perf_v2`, Native0.3.1의19SHA256, Torch2.14.1 CPU runtime. 역사launch SHA는미확인.
- 이번평가: fixed256states, ε=0/oracle, first-choice reachable 100.0000%, U 0/256. Gameplay우수성판정아님.
- 이번추론: `research_oracle_greedy, main_oracle_play_witness`, seed100000/limit6, optimizer/replayupdates0, 원file·weights불변.
- 과거 artifact별 gameplay: 미확인. 다른periodic/final의결과로대체하지않음.
- 초기화/데이터출처: 사용자진술random initialization·개인PC학습·환경생성replay. 현재코드/payload와부합하나각launch초기weights의증명은아님.
- 권리/사용조건: 사용자직접학습진술에근거한공개후보, model MIT 적용/사용자 최종승인. MIT 고지와license를동봉한다. 사용자가권리자표시HighKcal을최종승인했다.
- 한계: 정확resume불가, 장기성능/feasibility학습해결보장없음. OracleU0은mask효과. 원본만배포준비했으며변환없음.

## 6. oracle_mask.seed42.2m / step_500000

- Release Asset: `Temeong-oracle_mask-seed42-2m-periodic-500000.pt`
- Registry 원filename/배치위치: `runs/oracle_mask/seed_42/2m/checkpoint_000500000.pt`
- Bytes / SHA256: 5322057 / `f0de04cab322f7670512f8004505c36c5fe98839f79438b8002d1b6131d41ed9`
- Seed / 학습budget / 저장decisions / role: 42 / 2000000 / 500000 / periodic
- 정책조건: oracle, U penalty -0.05, DDQN/minimal.
- 학습설정: `{"decision_budget": 2000000, "algorithm": "double", "reward_mode": "minimal", "seed": 42, "device": "cuda", "learning_rate": 0.0001, "gamma": 0.99, "replay_capacity": 100000, "batch_size": 128, "learning_starts": 5000, "target_sync_updates": 5000, "gradient_clip": 10.0, "unreachable_penalty": -0.05, "epsilon_decay_override": 38000, "reachability_mask_mode": "oracle"}`
- 실제payload: top-level22keys,30finite tensors, online/target/Adam 포함, replay/RNG없음. Stored experiences 500000, gradient updates 495001, target sync 99.
- 저장시각(UTC)/status: 2026-10-06T19:23:37.403380+00:00 / running (Commit일과구분).
- 의존성: source revision`3801e53c236262b8196880b1fb7392559e91b13a`, profile`native_tetris_031_oracle_perf_v2`, Native0.3.1의19SHA256, Torch2.14.1 CPU runtime. 역사launch SHA는미확인.
- 이번평가: fixed256states, ε=0/oracle, first-choice reachable 100.0000%, U 0/256. Gameplay우수성판정아님.
- 이번추론: `research_oracle_greedy, main_oracle_play_witness`, seed100000/limit6, optimizer/replayupdates0, 원file·weights불변.
- 과거gameplay: historical_selection, 100episodes, seeds100000부터, ε0, 93.333333lines/1000placements, 3.5lines/episode, deaths100. Raw protocol/집계는 `docs/results` 사본 참고.
- 초기화/데이터출처: 사용자진술random initialization·개인PC학습·환경생성replay. 현재코드/payload와부합하나각launch초기weights의증명은아님.
- 권리/사용조건: 사용자직접학습진술에근거한공개후보, model MIT 적용/사용자 최종승인. MIT 고지와license를동봉한다. 사용자가권리자표시HighKcal을최종승인했다.
- 한계: 정확resume불가, 장기성능/feasibility학습해결보장없음. OracleU0은mask효과. 원본만배포준비했으며변환없음.

## 7. oracle_mask.seed42.2m / step_1000000

- Release Asset: `Temeong-oracle_mask-seed42-2m-periodic-1000000.pt`
- Registry 원filename/배치위치: `runs/oracle_mask/seed_42/2m/checkpoint_001000000.pt`
- Bytes / SHA256: 5322057 / `742a963a7e24a14f61c5159bad256c414c283d87af01f844517a792e56e48f8c`
- Seed / 학습budget / 저장decisions / role: 42 / 2000000 / 1000000 / periodic
- 정책조건: oracle, U penalty -0.05, DDQN/minimal.
- 학습설정: `{"decision_budget": 2000000, "algorithm": "double", "reward_mode": "minimal", "seed": 42, "device": "cuda", "learning_rate": 0.0001, "gamma": 0.99, "replay_capacity": 100000, "batch_size": 128, "learning_starts": 5000, "target_sync_updates": 5000, "gradient_clip": 10.0, "unreachable_penalty": -0.05, "epsilon_decay_override": 38000, "reachability_mask_mode": "oracle"}`
- 실제payload: top-level22keys,30finite tensors, online/target/Adam 포함, replay/RNG없음. Stored experiences 1000000, gradient updates 995001, target sync 199.
- 저장시각(UTC)/status: 2026-10-07T01:59:38.175815+00:00 / running (Commit일과구분).
- 의존성: source revision`3801e53c236262b8196880b1fb7392559e91b13a`, profile`native_tetris_031_oracle_perf_v2`, Native0.3.1의19SHA256, Torch2.14.1 CPU runtime. 역사launch SHA는미확인.
- 이번평가: fixed256states, ε=0/oracle, first-choice reachable 100.0000%, U 0/256. Gameplay우수성판정아님.
- 이번추론: `research_oracle_greedy, main_oracle_play_witness`, seed100000/limit6, optimizer/replayupdates0, 원file·weights불변.
- 과거gameplay: historical_selection, 100episodes, seeds100000부터, ε0, 133.180778lines/1000placements, 5.82lines/episode, deaths100. Raw protocol/집계는 `docs/results` 사본 참고.
- 초기화/데이터출처: 사용자진술random initialization·개인PC학습·환경생성replay. 현재코드/payload와부합하나각launch초기weights의증명은아님.
- 권리/사용조건: 사용자직접학습진술에근거한공개후보, model MIT 적용/사용자 최종승인. MIT 고지와license를동봉한다. 사용자가권리자표시HighKcal을최종승인했다.
- 한계: 정확resume불가, 장기성능/feasibility학습해결보장없음. OracleU0은mask효과. 원본만배포준비했으며변환없음.

## 8. oracle_mask.seed42.2m / step_2000000

- Release Asset: `Temeong-oracle_mask-seed42-2m-periodic-2000000.pt`
- Registry 원filename/배치위치: `runs/oracle_mask/seed_42/2m/checkpoint_002000000.pt`
- Bytes / SHA256: 5322057 / `7933ccfc4dbfee1064b3d3c13e34ed4eeb4f544919a682642f43fcd6eb9e4e34`
- Seed / 학습budget / 저장decisions / role: 42 / 2000000 / 2000000 / periodic
- 정책조건: oracle, U penalty -0.05, DDQN/minimal.
- 학습설정: `{"decision_budget": 2000000, "algorithm": "double", "reward_mode": "minimal", "seed": 42, "device": "cuda", "learning_rate": 0.0001, "gamma": 0.99, "replay_capacity": 100000, "batch_size": 128, "learning_starts": 5000, "target_sync_updates": 5000, "gradient_clip": 10.0, "unreachable_penalty": -0.05, "epsilon_decay_override": 38000, "reachability_mask_mode": "oracle"}`
- 실제payload: top-level22keys,30finite tensors, online/target/Adam 포함, replay/RNG없음. Stored experiences 2000000, gradient updates 1995001, target sync 399.
- 저장시각(UTC)/status: 2026-10-07T16:02:52.133018+00:00 / running (Commit일과구분).
- 의존성: source revision`3801e53c236262b8196880b1fb7392559e91b13a`, profile`native_tetris_031_oracle_perf_v2`, Native0.3.1의19SHA256, Torch2.14.1 CPU runtime. 역사launch SHA는미확인.
- 이번평가: fixed256states, ε=0/oracle, first-choice reachable 100.0000%, U 0/256. Gameplay우수성판정아님.
- 이번추론: `research_oracle_greedy, main_oracle_play_witness`, seed100000/limit6, optimizer/replayupdates0, 원file·weights불변.
- 과거 artifact별 gameplay: 미확인. 다른periodic/final의결과로대체하지않음.
- 초기화/데이터출처: 사용자진술random initialization·개인PC학습·환경생성replay. 현재코드/payload와부합하나각launch초기weights의증명은아님.
- 권리/사용조건: 사용자직접학습진술에근거한공개후보, model MIT 적용/사용자 최종승인. MIT 고지와license를동봉한다. 사용자가권리자표시HighKcal을최종승인했다.
- 한계: 정확resume불가, 장기성능/feasibility학습해결보장없음. OracleU0은mask효과. 원본만배포준비했으며변환없음.

## 9. oracle_mask.seed42.2m / final

- Release Asset: `Temeong-oracle_mask-seed42-2m-final.pt`
- Registry 원filename/배치위치: `runs/oracle_mask/seed_42/2m/checkpoint_final.pt`
- Bytes / SHA256: 5322057 / `ada68fbe3e5c6e68775212f906e60ab031aa0d27948c0d833643bffafe457ab2`
- Seed / 학습budget / 저장decisions / role: 42 / 2000000 / 2000000 / final
- 정책조건: oracle, U penalty -0.05, DDQN/minimal.
- 학습설정: `{"decision_budget": 2000000, "algorithm": "double", "reward_mode": "minimal", "seed": 42, "device": "cuda", "learning_rate": 0.0001, "gamma": 0.99, "replay_capacity": 100000, "batch_size": 128, "learning_starts": 5000, "target_sync_updates": 5000, "gradient_clip": 10.0, "unreachable_penalty": -0.05, "epsilon_decay_override": 38000, "reachability_mask_mode": "oracle"}`
- 실제payload: top-level22keys,30finite tensors, online/target/Adam 포함, replay/RNG없음. Stored experiences 2000000, gradient updates 1995001, target sync 399.
- 저장시각(UTC)/status: 2026-10-07T16:02:52.145018+00:00 / completed (Commit일과구분).
- 의존성: source revision`3801e53c236262b8196880b1fb7392559e91b13a`, profile`native_tetris_031_oracle_perf_v2`, Native0.3.1의19SHA256, Torch2.14.1 CPU runtime. 역사launch SHA는미확인.
- 이번평가: fixed256states, ε=0/oracle, first-choice reachable 100.0000%, U 0/256. Gameplay우수성판정아님.
- 이번추론: `research_oracle_greedy, main_oracle_play_witness`, seed100000/limit6, optimizer/replayupdates0, 원file·weights불변.
- 과거gameplay: historical_holdout, 50episodes, seeds100200부터, ε0, 132.778265lines/1000placements, 6.06lines/episode, deaths50. Raw protocol/집계는 `docs/results` 사본 참고.
- 초기화/데이터출처: 사용자진술random initialization·개인PC학습·환경생성replay. 현재코드/payload와부합하나각launch초기weights의증명은아님.
- 권리/사용조건: 사용자직접학습진술에근거한공개후보, model MIT 적용/사용자 최종승인. MIT 고지와license를동봉한다. 사용자가권리자표시HighKcal을최종승인했다.
- 한계: 정확resume불가, 장기성능/feasibility학습해결보장없음. OracleU0은mask효과. 원본만배포준비했으며변환없음.
