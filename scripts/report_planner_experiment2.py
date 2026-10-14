"""Summarize the separate correctness, wall-clock and profile evidence."""
import hashlib
import json
from pathlib import Path
import pstats
import subprocess

from _planner_benchmark import ROOT, write_json

OUT = ROOT / 'benchmarks/planner/live_drop_probe'
REPORT_OUTPUT = ROOT / '_validation/planner_historical_report'


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def profile(path):
    result = {}
    for (filename, line, name), (_, calls, own, cumulative, _) in pstats.Stats(str(path)).stats.items():
        if name in ('plan', 'transition', 'restore_branch', '_state_key') and Path(filename).name == 'planner.py':
            label = name
        elif name in ('get_state', 'set_state', 'step', '_get_obs') and Path(filename).name == 'tetris.py':
            label = 'Tetris.' + name
        elif name == '__copy__' and Path(filename).name == 'tetromino_randomizer.py':
            label = 'BagRandomizer.__copy__'
        else:
            continue
        result[label] = {'calls': calls, 'own_s': own, 'cumulative_s': cumulative}
    return result


def metric_rows(before, after):
    b = before['benchmark']['pooled_3_runs']
    a = after['benchmark']['pooled_3_runs']
    rows = []
    for group, metrics in (('overall', ('total', 'avg', 'median', 'p95', 'p99', 'max')),
                            ('REACHABLE', ('avg', 'p95')), ('UNREACHABLE', ('avg', 'p95'))):
        for key in metrics:
            x, y = b[group]['elapsed_ms'][key], a[group]['elapsed_ms'][key]
            rows.append((group + ' ' + key + ' (ms)', x, y, (y / x - 1) * 100))
    x, y = b['overall']['weighted_ms_per_expanded'], a['overall']['weighted_ms_per_expanded']
    rows.append(('weighted ms/expanded', x, y, (y / x - 1) * 100))
    x, y = before['e2e']['weighted_planner_ratio'], after['e2e']['weighted_planner_ratio']
    rows.append(('E2E Planner ratio (%)', x * 100, y * 100, (y / x - 1) * 100))
    for key in ('planner_total_s', 'game_total_s'):
        x, y = before['e2e'][key], after['e2e'][key]
        rows.append(('E2E ' + key, x, y, (y / x - 1) * 100))
    return rows


def main():
    REPORT_OUTPUT.mkdir(parents=True,exist_ok=False)
    old = load(ROOT / 'benchmarks/planner/snapshot_after_dedup/planner_v001_baseline.json')
    before = load(OUT / 'baseline_remeasure/planner_v001_baseline.json')
    after = load(OUT / 'planner_v001_baseline.json')
    validation = load(OUT / 'strict_validation.json')
    initial = load(OUT / 'initial_state.json')
    baseline_profile_metadata = load(ROOT / 'benchmarks/planner/snapshot_after_dedup/planner_v001_profile.json')['metadata']
    assert baseline_profile_metadata['tracked_sha256']['src/tetris_agent/planner.py'] == initial['baseline_source_sha256']
    preserved = {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() == digest
                 for p, digest in initial['preserved_artifact_hashes'].items()}
    assert all(preserved.values())
    tracked_changes = subprocess.check_output(['git', 'diff', '--name-only'], cwd=ROOT, text=True).splitlines()
    assert tracked_changes == ['src/tetris_agent/planner.py'], tracked_changes
    profiles = {g: {'before': profile(ROOT / f'benchmarks/experiment1/planner_v001_{g}.prof'),
                    'after': profile(OUT / f'planner_v001_{g}.prof')}
                for g in ('overall', 'reachable', 'unreachable', 'controlled')}
    same = ('Tetris.step', 'Tetris._get_obs', '_state_key')
    for group in profiles.values():
        assert all(group['before'][name]['calls'] == group['after'][name]['calls'] for name in same)
    overall = profiles['overall']
    assert overall['after']['Tetris.get_state']['calls'] == 270352
    assert overall['after']['BagRandomizer.__copy__']['calls'] == 270352
    changes = metric_rows(before, after)
    runs = [{'run': index + 1, 'before_ms': b['overall']['elapsed_ms']['total'],
             'after_ms': a['overall']['elapsed_ms']['total'],
             'change_pct': (a['overall']['elapsed_ms']['total'] / b['overall']['elapsed_ms']['total'] - 1) * 100}
            for index, (b, a) in enumerate(zip(before['benchmark']['runs'], after['benchmark']['runs']))]
    recommendation = ('유지 후보' if changes[0][3] < 0 and all(r['change_pct'] < 0 for r in runs)
                      else 'latency 재현 부족: revert 권고')
    write_json(REPORT_OUTPUT / 'comparison.json', {'validation': validation, 'benchmark_before': before,
        'benchmark_after': after, 'stored_experiment1_reference': old,
        'metrics': [{'metric': name, 'before': x, 'after': y, 'change_pct': delta}
                    for name, x, y, delta in changes], 'runs': runs, 'profiles': profiles,
        'preserved_artifacts': preserved, 'tracked_changes': tracked_changes,
        'recommendation': recommendation})
    table = ['| 지표 | Experiment 1 재측정 | Experiment 2 | 변화 |', '|---|---:|---:|---:|']
    table += [f'| {name} | {x:.6f} | {y:.6f} | {delta:+.2f}% |' for name, x, y, delta in changes]
    prof_table = ['| 함수 | calls E1 → E2 | cumulative seconds E1 → E2 |', '|---|---:|---:|']
    for name, b in overall['before'].items():
        a = overall['after'][name]
        prof_table.append(f'| {name} | {b["calls"]:,} → {a["calls"]:,} | {b["cumulative_s"]:.6f} → {a["cumulative_s"]:.6f} |')
    boundaries = load(OUT / 'boundary_validation.json')
    controlled = load(OUT / 'strict_controlled_correctness.json')
    cases_table = ['| controlled fixture | status | actions | drop | expanded / visited | target |',
                   '|---|---|---|---|---:|---|']
    cases_table += [f'| {r["id"]} | {r["status"]} | {r["actions"]} | {r["needs_hard_drop"]} | '
                    f'{r["expanded_states"]} / {r["visited_states"]} | {r["target_verified"]} |' for r in controlled]
    controlled_timing = ['| controlled fixture | E1 avg ms | E2 avg ms | 변화 |',
                         '|---|---:|---:|---:|']
    for name, b in before['controlled'].items():
        x, y = b['elapsed_ms']['avg'], after['controlled'][name]['elapsed_ms']['avg']
        controlled_timing.append(f'| {name} | {x:.6f} | {y:.6f} | {(y / x - 1) * 100:+.2f}% |')
    report = f'''# Planner v0.02 Experiment 2

판정: **{recommendation}**. 기준 commit은 `f961818`; Experiment 2는 commit하지 않았다.

## 1–3. 구현 전 가설 / snapshot alias / HARD_DROP mutation

설치된 tetris-gymnasium 0.3.1 코드를 직접 확인했다. `source_audit.json`에 파일 경로, 줄 번호, SHA-256와 해당 메서드의 원문을 저장했다.

- `get_state()`는 board, active Tetromino 객체·matrix·color_rgb, queue 객체·deque, randomizer 객체·bag·RNG를 복사한다. snapshot queue는 snapshot randomizer를 참조한다.
- holder 객체와 deque는 별도지만 내부 Tetromino는 공유한다. native HARD_DROP은 HOLD를 호출하지 않고 holder·Tetromino metadata를 변형하지 않는다. `_get_obs()`는 holder가 반환한 새 list에 padded matrix를 넣는다.
- Gym `np_random`은 TetrisState에 없다. native commit/drop/place/clear/spawn/score/observer/observation 경로는 이를 사용하지 않는다.
- score는 설치 코드에서 int 누계가 아니라 bound callable이며 snapshot에도 같은 callable 참조를 저장한다. native score는 rows/width로 reward를 계산할 뿐 state를 수정하지 않는다. queue/holder copy의 deque maxlen 손실은 기존 설치 코드 동작이며 이번 변경 대상이 아니다. finally는 원래 live 객체/reference를 복원해 metadata를 보존한다.
- drop은 live y를 변경한다. place는 board.copy() projection을 설치하고 active 참조를 None으로 바꾼다. clear는 이 새 board를 제자리 수정한다. spawn은 live queue에서 pop/refill하고 live randomizer bag/RNG를 변경하며 active 참조·x/y·game_over/has_swapped를 갱신한다. score는 기본적으로 읽기 계산이다.
- lock observer는 placement 이전의 active pose를 읽고 원래 place를 호출한다. 저장된 child의 board·queue·bag·RNG·active metadata는 독립이라 이 변형을 받지 않는다. 공유 holder Tetromino도 native 경로에서는 읽기만 한다.
- `child = base.get_state()` 다음에는 `visited.add(key)`와 주석만 존재한다. live child를 변경하는 중간 호출은 없다. 가설이 성립한다.

## 4–5. 변경과 파일

`src/tetris_agent/planner.py`의 새 child discovery 지점에서 `transition(child, HARD_DROP)` 한 호출을 `locked_target = None`, `did_lock = False`, native `base.step(HARD_DROP)`로 대체했다. root probe, primitive transition, parent restore, lock observer, goal verification, 실패 시 child enqueue, BFS action order, visited 정의와 finally restoration은 동일하다.

검증/측정용 추가 파일: `scripts/validate_planner_experiment2.py`, `scripts/measure_planner_experiment2_baseline.py`, `scripts/report_planner_experiment2.py`. 결과는 `benchmarks/experiment2/`에만 저장했다. 기존 untracked 작업 파일은 유지했다. 이전 Experiment 1/results artifact {len(preserved)}개 SHA-256 모두 동일하다.

## 6–9. correctness와 경계 조건

- 기존 pytest: **215 passed in 7.00s**. 기존 테스트 수정·삭제·완화 없음.
- strict corpus: **1,003/1,003 동일**. 원본 v0.01이 아니라 `f961818` planner source를 별도 모듈로 직접 실행했다. status/actions/path length/needs_hard_drop/expanded/visited/target_verified와 target을 비교했다.
- 총 expanded={validation['expanded_states']:,}, visited={validation['visited_states']:,}. shortest primitive prefix와 search graph 결과 동일.
- controlled: **10/10 동일**. 위 필드에 더해 native action 순서와 step 직전 전체 snapshot 값도 동일.
- 특별 경계: **{len(boundaries)}/15 동일**. populated holder와 bag index 6(shuffle 직전), 0(직후), 1을 각각 spawn/line clear/next spawn game-over/sealed unreachable/auto-lock에 조합했다. 저장된 모든 child fingerprint가 native step 이후 동일하며, child discovery snapshot과 probe 직전 live 값도 동일함을 확인했다. line clear가 실제 일어났고, child HARD_DROP 성공 후 spawn game-over와 primitive auto-lock 결과를 별도 assertion으로 확인했다.
- optimized child HARD_DROP 직후 exception을 주입: baseline/candidate × 기존 observer 유무 **4 checks 통과**. observer, original 객체/reference, 전체 값과 Gym RNG identity/state 복원을 확인했다. 기존 pytest도 bag shuffle 이후 40-piece queue/RNG continuation, score override, custom action mapping/geometry, row clear와 exception 복원을 검증한다.

{chr(10).join(cases_table)}

## 10–12. plan-only 구조적 호출 수

| 항목 | Experiment 1 | Experiment 2 | 변화 |
|---|---:|---:|---:|
| Tetris.get_state | 317,397 | 270,352 | −47,045 (−14.82%) |
| BagRandomizer.__copy__ | 317,397 | 270,352 | −47,045 (−14.82%) |
| Tetris.step | 269,349 | 269,349 | 0 |
| Tetris._get_obs | 269,349 | 269,349 | 0 |
| HARD_DROP probe | 48,041 | 48,041 | 0 |
| _state_key (profile) | 182,587 | 182,587 | 0 |

집계 범위는 1,003 plan 호출만이다. 입력 restore·oracle·controlled·boundary 검증은 제외한다. 새 child restore 제거 47,045회와 정확히 일치한다. root HARD_DROP과 모든 primitive parent restore는 유지했다.

## 13. 비프로파일 benchmark

같은 corpus SHA-256 `{validation['corpus_sha256']}`, 입력 순서, 최초 10-call warm-up, 3 replay, 외부 perf_counter, 기존 controlled 5 warm-ups/20 timed calls, 같은 32 seed-pair E2E를 사용했다. correctness가 전부 통과한 뒤 baseline 재측정 → candidate 순서로 CPU 작업을 직렬 실행했다. baseline은 저장된 f961818 source를 메모리에 로드했으며 workspace source 교체는 없다. `baseline_remeasure/` metadata는 이 출처와 hash를 명시한다.

Planner total은 3 replay 합계, 나머지 latency는 3,009-call pooled 통계다. weighted ms/expanded는 기존 harness 정의대로 expanded>0인 입력만 사용한다. E2E ratio는 32게임 총 planner 시간 / 총 game 시간이다.

{chr(10).join(table)}

Replay 별 total 변화: {', '.join(f'run {r["run"]}: {r["change_pct"]:+.2f}%' for r in runs)}.

이전 저장된 Experiment 1 total {old['benchmark']['pooled_3_runs']['overall']['elapsed_ms']['total'] / 1000:.6f}s → 이번 Experiment 2 {after['benchmark']['pooled_3_runs']['overall']['elapsed_ms']['total'] / 1000:.6f}s ({(after['benchmark']['pooled_3_runs']['overall']['elapsed_ms']['total'] / old['benchmark']['pooled_3_runs']['overall']['elapsed_ms']['total'] - 1) * 100:+.2f}%). 주 비교는 이번 baseline 재측정이다. 양쪽 E2E executor failures=0.

{chr(10).join(controlled_timing)}

Controlled latency는 개선과 악화가 섞여 있다. E2E planner total은 {before['e2e']['planner_total_s']:.6f}s → {after['e2e']['planner_total_s']:.6f}s ({(after['e2e']['planner_total_s'] / before['e2e']['planner_total_s'] - 1) * 100:+.2f}%), game total은 {(after['e2e']['game_total_s'] / before['e2e']['game_total_s'] - 1) * 100:+.2f}%로 늘었다. E2E ratio의 소폭 감소만으로 게임 속도 개선을 주장하지 않는다. 별도 E2E는 한 pass이며 corpus replay의 작은 개선이 모든 workload에서 재현된다는 근거는 없다.

## 14. 별도 profile

Candidate profile은 비프로파일 benchmark 이후 별도 실행했다. before는 보존된 Experiment 1 profile이며 그 planner SHA-256가 f961818 source와 정확히 같음을 확인했다. overall/reachable/unreachable/controlled profile 모두 `comparison.json`에 calls/exclusive/cumulative를 저장했다. cumulative 시간은 호출된 하위 함수를 포함하므로 합산하지 않는다. profile timing은 latency benchmark가 아니다.

{chr(10).join(prof_table)}

transition calls 감소는 child probe가 native step을 직접 호출하기 때문이다. restore_branch는 같은 47,045회 감소한다. step/_get_obs/_state_key는 네 profile group 모두 동일하다.

overall profile plan cumulative는 {overall['before']['plan']['cumulative_s']:.6f}s → {overall['after']['plan']['cumulative_s']:.6f}s ({(overall['after']['plan']['cumulative_s'] / overall['before']['plan']['cumulative_s'] - 1) * 100:+.2f}%)로 소폭 늘었다. get_state/copy/restore cumulative는 감소했지만 native step/_get_obs cumulative는 늘었다. source hash와 calls가 일치하는 보존 baseline을 사용했으며 측정 시점과 profiler 영향을 포함하는 진단 수치이므로, 공식 latency 판단은 별도 비프로파일 replay를 사용한다.

## 15–16. 예상 밖 결과 / 위험 / 권고

예상 snapshot 호출 감소와 correctness 동일성을 확인했다. 검증기 초안은 RNG pickle 전체를 비교해 PCG64 생성 시 임의 SeedSequence entropy를 실제 RNG 상태 차이로 오인했다. 이를 semantic bit_generator.state 비교로 수정한 뒤 검증을 완료했다. planner correctness 차이는 없었다.

세 replay는 순차 실행의 반복 관측이며 randomized ABAB나 신뢰구간 측정은 아니다. max/tail 변화와 profile cumulative 시간에는 시스템 부하 및 실행 시점의 영향이 있을 수 있다. corpus/경계 검증은 모든 가능한 입력의 증명은 아니다. snapshot 독립성 판단은 현재 설치된 native 환경 기준이다. 공유 holder 객체나 Gym RNG를 임의로 수정하는 custom hook은 이 native 분석으로 보장하지 않는다(기존 planner도 이들을 깊은 복사하지 않는다).

권고: **{recommendation}**. correctness/action/search volume/branch isolation과 호출 감소가 일치하고, 세 비프로파일 corpus replay total이 모두 감소했다. 구현 복잡도는 한 지점의 간단한 변경 수준이다. 이 workload에서의 작은 개선을 근거로 유지 후보로 남기며, E2E 개선은 미확인이고 controlled 개별 입력에는 일부 악화가 있다는 제한을 함께 둔다. Experiment 3 이후 최적화와 commit은 수행하지 않았다.
'''
    (REPORT_OUTPUT / 'report.md').write_text(report, encoding='utf-8')
    print(json.dumps({'recommendation': recommendation, 'metrics': changes, 'runs': runs},
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
