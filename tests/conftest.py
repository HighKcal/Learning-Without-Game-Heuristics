"""공개 단위 검사와 비공개 자료가 필요한 역사 진단의 실행 경계."""
from pathlib import Path

import pytest


# 기존 진단 파일은 저장된 implementation hash의 대상이므로 수정하지 않는다.
# fixture 없이 과거 파일을 직접 읽는 세 case도 명시적으로 분류한다.
HISTORICAL_INPUTS = {
    "test_u_competing_actions_diagnostic.py": (
        "historical", {"test_seed123_saved_inputs_and_schedule"}
    ),
    "test_u_feedback_timing_diagnostic.py": (
        "data", {"test_background_occurrence_multiplicity_source_integrity"}
    ),
}


def pytest_addoption(parser):
    parser.addoption(
        "--run-historical-diagnostics", action="store_true", default=False,
        help="비공개 원본 입력을 갖춘 격리 checkout에서 역사 진단을 실행한다.",
    )


def pytest_collection_modifyitems(config, items):
    enabled = config.getoption("--run-historical-diagnostics")
    for item in items:
        spec = HISTORICAL_INPUTS.get(Path(item.path).name)
        if spec is None:
            continue
        fixture, direct_readers = spec
        name = getattr(item, "originalname", item.name)
        if fixture not in item.fixturenames and name not in direct_readers:
            continue
        item.add_marker(pytest.mark.historical_diagnostic)
        if not enabled:
            item.add_marker(pytest.mark.skip(
                reason="비공개 역사 입력 필요: --run-historical-diagnostics로 opt-in"
            ))
