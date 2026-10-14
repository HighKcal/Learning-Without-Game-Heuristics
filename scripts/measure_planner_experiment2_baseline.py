"""Remeasure the exact Experiment 1 commit without replacing workspace source."""
import hashlib
import sys

import _planner_benchmark as helper
from validate_planner_experiment2 import baseline_class, BASELINE_SOURCE

original_metadata = helper.metadata


def baseline_metadata():
    value = original_metadata()
    value['measured_implementation'] = 'f961818:src/tetris_agent/planner.py'
    value['measured_planner_sha256'] = hashlib.sha256(
        BASELINE_SOURCE.read_bytes()).hexdigest()
    value['note'] = 'Baseline loaded in memory; tracked_sha256 describes workspace files.'
    return value


import benchmark_planner

if __name__ == '__main__':
    helper.TacticalPlanner = baseline_class()
    helper.metadata = baseline_metadata
    benchmark_planner.TacticalPlanner = helper.TacticalPlanner
    benchmark_planner.metadata = baseline_metadata
    benchmark_planner.main()
