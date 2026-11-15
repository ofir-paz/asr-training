"""Tests for preprocess/select_top_quality.py's budgeted best-first selection."""

import importlib.util
from pathlib import Path

import numpy as np

_spec = importlib.util.spec_from_file_location(
    "select_top_quality", Path(__file__).resolve().parents[2] / "preprocess" / "select_top_quality.py"
)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
select_top_quality = _module.select_top_quality


def test_takes_best_rows_until_budget_and_keeps_dataset_order():
    scores = np.array([0.5, 0.9, 0.7, 0.95, 0.1])
    durations = np.full(5, 30.0)
    np.testing.assert_array_equal(select_top_quality(scores, durations, 90), [1, 2, 3])


def test_never_exceeds_budget_with_uneven_durations():
    scores = np.array([0.9, 0.8, 0.7])
    durations = np.array([40.0, 30.0, 10.0])
    selected = select_top_quality(scores, durations, 75)
    np.testing.assert_array_equal(selected, [0, 1])
    assert durations[selected].sum() <= 75


def test_ties_resolve_to_dataset_order():
    scores = np.array([0.8, 0.8, 0.8, 0.8])
    np.testing.assert_array_equal(select_top_quality(scores, np.full(4, 30.0), 60), [0, 1])


def test_budget_below_one_row_selects_nothing():
    assert len(select_top_quality(np.array([0.9]), np.array([30.0]), 10)) == 0
