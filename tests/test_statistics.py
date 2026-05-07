import pandas as pd

from umutextstats.summary.statistics import compute_statistics


def test_compute_statistics_numeric_series():
    series = pd.Series([1, 2, 2, 4])

    stats = compute_statistics(series)

    assert stats["count"] == 4
    assert stats["sum"] == 9.0
    assert stats["mean"] == 2.25
    assert stats["min"] == 1.0
    assert stats["median"] == 2.0
    assert stats["max"] == 4.0
    assert stats["mode"] == 2.0
    assert stats["nonzero_count"] == 4
    assert stats["nonzero_ratio"] == 1.0


def test_compute_statistics_handles_missing_and_non_numeric_values():
    series = pd.Series([1, 2, None, "bad"])

    stats = compute_statistics(series)

    assert stats["count"] == 2
    assert stats["missing_count"] == 2
    assert stats["missing_ratio"] == 0.5
    assert stats["sum"] == 3.0
    assert stats["mean"] == 1.5


def test_compute_statistics_handles_empty_series():
    series = pd.Series([])

    stats = compute_statistics(series)

    assert stats["count"] == 0
    assert stats["missing_count"] == 0
    assert stats["missing_ratio"] is None
    assert stats["sum"] is None
    assert stats["mean"] is None
    assert stats["std"] is None


def test_compute_statistics_with_selected_statistics():
    series = pd.Series([1, 2, 3])

    stats = compute_statistics(series, statistics=["mean", "max"])

    assert stats == {
        "mean": 2.0,
        "max": 3.0,
    }