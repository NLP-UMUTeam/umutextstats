import pandas as pd

from umutextstats.summary.summarize import (
    get_numeric_feature_columns,
    summarize_features,
    summarize_features_long,
)


def test_get_numeric_feature_columns():
    df = pd.DataFrame(
        {
            "id": [1, 2],
            "feature_a": [1.0, 2.0],
            "feature_b": [3, 4],
            "text": ["hello", "world"],
            "category": ["a", "b"],
        }
    )

    columns = get_numeric_feature_columns(df)

    assert "feature_a" in columns
    assert "feature_b" in columns

    assert "id" not in columns
    assert "text" not in columns
    assert "category" not in columns


def test_summarize_features():
    df = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "feature_a": [1, 2, 3],
            "feature_b": [10, 20, 30],
        }
    )

    summary = summarize_features(df)

    assert summary["metadata"]["documents"] == 3
    assert summary["metadata"]["features"] == 2

    assert summary["summary"]["feature_a"]["mean"] == 2.0
    assert summary["summary"]["feature_b"]["mean"] == 20.0


def test_summarize_features_long():
    df = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "feature_a": [1, 2, 3],
        }
    )

    summary = summarize_features_long(df)

    assert not summary.empty

    assert "feature" in summary.columns
    assert "statistic" in summary.columns
    assert "value" in summary.columns

    mean_row = summary[
        (summary["feature"] == "feature_a")
        & (summary["statistic"] == "mean")
    ]

    assert not mean_row.empty
    assert mean_row.iloc[0]["value"] == 2.0


def test_summarize_features_with_non_numeric_columns():
    df = pd.DataFrame(
        {
            "id": [1, 2],
            "feature_a": [1, 2],
            "label": ["positive", "negative"],
        }
    )

    summary = summarize_features(df)

    assert "feature_a" in summary["summary"]
    assert "label" not in summary["summary"]