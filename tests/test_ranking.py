import pandas as pd

from umutextstats.summary.ranking import (
    get_sparse_features,
    get_zero_only_features,
    rank_features,
    summary_to_long_dataframe,
)
from umutextstats.summary.summarize import summarize_features


def build_test_summary():
    df = pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "feature_a": [10, 20, 30, 40],
            "feature_b": [1, 0, 0, 0],
            "feature_c": [0, 0, 0, 0],
        }
    )

    return summarize_features(df)


def test_summary_to_long_dataframe():
    summary = build_test_summary()

    long_df = summary_to_long_dataframe(summary)

    assert not long_df.empty

    assert "feature" in long_df.columns
    assert "statistic" in long_df.columns
    assert "value" in long_df.columns


def test_rank_features_descending():
    summary = build_test_summary()

    ranking = rank_features(
        summary,
        by="mean",
        top=2,
        ascending=False,
    )

    assert len(ranking) == 2

    assert ranking.iloc[0]["feature"] == "feature_a"
    assert ranking.iloc[1]["feature"] == "feature_b"


def test_rank_features_ascending():
    summary = build_test_summary()

    ranking = rank_features(
        summary,
        by="mean",
        top=1,
        ascending=True,
    )

    assert ranking.iloc[0]["feature"] == "feature_c"


def test_get_zero_only_features():
    summary = build_test_summary()

    zero_df = get_zero_only_features(summary)

    assert not zero_df.empty

    assert "feature_c" in zero_df["feature"].values
    assert "feature_a" not in zero_df["feature"].values


def test_get_sparse_features():
    summary = build_test_summary()

    sparse_df = get_sparse_features(
        summary,
        threshold=0.30,
    )

    assert not sparse_df.empty

    assert "feature_c" in sparse_df["feature"].values