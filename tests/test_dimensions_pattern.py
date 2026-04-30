import pandas as pd

from umutextstats.dimensions.pattern import PatternDimension


def test_pattern_dimension_counts_matches():
    df = pd.DataFrame({
        "text_raw": ["holaaaa mundo", "normal"],
    })

    dim = PatternDimension(
        key="expressive",
        pattern=r"(.)\1{3,}",
        input_column="text_raw",
    )

    result = dim.compute(df)

    assert list(result) == [1, 0]


def test_pattern_dimension_percentage():
    df = pd.DataFrame({
        "text_raw": ["aaaabbbb", ""],
    })

    dim = PatternDimension(
        key="expressive",
        pattern=r"(.)\1{3,}",
        input_column="text_raw",
        percentage=True,
    )

    result = dim.compute(df)

    assert list(result) == [25.0, 0.0]