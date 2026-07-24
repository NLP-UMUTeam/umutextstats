from __future__ import annotations

import pandas as pd
import pytest

from umutextstats.dimensions.ratio import (
    RatioDimension,
)


def test_ratio_single_compute_and_result_are_consistent():
    dimension = RatioDimension(
        key="test-ratio",
        numerator=[
            "numerator_a",
            "numerator_b",
        ],
        denominator=[
            "denominator_a",
            "denominator_b",
        ],
        scale=100.0,
        zero_division=0.0,
    )

    df = pd.DataFrame(
        {
            "numerator_a": [2.0],
            "numerator_b": [3.0],
            "denominator_a": [4.0],
            "denominator_b": [6.0],
        }
    )

    result = dimension.compute_result(df)
    values = dimension.compute(df)
    single_value = dimension.compute_single(
        df.iloc[0]
    )

    expected_numerator = 5.0
    expected_denominator = 10.0
    expected_value = 50.0

    assert result.numerators.iloc[0] == pytest.approx(
        expected_numerator
    )

    assert result.denominators.iloc[0] == pytest.approx(
        expected_denominator
    )

    assert result.values.iloc[0] == pytest.approx(
        expected_value
    )

    assert values.iloc[0] == pytest.approx(
        expected_value
    )

    assert single_value == pytest.approx(
        expected_value
    )


def test_ratio_inspection_uses_same_components_and_value():
    dimension = RatioDimension(
        key="test-ratio",
        numerator=[
            "positive",
            "neutral",
        ],
        denominator="total",
        scale=100.0,
        zero_division=0.0,
    )

    row = pd.Series(
        {
            "positive": 3.0,
            "neutral": 2.0,
            "total": 10.0,
        }
    )

    single_value = dimension.compute_single(
        row
    )

    inspection = dimension.inspect(
        row
    )

    assert single_value == pytest.approx(
        50.0
    )

    assert inspection.key == "test-ratio"
    assert inspection.class_name == "RatioDimension"
    assert inspection.matches == []
    assert inspection.discarded_matches == []

    assert "Value: 50.0" in inspection.debug_text
    assert (
        "Numerator keys: positive + neutral"
        in inspection.debug_text
    )
    assert (
        "Numerator value: 5.0"
        in inspection.debug_text
    )
    assert (
        "Denominator keys: total"
        in inspection.debug_text
    )
    assert (
        "Denominator value: 10.0"
        in inspection.debug_text
    )
    assert "Scale: 100.0" in inspection.debug_text
    assert (
        "Zero division: 0.0"
        in inspection.debug_text
    )


def test_ratio_multiple_rows_preserve_expected_values():
    dimension = RatioDimension(
        key="test-ratio",
        numerator=[
            "a",
            "b",
        ],
        denominator=[
            "total",
        ],
        scale=1.0,
        zero_division=0.0,
    )

    df = pd.DataFrame(
        {
            "a": [
                2.0,
                3.0,
                0.0,
            ],
            "b": [
                2.0,
                1.0,
                5.0,
            ],
            "total": [
                8.0,
                2.0,
                10.0,
            ],
        },
        index=[
            10,
            20,
            30,
        ],
    )

    result = dimension.compute_result(df)

    assert result.values.index.tolist() == [
        10,
        20,
        30,
    ]

    assert result.numerators.tolist() == pytest.approx(
        [
            4.0,
            4.0,
            5.0,
        ]
    )

    assert result.denominators.tolist() == pytest.approx(
        [
            8.0,
            2.0,
            10.0,
        ]
    )

    assert result.values.tolist() == pytest.approx(
        [
            0.5,
            2.0,
            0.5,
        ]
    )

    for position in range(len(df)):
        assert dimension.compute_single(
            df.iloc[position]
        ) == pytest.approx(
            result.values.iloc[position]
        )


def test_ratio_non_numeric_values_are_treated_as_zero():
    dimension = RatioDimension(
        key="test-ratio",
        numerator=[
            "a",
            "b",
        ],
        denominator=[
            "total",
        ],
        scale=1.0,
        zero_division=0.0,
    )

    df = pd.DataFrame(
        {
            "a": [
                "not-a-number",
            ],
            "b": [
                4.0,
            ],
            "total": [
                8.0,
            ],
        }
    )

    result = dimension.compute_result(
        df
    )

    single_value = dimension.compute_single(
        df.iloc[0]
    )

    assert result.numerators.iloc[0] == pytest.approx(
        4.0
    )

    assert result.denominators.iloc[0] == pytest.approx(
        8.0
    )

    assert result.values.iloc[0] == pytest.approx(
        0.5
    )

    assert single_value == pytest.approx(
        0.5
    )


def test_ratio_zero_denominator_is_consistent():
    """
    The configured zero_division value must be returned directly,
    without applying the ratio scale to it.
    """
    dimension = RatioDimension(
        key="test-ratio",
        numerator="matches",
        denominator="words",
        scale=100.0,
        zero_division=-1.0,
    )

    df = pd.DataFrame(
        {
            "matches": [
                3.0,
            ],
            "words": [
                0.0,
            ],
        }
    )

    result = dimension.compute_result(
        df
    )

    single_value = dimension.compute_single(
        df.iloc[0]
    )

    inspection = dimension.inspect(
        df.iloc[0]
    )

    assert single_value == pytest.approx(
        -1.0
    )

    assert result.values.iloc[0] == pytest.approx(
        -1.0
    )

    assert (
        "Value: -1.0"
        in inspection.debug_text
    )


def test_ratio_missing_numerator_column_raises_error():
    dimension = RatioDimension(
        key="test-ratio",
        numerator=[
            "missing-column",
        ],
        denominator=[
            "total",
        ],
    )

    df = pd.DataFrame(
        {
            "total": [
                10.0,
            ],
        }
    )

    with pytest.raises(
        ValueError,
        match="missing-column",
    ):
        dimension.compute_result(
            df
        )


def test_ratio_missing_denominator_column_raises_error():
    dimension = RatioDimension(
        key="test-ratio",
        numerator=[
            "matches",
        ],
        denominator=[
            "missing-total",
        ],
    )

    df = pd.DataFrame(
        {
            "matches": [
                2.0,
            ],
        }
    )

    with pytest.raises(
        ValueError,
        match="missing-total",
    ):
        dimension.compute_result(
            df
        )


def test_ratio_result_metadata_describes_computation():
    dimension = RatioDimension(
        key="test-ratio",
        numerator=[
            "a",
            "b",
        ],
        denominator=[
            "total",
        ],
        scale=100.0,
        zero_division=-1.0,
    )

    df = pd.DataFrame(
        {
            "a": [
                2.0,
            ],
            "b": [
                3.0,
            ],
            "total": [
                10.0,
            ],
        }
    )

    result = dimension.compute_result(
        df
    )

    assert result.metadata == {
        "measure": "ratio",
        "numerator_dimensions": [
            "a",
            "b",
        ],
        "denominator_dimensions": [
            "total",
        ],
        "scale": 100.0,
        "zero_division": -1.0,
        "formula": (
            "scale * "
            "sum(numerator_dimensions) / "
            "sum(denominator_dimensions)"
        ),
    }