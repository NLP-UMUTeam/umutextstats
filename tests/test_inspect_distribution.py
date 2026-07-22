from __future__ import annotations

import pytest

from umutextstats.config.inspect import (
    build_match_distribution,
)
from umutextstats.inspection.models import (
    DimensionInspection,
    InspectMatch,
)


def make_inspection(
    matches: list[InspectMatch],
) -> DimensionInspection:
    return DimensionInspection(
        key="negative",
        class_name="WordPerDictionary",
        pattern=None,
        dictionary="negative",
        matches=matches,
    )


def test_build_match_distribution_three_segments():
    inspection = make_inspection(
        [
            InspectMatch(
                match="first",
                start=5,
                end=10,
            ),
            InspectMatch(
                match="middle",
                start=45,
                end=55,
            ),
            InspectMatch(
                match="last-a",
                start=75,
                end=80,
            ),
            InspectMatch(
                match="last-b",
                start=90,
                end=95,
            ),
        ]
    )

    result = build_match_distribution(
        inspection=inspection,
        text="x" * 100,
        segments=3,
    )

    assert result.text_length == 100
    assert result.total_matches == 4

    assert [
        segment.count
        for segment in result.segments
    ] == [
        1,
        1,
        2,
    ]

    assert [
        segment.share
        for segment in result.segments
    ] == pytest.approx(
        [
            0.25,
            0.25,
            0.50,
        ]
    )

    assert result.delta == pytest.approx(
        0.25
    )


def test_build_match_distribution_without_matches():
    inspection = make_inspection(
        []
    )

    result = build_match_distribution(
        inspection=inspection,
        text="x" * 90,
        segments=3,
    )

    assert result.total_matches == 0

    assert [
        segment.count
        for segment in result.segments
    ] == [
        0,
        0,
        0,
    ]

    assert all(
        segment.share is None
        for segment in result.segments
    )

    assert result.delta is None


def test_build_match_distribution_one_segment():
    inspection = make_inspection(
        [
            InspectMatch(
                match="test",
                start=20,
                end=24,
            )
        ]
    )

    result = build_match_distribution(
        inspection=inspection,
        text="x" * 50,
        segments=1,
    )

    assert len(
        result.segments
    ) == 1

    assert result.segments[0].count == 1
    assert result.segments[0].share == 1.0
    assert result.delta == 0.0


def test_build_match_distribution_rejects_zero_segments():
    inspection = make_inspection(
        []
    )

    with pytest.raises(
        ValueError,
        match=(
            "segments must be greater "
            "than or equal to 1"
        ),
    ):
        build_match_distribution(
            inspection=inspection,
            text="text",
            segments=0,
        )


def test_build_match_distribution_handles_empty_text():
    inspection = make_inspection(
        []
    )

    result = build_match_distribution(
        inspection=inspection,
        text="",
        segments=3,
    )

    assert result.text_length == 0

    assert [
        segment.start
        for segment in result.segments
    ] == [
        0,
        0,
        0,
    ]

    assert [
        segment.end
        for segment in result.segments
    ] == [
        0,
        0,
        0,
    ]