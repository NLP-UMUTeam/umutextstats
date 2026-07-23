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
    *,
    offset_source: str | None = None,
    offset_unit: str = "characters",
) -> DimensionInspection:
    return DimensionInspection(
        key="negative",
        class_name="WordPerDictionary",
        pattern=None,
        dictionary="negative",
        matches=matches,
        offset_source=offset_source,
        offset_unit=offset_unit,
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

    result, reference_text = build_match_distribution(
        inspection=inspection,
        text="x" * 100,
        segments=3,
    )

    assert reference_text == "x" * 100
    assert result.reference_length == 100
    assert result.total_occurrences == 4
    assert result.offset_source == "text"
    assert result.offset_unit == "characters"

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

    result, reference_text = build_match_distribution(
        inspection=inspection,
        text="x" * 90,
        segments=3,
    )

    assert reference_text == "x" * 90
    assert result.reference_length == 90
    assert result.total_occurrences == 0

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

    result, reference_text = build_match_distribution(
        inspection=inspection,
        text="x" * 50,
        segments=1,
    )

    assert reference_text == "x" * 50
    assert len(result.segments) == 1
    assert result.total_occurrences == 1
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

    result, reference_text = build_match_distribution(
        inspection=inspection,
        text="",
        segments=3,
    )

    assert reference_text == ""
    assert result.reference_length == 0
    assert result.total_occurrences == 0

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


def test_build_match_distribution_uses_annotation_reference():
    inspection = make_inspection(
        [
            InspectMatch(
                match="NOUN",
                start=10,
                end=14,
            ),
        ],
        offset_source="tagged_pos",
    )

    annotations = {
        "tagged_pos": "x" * 30,
    }

    result, reference_text = build_match_distribution(
        inspection=inspection,
        text="short text",
        annotations=annotations,
        segments=3,
    )

    assert reference_text == "x" * 30
    assert result.reference_length == 30
    assert result.offset_source == "tagged_pos"
    assert result.offset_unit == "characters"

    assert [
        segment.count
        for segment in result.segments
    ] == [
        0,
        1,
        0,
    ]


def test_build_match_distribution_requires_annotation_reference():
    inspection = make_inspection(
        [],
        offset_source="tagged_pos",
    )

    with pytest.raises(
        ValueError,
        match=(
            "Positional reference "
            "'tagged_pos' requires annotations"
        ),
    ):
        build_match_distribution(
            inspection=inspection,
            text="text",
            annotations=None,
            segments=3,
        )


def test_build_match_distribution_rejects_missing_annotation():
    inspection = make_inspection(
        [],
        offset_source="tagged_pos",
    )

    with pytest.raises(
        ValueError,
        match=(
            "Positional reference "
            "'tagged_pos' is not available"
        ),
    ):
        build_match_distribution(
            inspection=inspection,
            text="text",
            annotations={
                "tagged_ner": "annotation",
            },
            segments=3,
        )


def test_build_match_distribution_preserves_token_unit():
    inspection = make_inspection(
        [
            InspectMatch(
                match="noun",
                start=1,
                end=2,
            ),
            InspectMatch(
                match="verb",
                start=8,
                end=9,
            ),
        ],
        offset_source="tagged_pos",
        offset_unit="tokens",
    )

    result, reference_text = build_match_distribution(
        inspection=inspection,
        text="unused",
        annotations={
            "tagged_pos": "x" * 10,
        },
        segments=2,
    )

    assert reference_text == "x" * 10
    assert result.offset_source == "tagged_pos"
    assert result.offset_unit == "tokens"

    assert [
        segment.count
        for segment in result.segments
    ] == [
        1,
        1,
    ]


from umutextstats.config.inspect import (
    build_distribution_sparkline,
)


def test_build_distribution_sparkline():
    inspection = make_inspection(
        [
            InspectMatch(
                match="middle-a",
                start=40,
                end=41,
            ),
            InspectMatch(
                match="middle-b",
                start=50,
                end=51,
            ),
            InspectMatch(
                match="last-a",
                start=75,
                end=76,
            ),
            InspectMatch(
                match="last-b",
                start=85,
                end=86,
            ),
        ]
    )

    distribution, _ = build_match_distribution(
        inspection=inspection,
        text="x" * 100,
        segments=3,
    )

    assert build_distribution_sparkline(
        distribution
    ) == "▁██"