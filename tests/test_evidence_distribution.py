from __future__ import annotations

import pytest

from umutextstats.evidence import (
    EvidenceOccurrence,
    build_positional_distribution,
)


def test_build_positional_distribution():
    occurrences = [
        EvidenceOccurrence(
            label="first",
            start=5,
            end=10,
        ),
        EvidenceOccurrence(
            label="middle",
            start=45,
            end=55,
        ),
        EvidenceOccurrence(
            label="last-a",
            start=75,
            end=80,
        ),
        EvidenceOccurrence(
            label="last-b",
            start=90,
            end=95,
        ),
    ]

    result = build_positional_distribution(
        occurrences=occurrences,
        reference_length=100,
        segments=3,
        offset_source="text",
    )

    assert result.reference_length == 100
    assert result.total_occurrences == 4

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


def test_distribution_without_occurrences():
    result = build_positional_distribution(
        occurrences=[],
        reference_length=90,
        segments=3,
    )

    assert result.total_occurrences == 0
    assert result.delta is None

    assert all(
        segment.share is None
        for segment in result.segments
    )


def test_distribution_rejects_invalid_segments():
    with pytest.raises(
        ValueError,
        match="segments must be greater than or equal to 1",
    ):
        build_positional_distribution(
            occurrences=[],
            reference_length=100,
            segments=0,
        )


def test_distribution_rejects_out_of_range_occurrence():
    occurrences = [
        EvidenceOccurrence(
            start=90,
            end=110,
        )
    ]

    with pytest.raises(
        ValueError,
        match="exceeds the reference length",
    ):
        build_positional_distribution(
            occurrences=occurrences,
            reference_length=100,
            segments=3,
        )


def test_distribution_supports_token_offsets():
    occurrences = [
        EvidenceOccurrence(
            label="noun",
            start=1,
            end=2,
            offset_source="tokens",
            offset_unit="tokens",
        ),
        EvidenceOccurrence(
            label="verb",
            start=8,
            end=9,
            offset_source="tokens",
            offset_unit="tokens",
        ),
    ]

    result = build_positional_distribution(
        occurrences=occurrences,
        reference_length=10,
        segments=2,
        offset_source="tokens",
        offset_unit="tokens",
    )

    assert [
        segment.count
        for segment in result.segments
    ] == [
        1,
        1,
    ]

    assert result.offset_unit == "tokens"