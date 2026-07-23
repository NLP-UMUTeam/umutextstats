from __future__ import annotations

import pytest

from umutextstats.evidence import (
    EvidenceOccurrence,
    aggregate_positional_distributions,
    build_positional_distribution,
)


def make_occurrence(
    start: int,
    end: int | None = None,
) -> EvidenceOccurrence:
    if end is None:
        end = start + 1

    return EvidenceOccurrence(
        start=start,
        end=end,
        offset_source="text_norm",
        offset_unit="characters",
    )


def make_distribution(
    positions: list[int],
    *,
    reference_length: int = 90,
    segments: int = 3,
):
    return build_positional_distribution(
        occurrences=[
            make_occurrence(position)
            for position in positions
        ],
        reference_length=reference_length,
        segments=segments,
        offset_source="text_norm",
        offset_unit="characters",
    )


def test_aggregate_positional_distributions():
    distributions = [
        # Shares: [0.5, 0.5, 0.0]
        make_distribution(
            [
                10,
                40,
            ]
        ),
        # Shares: [0.0, 0.0, 1.0]
        make_distribution(
            [
                70,
                80,
            ]
        ),
        # Empty document.
        make_distribution(
            []
        ),
    ]

    result = (
        aggregate_positional_distributions(
            distributions
        )
    )

    assert result.segments == 3
    assert result.document_count == 3
    assert result.documents_with_evidence == 2
    assert result.total_occurrences == 4

    assert result.occurrence_counts == [
        1,
        1,
        2,
    ]

    assert result.occurrence_shares == pytest.approx(
        [
            0.25,
            0.25,
            0.50,
        ]
    )

    assert result.mean_document_shares == (
        pytest.approx(
            [
                0.25,
                0.25,
                0.50,
            ]
        )
    )

    assert result.occurrence_delta == (
        pytest.approx(0.25)
    )

    assert result.mean_document_delta == (
        pytest.approx(0.25)
    )


def test_occurrence_and_document_weighting_can_differ():
    distributions = [
        # One document with four initial occurrences.
        make_distribution(
            [
                2,
                5,
                10,
                20,
            ]
        ),
        # One document with one final occurrence.
        make_distribution(
            [
                80,
            ]
        ),
    ]

    result = (
        aggregate_positional_distributions(
            distributions
        )
    )

    assert result.occurrence_counts == [
        4,
        0,
        1,
    ]

    assert result.occurrence_shares == pytest.approx(
        [
            0.8,
            0.0,
            0.2,
        ]
    )

    # Each non-empty document receives the same weight.
    assert result.mean_document_shares == (
        pytest.approx(
            [
                0.5,
                0.0,
                0.5,
            ]
        )
    )

    assert result.occurrence_delta == (
        pytest.approx(-0.6)
    )

    assert result.mean_document_delta == (
        pytest.approx(0.0)
    )


def test_empty_documents_are_counted_but_not_averaged():
    distributions = [
        make_distribution([]),
        make_distribution([]),
    ]

    result = (
        aggregate_positional_distributions(
            distributions
        )
    )

    assert result.document_count == 2
    assert result.documents_with_evidence == 0
    assert result.total_occurrences == 0

    assert result.occurrence_counts == [
        0,
        0,
        0,
    ]

    assert result.occurrence_shares == [
        0.0,
        0.0,
        0.0,
    ]

    assert result.mean_document_shares == [
        0.0,
        0.0,
        0.0,
    ]

    assert result.occurrence_delta is None
    assert result.mean_document_delta is None


def test_rejects_empty_collection():
    with pytest.raises(
        ValueError,
        match=(
            "At least one positional "
            "distribution is required"
        ),
    ):
        aggregate_positional_distributions(
            []
        )


def test_rejects_different_segment_counts():
    distributions = [
        make_distribution(
            [10],
            segments=3,
        ),
        make_distribution(
            [10],
            segments=4,
        ),
    ]

    with pytest.raises(
        ValueError,
        match=(
            "same number of segments"
        ),
    ):
        aggregate_positional_distributions(
            distributions
        )