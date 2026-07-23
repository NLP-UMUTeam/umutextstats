from __future__ import annotations

from dataclasses import dataclass
from math import isclose
from typing import Iterable

from umutextstats.evidence.models import (
    PositionalDistribution,
)


@dataclass(frozen=True)
class AggregatedPositionalDistribution:
    """
    Corpus-level aggregation of positional evidence distributions.

    `occurrence_shares` gives equal weight to every occurrence.

    `mean_document_shares` gives equal weight to every document
    containing at least one occurrence.
    """

    segments: int
    document_count: int
    documents_with_evidence: int
    total_occurrences: int
    occurrence_counts: list[int]
    occurrence_shares: list[float]
    mean_document_shares: list[float]
    occurrence_delta: float | None
    mean_document_delta: float | None


def aggregate_positional_distributions(
    distributions: Iterable[
        PositionalDistribution
    ],
) -> AggregatedPositionalDistribution:
    """
    Aggregate document-level positional distributions.

    Two complementary summaries are produced:

    - occurrence shares: every occurrence has equal weight;
    - mean document shares: every non-empty document has equal weight.
    """
    items = list(distributions)

    if not items:
        raise ValueError(
            "At least one positional distribution is required."
        )

    segment_count = len(
        items[0].segments
    )

    if segment_count == 0:
        raise ValueError(
            "Positional distributions must contain "
            "at least one segment."
        )

    occurrence_counts = [
        0
        for _ in range(segment_count)
    ]

    document_share_sums = [
        0.0
        for _ in range(segment_count)
    ]

    documents_with_evidence = 0

    for document_index, distribution in enumerate(
        items
    ):
        _validate_distribution(
            distribution=distribution,
            expected_segments=segment_count,
            document_index=document_index,
        )

        for index, segment in enumerate(
            distribution.segments
        ):
            occurrence_counts[index] += (
                segment.count
            )

        if distribution.total_occurrences == 0:
            continue

        documents_with_evidence += 1

        for index, segment in enumerate(
            distribution.segments
        ):
            if segment.share is None:
                raise ValueError(
                    "A non-empty positional distribution "
                    "contains a segment without a share."
                )

            document_share_sums[index] += (
                segment.share
            )

    total_occurrences = sum(
        occurrence_counts
    )

    occurrence_shares = _normalize_counts(
        occurrence_counts
    )

    mean_document_shares = (
        [
            value / documents_with_evidence
            for value in document_share_sums
        ]
        if documents_with_evidence
        else [
            0.0
            for _ in range(segment_count)
        ]
    )

    occurrence_delta = (
        occurrence_shares[-1]
        - occurrence_shares[0]
        if total_occurrences
        else None
    )

    mean_document_delta = (
        mean_document_shares[-1]
        - mean_document_shares[0]
        if documents_with_evidence
        else None
    )

    return AggregatedPositionalDistribution(
        segments=segment_count,
        document_count=len(items),
        documents_with_evidence=(
            documents_with_evidence
        ),
        total_occurrences=total_occurrences,
        occurrence_counts=occurrence_counts,
        occurrence_shares=occurrence_shares,
        mean_document_shares=(
            mean_document_shares
        ),
        occurrence_delta=occurrence_delta,
        mean_document_delta=(
            mean_document_delta
        ),
    )


def _validate_distribution(
    distribution: PositionalDistribution,
    *,
    expected_segments: int,
    document_index: int,
) -> None:
    """
    Validate one document-level distribution before aggregation.
    """
    if (
        len(distribution.segments)
        != expected_segments
    ):
        raise ValueError(
            "All positional distributions must have "
            "the same number of segments. "
            f"Document at position {document_index} has "
            f"{len(distribution.segments)} segments; "
            f"expected {expected_segments}."
        )

    segment_occurrences = sum(
        segment.count
        for segment in distribution.segments
    )

    if (
        segment_occurrences
        != distribution.total_occurrences
    ):
        raise ValueError(
            "The sum of segment counts does not match "
            "the distribution's total occurrences. "
            f"Document at position {document_index}: "
            f"{segment_occurrences} != "
            f"{distribution.total_occurrences}."
        )

    if distribution.total_occurrences == 0:
        return

    shares = [
        segment.share
        for segment in distribution.segments
    ]

    if any(
        share is None
        for share in shares
    ):
        raise ValueError(
            "A non-empty positional distribution "
            "contains a segment without a share."
        )

    share_total = sum(
        float(share)
        for share in shares
        if share is not None
    )

    if not isclose(
        share_total,
        1.0,
        rel_tol=1e-9,
        abs_tol=1e-9,
    ):
        raise ValueError(
            "The shares of a non-empty positional "
            "distribution must sum to 1. "
            f"Document at position {document_index}: "
            f"{share_total}."
        )


def _normalize_counts(
    counts: list[int],
) -> list[float]:
    """
    Convert occurrence counts into corpus-level shares.
    """
    total = sum(counts)

    if total == 0:
        return [
            0.0
            for _ in counts
        ]

    return [
        count / total
        for count in counts
    ]