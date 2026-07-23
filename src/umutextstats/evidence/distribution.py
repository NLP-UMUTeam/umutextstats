from __future__ import annotations

from umutextstats.evidence.models import (
    EvidenceOccurrence,
    PositionalDistribution,
    PositionalSegment,
)


def build_positional_distribution(
    occurrences: list[EvidenceOccurrence],
    reference_length: int,
    segments: int = 3,
    *,
    offset_source: str | None = None,
    offset_unit: str = "characters",
) -> PositionalDistribution:
    """
    Distribute evidence occurrences across relative segments.

    Each occurrence is assigned using the midpoint between `start`
    and `end`.

    The function is independent of inspection, JSONL and rendering.
    """
    if segments < 1:
        raise ValueError(
            "segments must be greater than or equal to 1"
        )

    if reference_length < 0:
        raise ValueError(
            "reference_length must be greater than or equal to 0"
        )

    counts = [0] * segments

    if reference_length > 0:
        for occurrence in occurrences:
            _validate_occurrence(
                occurrence,
                reference_length=reference_length,
            )

            midpoint = (
                occurrence.start
                + occurrence.end
            ) / 2

            relative_position = (
                midpoint
                / reference_length
            )

            segment_index = min(
                int(relative_position * segments),
                segments - 1,
            )

            segment_index = max(
                segment_index,
                0,
            )

            counts[segment_index] += 1

    segment_results: list[PositionalSegment] = []
    total_occurrences = len(occurrences)

    for index, count in enumerate(counts):
        start_ratio = index / segments
        end_ratio = (index + 1) / segments

        start = round(
            start_ratio
            * reference_length
        )

        end = round(
            end_ratio
            * reference_length
        )

        share = (
            count / total_occurrences
            if total_occurrences > 0
            else None
        )

        segment_results.append(
            PositionalSegment(
                index=index,
                start=start,
                end=end,
                start_ratio=start_ratio,
                end_ratio=end_ratio,
                count=count,
                share=share,
            )
        )

    delta = _calculate_delta(
        segment_results,
        total_occurrences=total_occurrences,
    )

    return PositionalDistribution(
        reference_length=reference_length,
        total_occurrences=total_occurrences,
        segments=segment_results,
        delta=delta,
        offset_source=offset_source,
        offset_unit=offset_unit,
    )


def _validate_occurrence(
    occurrence: EvidenceOccurrence,
    *,
    reference_length: int,
) -> None:
    if occurrence.start < 0:
        raise ValueError(
            "Evidence occurrence start cannot be negative"
        )

    if occurrence.end < occurrence.start:
        raise ValueError(
            "Evidence occurrence end cannot be lower than start"
        )

    if occurrence.end > reference_length:
        raise ValueError(
            "Evidence occurrence exceeds the reference length: "
            f"{occurrence.start}:{occurrence.end} > "
            f"{reference_length}"
        )


def _calculate_delta(
    segments: list[PositionalSegment],
    *,
    total_occurrences: int,
) -> float | None:
    if total_occurrences == 0:
        return None

    first_share = (
        segments[0].share
        or 0.0
    )

    last_share = (
        segments[-1].share
        or 0.0
    )

    return last_share - first_share