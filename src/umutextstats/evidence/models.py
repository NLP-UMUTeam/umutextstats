from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EvidenceOccurrence:
    """
    A positional occurrence emitted by a dimension.

    Positions are expressed in the coordinate system indicated by
    `offset_source` and `offset_unit`.
    """

    start: int
    end: int
    label: str | None = None
    offset_source: str | None = None
    offset_unit: str = "characters"


@dataclass(frozen=True)
class PositionalSegment:
    """
    One relative segment of a positional distribution.
    """

    index: int
    start: int
    end: int
    start_ratio: float
    end_ratio: float
    count: int
    share: float | None


@dataclass(frozen=True)
class PositionalDistribution:
    """
    Distribution of evidence occurrences across relative segments.
    """

    reference_length: int
    total_occurrences: int
    segments: list[PositionalSegment]
    delta: float | None
    offset_source: str | None = None
    offset_unit: str = "characters"