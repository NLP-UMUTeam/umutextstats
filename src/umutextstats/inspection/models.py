from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class InspectMatch:
    match: str
    start: int
    end: int


@dataclass(frozen=True)
class DimensionInspection:
    key: str
    class_name: str | None
    pattern: str | None
    dictionary: str | None
    matches: list[InspectMatch]
    discarded_matches: list[InspectMatch] | None = None
    debug_text: str | None = None
    internal_representation: str | None = None
    description: str | None = None


@dataclass(frozen=True)
class InspectSegment:
    index: int
    start: int
    end: int
    start_ratio: float
    end_ratio: float
    count: int
    share: float | None


@dataclass(frozen=True)
class InspectDistribution:
    text_length: int
    total_matches: int
    segments: list[InspectSegment]
    delta: float | None