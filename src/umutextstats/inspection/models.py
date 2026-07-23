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
    offset_source: str | None = None
    offset_unit: str = "characters"