# src/umutextstats/extraction/models.py

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass
class BatchDimensionResult:
    """
    Result of computing one dimension for all input documents.

    `values` contains the global dimension value.

    `segments` contains values calculated for document regions, such as
    start, middle, and end.

    `derived` contains values calculated from the global or segmented
    results, such as delta, variability, or curvature.
    """

    key: str
    values: pd.Series
    kind: str = "atomic"
    numerators: pd.Series | None = None
    denominators: pd.Series | None = None
    evidence: pd.Series | None = None
    segments: dict[str, pd.Series] = field(default_factory=dict)
    derived: dict[str, pd.Series] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.values = self._ensure_series(self.values)

        if self.numerators is not None:
            self.numerators = self._ensure_series(
                self.numerators
            )

        if self.denominators is not None:
            self.denominators = self._ensure_series(
                self.denominators
            )

        if self.evidence is not None:
            self.evidence = self._ensure_series(
                self.evidence
            )

        self.segments = {
            key: self._ensure_series(values)
            for key, values in self.segments.items()
        }

        self.derived = {
            key: self._ensure_series(values)
            for key, values in self.derived.items()
        }

    @staticmethod
    def _ensure_series(
        values,
    ) -> pd.Series:
        if isinstance(values, pd.Series):
            return values

        return pd.Series(values)


@dataclass
class ExtractionResult:
    """
    Structured result produced by the extraction engine.
    """

    dimensions: dict[str, BatchDimensionResult]
    ids: pd.Series | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dataframe(
        self,
        include_segments: bool = False,
        include_derived: bool = False,
    ) -> pd.DataFrame:
        """
        Project dimension results into a wide feature matrix.

        Global values are always included. Segments and derived values are
        optional so the default remains compatible with the original
        UMUTextStats matrix.
        """
        data = {}

        if self.ids is not None:
            data["id"] = self.ids.reset_index(drop=True)

        for key, result in self.dimensions.items():
            data[key] = result.values.reset_index(drop=True)

            if include_segments:
                for segment, values in result.segments.items():
                    data[f"{key}__{segment}"] = values.reset_index(
                        drop=True
                    )

            if include_derived:
                for statistic, values in result.derived.items():
                    data[f"{key}__{statistic}"] = values.reset_index(
                        drop=True
                    )

        return pd.DataFrame(data)