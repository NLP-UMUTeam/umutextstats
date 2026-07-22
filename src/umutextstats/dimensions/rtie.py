from __future__ import annotations

import statistics

import pandas as pd

from umutextstats.config.params import param
from umutextstats.dimensions.mixins import TextComputeMixin
from umutextstats.dimensions.results import DimensionComputation
from umutextstats.inspection.scalar_inspectable_dimension import (
    ScalarInspectableDimension,
)
from umutextstats.text.patterns import SENTENCE_SPAN_REGEX
from umutextstats.text.tokenization import get_lexical_tokens


class RTIEBaseDimension(
    TextComputeMixin,
    ScalarInspectableDimension,
):
    """
    Base class for RTIE-style lexical diversity dimensions.

    The text can be evaluated as a whole, split by sentence, or split
    into fixed-size word chunks.
    """

    def __init__(
        self,
        key: str,
        input_column: str = "text_norm",
        separator: str = "by-chunks",
        chunk_size: int = 1000,
    ):
        super().__init__(
            key=key,
            input_column=input_column,
        )

        self.separator = separator or "by-chunks"
        self.chunk_size = int(
            chunk_size or 1000
        )

    @classmethod
    def from_config(
        cls,
        dimension,
        input_column: str = "text_norm",
    ):
        """
        Build the dimension from configuration.
        """
        return cls(
            key=dimension.key,
            input_column=input_column,
            separator=param(
                dimension,
                "separator",
                "by-chunks",
            ),
            chunk_size=int(
                param(
                    dimension,
                    "chunk_size",
                    1000,
                )
                or 1000
            ),
        )

    def _ratios(
        self,
        text: str,
    ) -> list[float]:
        """
        Compute type-token ratios according to the configured separator.
        """
        text = (
            ""
            if text is None
            else str(text)
        )

        if self.separator == "whole":
            return [
                self._ttr_whole(text)
            ]

        if self.separator == "by-sentence":
            return self._ttr_by_sentence(
                text
            )

        return self._ttr_by_chunks(
            text
        )

    def _ttr_whole(
        self,
        text: str,
    ) -> float:
        """
        Compute type-token ratio over the whole text.
        """
        words = self._words(text)

        if not words:
            return 0.0

        return (
            len(set(words))
            / len(words)
        )

    def _ttr_by_chunks(
        self,
        text: str,
    ) -> list[float]:
        """
        Compute type-token ratios over fixed-size word chunks.
        """
        words = self._words(text)

        if not words:
            return []

        ratios = []

        for start in range(
            0,
            len(words),
            self.chunk_size,
        ):
            chunk = words[
                start:
                start + self.chunk_size
            ]

            if not chunk:
                continue

            ratios.append(
                len(set(chunk))
                / len(chunk)
            )

        return ratios

    def _ttr_by_sentence(
        self,
        text: str,
    ) -> list[float]:
        """
        Compute type-token ratios sentence by sentence.
        """
        ratios = []

        for sentence in self._sentences(
            text
        ):
            words = self._words(
                sentence
            )

            if words:
                ratios.append(
                    len(set(words))
                    / len(words)
                )
            else:
                ratios.append(0.0)

        return ratios

    def _words(
        self,
        text: str,
    ) -> list[str]:
        """
        Extract lexical tokens from text.
        """
        return get_lexical_tokens(
            text
        )

    def _sentences(
        self,
        text: str,
    ) -> list[str]:
        """
        Extract sentence spans from text.
        """
        text = (
            ""
            if text is None
            else str(text)
        )

        return [
            match.group(0).strip()
            for match in SENTENCE_SPAN_REGEX.finditer(
                text
            )
            if match.group(0).strip()
        ]

    def _metadata(
        self,
    ) -> dict:
        """
        Return metadata shared by RTIE dimensions.
        """
        return {
            "separator": self.separator,
            "chunk_size": self.chunk_size,
            "segment_unit": self._segment_unit(),
            "ratio": "unique_tokens / lexical_tokens",
        }

    def _segment_unit(
        self,
    ) -> str:
        """
        Return the unit used to divide the text.
        """
        if self.separator == "whole":
            return "documents"

        if self.separator == "by-sentence":
            return "sentences"

        return "word_chunks"

    def inspection_debug_text(
        self,
    ) -> str:
        """
        Return configuration details used during inspection.
        """
        return (
            f"Separator: {self.separator}\n"
            f"Chunk size: {self.chunk_size}"
        )


class RTIEDimension(RTIEBaseDimension):
    """
    Compute the average type-token ratio.
    """

    def _compute_text(
        self,
        text: str,
    ) -> float:
        """
        Compute the average TTR across the configured segments.
        """
        ratios = self._ratios(text)

        if not ratios:
            return 0.0

        return (
            sum(ratios)
            / len(ratios)
        )

    def compute_result(
        self,
        df: pd.DataFrame,
    ) -> DimensionComputation:
        """
        Compute average TTR values and their aggregation components.
        """
        texts = self.get_text_series(df)

        ratios = texts.apply(
            self._ratios
        )

        numerators = ratios.apply(
            sum
        )

        denominators = ratios.apply(
            len
        )

        values = pd.Series(
            [
                (
                    numerator / denominator
                    if denominator
                    else 0.0
                )
                for numerator, denominator in zip(
                    numerators,
                    denominators,
                )
            ],
            index=df.index,
            dtype=float,
        )

        return DimensionComputation(
            values=values,
            numerators=numerators,
            denominators=denominators,
            metadata={
                **self._metadata(),
                "measure": (
                    "mean_type_token_ratio"
                ),
                "numerator_unit": (
                    "sum_of_segment_ttr"
                ),
                "normalization_unit": (
                    "evaluated_segments"
                ),
                "unit": "ratio",
            },
        )


class RTIEDeviationDimension(
    RTIEBaseDimension
):
    """
    Compute the population standard deviation of type-token ratios.
    """

    def _compute_text(
        self,
        text: str,
    ) -> float:
        """
        Compute the population standard deviation of segment TTRs.
        """
        ratios = self._ratios(text)

        if len(ratios) <= 1:
            return 0.0

        return float(
            statistics.pstdev(
                ratios
            )
        )

    def compute_result(
        self,
        df: pd.DataFrame,
    ) -> DimensionComputation:
        """
        Compute population deviations of segment TTR values.
        """
        texts = self.get_text_series(df)

        ratios = texts.apply(
            self._ratios
        )

        values = ratios.apply(
            self._deviation
        ).astype(float)

        return DimensionComputation(
            values=values,
            metadata={
                **self._metadata(),
                "measure": (
                    "population_standard_deviation"
                ),
                "observation_unit": (
                    "segment_type_token_ratios"
                ),
                "unit": "ratio",
            },
        )

    @staticmethod
    def _deviation(
        ratios: list[float],
    ) -> float:
        """
        Compute population standard deviation from segment ratios.
        """
        if len(ratios) <= 1:
            return 0.0

        return float(
            statistics.pstdev(
                ratios
            )
        )