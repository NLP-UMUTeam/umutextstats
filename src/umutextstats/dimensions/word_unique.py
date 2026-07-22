from __future__ import annotations

import pandas as pd

from umutextstats.config.params import percentage_param
from umutextstats.dimensions.mixins import TextComputeMixin
from umutextstats.dimensions.results import DimensionComputation
from umutextstats.inspection.scalar_inspectable_dimension import (
    ScalarInspectableDimension,
)
from umutextstats.text.tokenization import get_lexical_tokens


class WordUniqueDimension(
    TextComputeMixin,
    ScalarInspectableDimension,
):
    """
    Count unique lexical words or compute their percentage over total words.
    """

    def __init__(
        self,
        key: str,
        input_column: str = "text_norm",
        percentage: bool = True,
    ):
        super().__init__(
            key=key,
            input_column=input_column,
        )

        self.percentage = percentage

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
            percentage=percentage_param(
                dimension
            ),
        )

    def _compute_text(
        self,
        text: str,
    ) -> float:
        """
        Count unique lexical words or normalize them by total word count.
        """
        unique_words, total_words = (
            self._analyze_text(text)
        )

        if not self.percentage:
            return float(unique_words)

        if total_words == 0:
            return 0.0

        return (
            100.0
            * unique_words
            / total_words
        )

    def compute_result(
        self,
        df: pd.DataFrame,
    ) -> DimensionComputation:
        """
        Compute unique-word values and their components.
        """
        texts = self.get_text_series(df)

        analyses = texts.apply(
            self._analyze_text
        )

        numerators = analyses.apply(
            lambda analysis: analysis[0]
        )

        total_words = analyses.apply(
            lambda analysis: analysis[1]
        )

        if not self.percentage:
            return DimensionComputation(
                values=numerators.astype(float),
                numerators=numerators,
                metadata={
                    "measure": "count",
                    "unit": "unique_lexical_tokens",
                },
            )

        values = pd.Series(
            [
                (
                    100.0
                    * numerator
                    / denominator
                    if denominator
                    else 0.0
                )
                for numerator, denominator in zip(
                    numerators,
                    total_words,
                )
            ],
            index=df.index,
            dtype=float,
        )

        return DimensionComputation(
            values=values,
            numerators=numerators,
            denominators=total_words,
            metadata={
                "measure": "rate",
                "numerator_unit": (
                    "unique_lexical_tokens"
                ),
                "normalization_unit": (
                    "lexical_tokens"
                ),
                "scale": 100.0,
            },
        )

    @staticmethod
    def _analyze_text(
        text: str,
    ) -> tuple[int, int]:
        """
        Return unique and total lexical-token counts.
        """
        words = get_lexical_tokens(text)

        return (
            len(set(words)),
            len(words),
        )

    def inspection_debug_text(
        self,
    ) -> str:
        return (
            f"Percentage: {self.percentage}"
        )