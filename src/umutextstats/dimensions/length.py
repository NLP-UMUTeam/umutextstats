from __future__ import annotations

import pandas as pd

from umutextstats.config.params import param
from umutextstats.dimensions.mixins import TextComputeMixin
from umutextstats.dimensions.results import DimensionComputation
from umutextstats.inspection.scalar_inspectable_dimension import (
    ScalarInspectableDimension,
)
from umutextstats.text.tokenization import get_lexical_tokens


class LengthDimension(ScalarInspectableDimension):
    """
    Count the number of characters in the configured input column.
    """

    def compute_single(
        self,
        row: pd.Series,
    ) -> int:
        """
        Compute text length for a single row.
        """
        return len(self.get_text(row))

    def compute(
        self,
        df: pd.DataFrame,
    ) -> pd.Series:
        """
        Compute text length for all rows.

        If the DataFrame already contains `text_length`, reuse it.
        """
        if "text_length" in df.columns:
            return df["text_length"]

        return self.get_text_series(df).str.len()

    def compute_result(
        self,
        df: pd.DataFrame,
    ) -> DimensionComputation:
        """
        Compute text lengths with structured metadata.
        """
        values = self.compute(df)

        return DimensionComputation(
            values=values,
            numerators=values.copy(),
            metadata={
                "measure": "count",
                "unit": "characters",
            },
        )


class AverageWordLengthDimension(
    TextComputeMixin,
    ScalarInspectableDimension,
):
    """
    Compute the average length of lexical words in the configured text.
    """

    def _compute_text(
        self,
        text: str,
    ) -> float:
        """
        Compute average lexical-token length.
        """
        total_characters, total_words = (
            self._analyze_text(text)
        )

        if total_words == 0:
            return 0.0

        return (
            total_characters
            / total_words
        )

    def compute_result(
        self,
        df: pd.DataFrame,
    ) -> DimensionComputation:
        """
        Compute average word length and its components.
        """
        texts = self.get_text_series(df)

        analyses = texts.apply(
            self._analyze_text
        )

        numerators = analyses.apply(
            lambda analysis: analysis[0]
        )

        denominators = analyses.apply(
            lambda analysis: analysis[1]
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
                "measure": "mean",
                "numerator_unit": (
                    "lexical_token_characters"
                ),
                "normalization_unit": (
                    "lexical_tokens"
                ),
                "unit": (
                    "characters_per_lexical_token"
                ),
            },
        )

    @staticmethod
    def _analyze_text(
        text: str,
    ) -> tuple[int, int]:
        """
        Return total lexical-token characters and token count.
        """
        words = get_lexical_tokens(text)

        return (
            sum(
                len(word)
                for word in words
            ),
            len(words),
        )


class WordLengthDimension(
    TextComputeMixin,
    ScalarInspectableDimension,
):
    """
    Count or compute the percentage of words whose length matches a condition.

    Supported comparators are: >, >=, <, <=, =, ==.
    """

    def __init__(
        self,
        key: str,
        length: int,
        comparator: str = "=",
        input_column: str = "text_norm",
        percentage: bool = True,
    ):
        super().__init__(
            key=key,
            input_column=input_column,
        )

        self.length = int(length)
        self.comparator = comparator or "="
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
        percentage = str(
            param(
                dimension,
                "percentage",
                True,
            )
        ).lower() not in {
            "0",
            "false",
            "no",
        }

        return cls(
            key=dimension.key,
            length=int(
                param(
                    dimension,
                    "length",
                    0,
                )
            ),
            comparator=param(
                dimension,
                "comparator",
                "=",
            ),
            input_column=input_column,
            percentage=percentage,
        )

    def _compare(
        self,
        value: int,
    ) -> bool:
        """
        Compare a word length against the configured threshold.
        """
        if self.comparator == ">":
            return value > self.length

        if self.comparator == ">=":
            return value >= self.length

        if self.comparator == "<":
            return value < self.length

        if self.comparator == "<=":
            return value <= self.length

        if self.comparator in {
            "=",
            "==",
        }:
            return value == self.length

        return value == self.length

    def _compute_text(
        self,
        text: str,
    ) -> float:
        """
        Count or compute the percentage of words matching the length rule.
        """
        matching_words, total_words = (
            self._analyze_text(text)
        )

        if not self.percentage:
            return float(matching_words)

        if total_words == 0:
            return 0.0

        return (
            100.0
            * matching_words
            / total_words
        )

    def compute_result(
        self,
        df: pd.DataFrame,
    ) -> DimensionComputation:
        """
        Compute word-length values and their components.
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

        common_metadata = {
            "threshold": self.length,
            "comparator": self.comparator,
        }

        if not self.percentage:
            return DimensionComputation(
                values=numerators.astype(float),
                numerators=numerators,
                metadata={
                    **common_metadata,
                    "measure": "count",
                    "unit": (
                        "matching_lexical_tokens"
                    ),
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
                **common_metadata,
                "measure": "rate",
                "numerator_unit": (
                    "matching_lexical_tokens"
                ),
                "normalization_unit": (
                    "lexical_tokens"
                ),
                "scale": 100.0,
            },
        )

    def _analyze_text(
        self,
        text: str,
    ) -> tuple[int, int]:
        """
        Return matching and total lexical-token counts.
        """
        words = get_lexical_tokens(text)

        matching_words = sum(
            1
            for word in words
            if self._compare(
                len(word)
            )
        )

        return matching_words, len(words)