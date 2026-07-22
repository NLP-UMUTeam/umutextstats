from __future__ import annotations

import pandas as pd

from umutextstats.config.params import param
from umutextstats.dimensions.mixins import TextComputeMixin
from umutextstats.dimensions.results import DimensionComputation
from umutextstats.inspection.scalar_inspectable_dimension import (
    ScalarInspectableDimension,
)
from umutextstats.text.patterns import (
    LEADING_MENTION_REGEX,
    URL_REGEX,
    WORD_TOKEN_REGEX,
)


class WordCase(
    TextComputeMixin,
    ScalarInspectableDimension,
):
    """
    Compute the percentage of words matching a casing rule.

    Supported comparators:
    - "upper"
    - "lower"
    - "title"
    """

    def __init__(
        self,
        key: str,
        comparator: str = "upper",
        input_column: str = "text_raw",
    ):
        super().__init__(
            key=key,
            input_column=input_column,
        )

        self.comparator = comparator or "upper"

    @classmethod
    def from_config(
        cls,
        dimension,
        input_column: str = "text_raw",
    ):
        """
        Build the dimension from configuration.
        """
        return cls(
            key=dimension.key,
            comparator=(
                param(
                    dimension,
                    "word_comparator",
                )
                or param(
                    dimension,
                    "comparator",
                )
                or "upper"
            ),
            input_column=input_column,
        )

    def _compute_text(
        self,
        text: str,
    ) -> float:
        """
        Compute the percentage of words matching the configured casing rule.
        """
        matching_words, total_words = (
            self._analyze_text(text)
        )

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
        Compute word-case percentages and their components.
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
                    100.0
                    * numerator
                    / denominator
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
                "measure": "rate",
                "numerator_unit": (
                    "words_matching_case_rule"
                ),
                "normalization_unit": (
                    "eligible_word_tokens"
                ),
                "scale": 100.0,
                "comparator": self.comparator,
                "preprocessing": [
                    "remove_urls",
                    "remove_leading_mention",
                ],
            },
        )

    def _analyze_text(
        self,
        text: str,
    ) -> tuple[int, int]:
        """
        Return matching and eligible word counts.
        """
        text = (
            ""
            if text is None
            else str(text)
        )

        text = URL_REGEX.sub(
            "",
            text,
        )

        text = LEADING_MENTION_REGEX.sub(
            "",
            text,
        ).strip()

        words = WORD_TOKEN_REGEX.findall(
            text
        )

        if self.comparator == "title":
            words = [
                word
                for index, word in enumerate(words)
                if (
                    index == 0
                    or len(word) > 3
                )
            ]

        total_words = 0
        matching_words = 0

        for word in words:
            if word.isdigit():
                continue

            if word.startswith("@"):
                continue

            total_words += 1

            if self._fits(word):
                matching_words += 1

        return matching_words, total_words

    def _fits(
        self,
        word: str,
    ) -> bool:
        """
        Check whether a word matches the configured casing rule.
        """
        if self.comparator == "lower":
            return (
                word == word.lower()
            )

        if self.comparator == "title":
            return (
                word == word.title()
            )

        return (
            word == word.upper()
        )

    def inspection_debug_text(
        self,
    ) -> str:
        """
        Return configuration details used during inspection.
        """
        return (
            f"Comparator: {self.comparator}"
        )