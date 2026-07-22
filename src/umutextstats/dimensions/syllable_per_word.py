from __future__ import annotations

import pandas as pd

from umutextstats.dimensions.base import BaseDimension
from umutextstats.dimensions.results import DimensionComputation
from umutextstats.text.syllables import count_syllables_text
from umutextstats.text.tokenization import get_syllabifiable_words


class SyllablePerWordDimension(BaseDimension):
    """
    Compute the average number of syllables per syllabifiable word.
    """

    def compute(
        self,
        df: pd.DataFrame,
    ) -> pd.Series:
        """
        Compute syllables per word for every row.
        """
        return (
            df[self.input_column]
            .fillna("")
            .astype(str)
            .apply(self._compute_text)
        )

    def compute_result(
        self,
        df: pd.DataFrame,
    ) -> DimensionComputation:
        """
        Compute syllables-per-word values and their components.
        """
        texts = (
            df[self.input_column]
            .fillna("")
            .astype(str)
        )

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
                "numerator_unit": "syllables",
                "normalization_unit": (
                    "syllabifiable_words"
                ),
                "unit": (
                    "syllables_per_syllabifiable_word"
                ),
            },
        )

    def _compute_text(
        self,
        text: str,
    ) -> float:
        """
        Compute syllables per syllabifiable word for one text.
        """
        syllables, total_words = (
            self._analyze_text(text)
        )

        if total_words == 0:
            return 0.0

        return syllables / total_words

    @staticmethod
    def _analyze_text(
        text: str,
    ) -> tuple[int, int]:
        """
        Return syllable and syllabifiable-word counts.
        """
        text = (
            ""
            if text is None
            else str(text)
        )

        words = get_syllabifiable_words(
            text
        )

        if not words:
            return 0, 0

        syllables = count_syllables_text(
            text
        )

        return syllables, len(words)