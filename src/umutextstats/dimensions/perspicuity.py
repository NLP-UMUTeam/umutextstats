from __future__ import annotations

import pandas as pd

from umutextstats.dimensions.mixins import TextComputeMixin
from umutextstats.dimensions.results import DimensionComputation
from umutextstats.inspection.scalar_inspectable_dimension import (
    ScalarInspectableDimension,
)
from umutextstats.text.sentence import count_sentences
from umutextstats.text.syllables import count_syllables_text
from umutextstats.text.tokenization import get_lexical_tokens


class PerspicuityDimension(
    TextComputeMixin,
    ScalarInspectableDimension,
):
    """
    Compute the Fernández-Huerta perspicuity score.

    Higher values indicate easier texts, while lower values indicate
    more complex texts.

    Formula:

        206.835
        - 62.3 * (syllables / words)
        - (words / sentences)
    """

    def _compute_text(
        self,
        text: str,
    ) -> float:
        """
        Compute the perspicuity score from raw text.
        """
        word_count, syllable_count, sentence_count = (
            self._analyze_text(text)
        )

        return self._compute_score(
            word_count=word_count,
            syllable_count=syllable_count,
            sentence_count=sentence_count,
        )

    def compute_result(
        self,
        df: pd.DataFrame,
    ) -> DimensionComputation:
        """
        Compute perspicuity scores with formula metadata.
        """
        texts = self.get_text_series(df)

        analyses = texts.apply(
            self._analyze_text
        )

        values = analyses.apply(
            lambda analysis: self._compute_score(
                word_count=analysis[0],
                syllable_count=analysis[1],
                sentence_count=analysis[2],
            )
        ).astype(float)

        return DimensionComputation(
            values=values,
            metadata={
                "measure": "perspicuity_score",
                "formula": (
                    "206.835 "
                    "- 62.3 * (syllables / words) "
                    "- (words / sentences)"
                ),
                "components": [
                    "lexical_tokens",
                    "syllables",
                    "sentences",
                ],
                "scale": "unbounded",
            },
        )

    @staticmethod
    def _analyze_text(
        text: str,
    ) -> tuple[int, int, int]:
        """
        Return lexical-word, syllable, and sentence counts.
        """
        text = (
            ""
            if text is None
            else str(text)
        )

        word_count = len(
            get_lexical_tokens(text)
        )

        syllable_count = count_syllables_text(
            text
        )

        sentence_count = count_sentences(
            text
        )

        return (
            word_count,
            syllable_count,
            sentence_count,
        )

    @staticmethod
    def _compute_score(
        word_count: int,
        syllable_count: int,
        sentence_count: int,
    ) -> float:
        """
        Compute the Fernández-Huerta score from its components.
        """
        if (
            word_count == 0
            or sentence_count == 0
        ):
            return 0.0

        return float(
            206.835
            - (
                62.3
                * (
                    syllable_count
                    / word_count
                )
            )
            - (
                word_count
                / sentence_count
            )
        )

    def inspection_debug_text(
        self,
    ) -> str:
        """
        Return the formula used by this dimension.
        """
        return (
            "Formula: 206.835 - "
            "62.3 * (syllables / words) - "
            "(words / sentences)"
        )