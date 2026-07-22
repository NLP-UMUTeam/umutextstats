from __future__ import annotations

import pandas as pd

from umutextstats.dimensions.mixins import TextComputeMixin
from umutextstats.dimensions.results import DimensionComputation
from umutextstats.inspection.scalar_inspectable_dimension import (
    ScalarInspectableDimension,
)
from umutextstats.text.patterns import SENTENCE_SPAN_REGEX
from umutextstats.text.syllables import count_syllables_text
from umutextstats.text.tokenization import get_lexical_tokens


class ReadabilityDimension(
    TextComputeMixin,
    ScalarInspectableDimension,
):
    """
    Compute the readability score for the configured input text.

    Formula:

        206.84
        - 60 * (syllables / words)
        - 102 * (sentences / words)
    """

    def _compute_text(
        self,
        text: str,
    ) -> float:
        """
        Compute readability from plain text.
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
        Compute readability scores with formula metadata.
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
                "measure": "readability_score",
                "formula": (
                    "206.84 "
                    "- 60 * (syllables / words) "
                    "- 102 * (sentences / words)"
                ),
                "components": [
                    "lexical_tokens",
                    "syllables",
                    "sentences",
                ],
                "scale": "unbounded",
            },
        )

    def _analyze_text(
        self,
        text: str,
    ) -> tuple[int, int, int]:
        """
        Return word, syllable, and sentence counts.
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

        sentence_count = self._count_sentences(
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
        Compute the readability score from its components.
        """
        if (
            word_count == 0
            or sentence_count == 0
        ):
            return 0.0

        return float(
            206.84
            - (
                60
                * (
                    syllable_count
                    / word_count
                )
            )
            - (
                102
                * (
                    sentence_count
                    / word_count
                )
            )
        )

    def _count_sentences(
        self,
        text: str,
    ) -> int:
        """
        Count sentence spans.

        Empty text returns 0. Non-empty text without sentence spans
        returns 1.
        """
        text = (
            ""
            if text is None
            else str(text)
        ).strip()

        if not text:
            return 0

        count = len(
            SENTENCE_SPAN_REGEX.findall(
                text
            )
        )

        if count == 0:
            return 1

        return count

    def inspection_debug_text(
        self,
    ) -> str:
        """
        Return the formula used by this dimension.
        """
        return (
            "Formula: "
            "206.84 - "
            "60 * (syllables / words) - "
            "102 * (sentences / words)"
        )