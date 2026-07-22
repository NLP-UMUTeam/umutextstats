import pandas as pd

from umutextstats.dimensions.mixins import TextComputeMixin
from umutextstats.dimensions.results import DimensionComputation
from umutextstats.inspection.iterable_inspectable_dimension import (
    IterableInspectableDimension,
)
from umutextstats.text.sentence import count_sentences
from umutextstats.text.tokenization import get_lexical_tokens


class WordPerSentenceDimension(
    TextComputeMixin,
    IterableInspectableDimension,
):
    """
    Compute the average number of lexical words per sentence.
    """

    def _analyze_text(
        self,
        text: str,
    ) -> tuple[int, int]:
        """
        Return lexical-word and sentence counts.
        """
        total_words = len(
            get_lexical_tokens(text)
        )

        total_sentences = count_sentences(text)

        return total_words, total_sentences

    def _compute_text(
        self,
        text: str,
    ) -> float:
        """
        Compute the average number of lexical words per sentence.
        """
        total_words, total_sentences = (
            self._analyze_text(text)
        )

        if total_words == 0:
            return 0.0

        if total_sentences == 0:
            return 0.0

        return total_words / total_sentences

    def compute_result(
        self,
        df: pd.DataFrame,
    ) -> DimensionComputation:
        """
        Compute words-per-sentence values and their components.
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
                    if numerator > 0
                    and denominator > 0
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
                "numerator_unit": "lexical_tokens",
                "normalization_unit": "sentences",
                "unit": "words_per_sentence",
            },
        )

    def iter_matches(
        self,
        text: str,
    ):
        """
        This aggregate dimension has no individual positive matches.
        """
        return []