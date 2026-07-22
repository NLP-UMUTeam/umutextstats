from __future__ import annotations

import pandas as pd

from umutextstats.config.params import (
    dictionary_param,
    percentage_param,
)
from umutextstats.dictionaries import DictionaryLoader
from umutextstats.dimensions.mixins import TextComputeMixin
from umutextstats.dimensions.results import DimensionComputation
from umutextstats.inspection.scalar_inspectable_dimension import (
    ScalarInspectableDimension,
)
from umutextstats.text.tokenization import get_lexical_tokens


AUX_VERB_HABER = {
    "ha",
    "habe",
    "habed",
    "haber",
    "habido",
    "habiendo",
    "habremos",
    "habrá",
    "habrán",
    "habrás",
    "habré",
    "habréis",
    "habría",
    "habríais",
    "habríamos",
    "habrían",
    "habrías",
    "habéis",
    "había",
    "habíais",
    "habíamos",
    "habían",
    "habías",
    "han",
    "has",
    "hay",
    "haya",
    "hayamos",
    "hayan",
    "hayas",
    "hayáis",
    "he",
    "hemos",
    "hube",
    "hubiera",
    "hubierais",
    "hubieran",
    "hubieras",
    "hubiere",
    "hubiereis",
    "hubieren",
    "hubieres",
    "hubieron",
    "hubiese",
    "hubieseis",
    "hubiesen",
    "hubieses",
    "hubimos",
    "hubiste",
    "hubisteis",
    "hubiéramos",
    "hubiéremos",
    "hubiésemos",
    "hubo",
}


class VerbPerDictionary(
    TextComputeMixin,
    ScalarInspectableDimension,
):
    """
    Count dictionary verb matches or compute their percentage over words.

    The dictionary can include simple verb entries and compound entries
    such as "haber participle". During computation, the previous auxiliary
    verb "haber" form is combined with the current word to detect those
    compound entries.
    """

    def __init__(
        self,
        key: str,
        dictionary_name: str,
        input_column: str = "text_norm",
        percentage: bool = True,
        dictionary_loader: DictionaryLoader | None = None,
    ):
        super().__init__(
            key=key,
            input_column=input_column,
        )

        self.dictionary_name = dictionary_name
        self.percentage = percentage

        self.dictionary_loader = (
            dictionary_loader
            or DictionaryLoader()
        )

        dictionary_names = [
            name.strip()
            for name in dictionary_name.split("|")
            if name.strip()
        ]

        words = []

        for name in dictionary_names:
            entries = self.dictionary_loader.load(name)
            words.extend(entries.words)

        self.words = {
            entry.lower()
            for entry in words
        }

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
            dictionary_name=dictionary_param(
                dimension
            ),
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
        Count matching verbs and optionally normalize by word count.
        """
        occurrences, total_words = (
            self._analyze_text(text)
        )

        if not self.percentage:
            return float(occurrences)

        if total_words == 0:
            return 0.0

        return (
            100.0
            * occurrences
            / total_words
        )

    def compute_result(
        self,
        df: pd.DataFrame,
    ) -> DimensionComputation:
        """
        Compute dictionary-verb values and their components.
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
            "dictionary": self.dictionary_name,
            "dictionary_entries": len(self.words),
            "matching_method": (
                "simple_or_haber_compound"
            ),
        }

        if not self.percentage:
            return DimensionComputation(
                values=numerators.astype(float),
                numerators=numerators,
                metadata={
                    **common_metadata,
                    "measure": "count",
                    "unit": (
                        "matching_dictionary_verbs"
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
                    "matching_dictionary_verbs"
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
        Return matching verb occurrences and total lexical-token count.

        Simple entries are matched directly. Compound entries are matched
        by combining a preceding form of `haber` with the current token.
        """
        words = get_lexical_tokens(text)
        total_words = len(words)

        occurrences = 0
        auxiliary = ""

        for raw_word in words:
            word = raw_word.lower()

            candidate = (
                f"{auxiliary}{word}"
            )

            if candidate in self.words:
                occurrences += 1

            if word in AUX_VERB_HABER:
                auxiliary = f"{word} "
            else:
                auxiliary = ""

        return occurrences, total_words

    def inspection_debug_text(
        self,
    ) -> str:
        """
        Return configuration details used during inspection.
        """
        return (
            f"Loaded dictionary: {self.dictionary_name}\n"
            f"Dictionary entries: {len(self.words)}\n"
            "Matches simple verbs and haber + participle/"
            "periphrastic entries"
        )