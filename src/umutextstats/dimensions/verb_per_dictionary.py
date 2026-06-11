import pandas as pd

from umutextstats.dimensions.mixins import TextComputeMixin
from umutextstats.config.params import dictionary_param, percentage_param
from umutextstats.dictionaries import DictionaryLoader
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


class VerbPerDictionary(TextComputeMixin, ScalarInspectableDimension):
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
        super().__init__(key=key, input_column=input_column)

        self.dictionary_name = dictionary_name
        self.percentage = percentage
        self.dictionary_loader = dictionary_loader or DictionaryLoader()

        dictionary_names = [
            name.strip()
            for name in dictionary_name.split("|")
            if name.strip()
        ]

        words = []

        for name in dictionary_names:
            entries = self.dictionary_loader.load(name)
            words.extend(entries.words)

        # Store dictionary entries as lowercase strings once.
        # Runtime checks then become O(1) set lookups.
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
            dictionary_name=dictionary_param(dimension),
            input_column=input_column,
            percentage=percentage_param(dimension),
        )

    def _compute_text(
        self,
        text: str,
    ) -> float:
        """
        Count matching verbs in a text and optionally normalize as percentage.
        """
        words = get_lexical_tokens(text)
        total_words = len(words)

        if total_words == 0:
            return 0.0

        occurrences = 0
        aux_verb = ""

        for word in words:
            word = word.lower()
            candidate = f"{aux_verb}{word}"

            if candidate in self.words:
                occurrences += 1

            if word in AUX_VERB_HABER:
                aux_verb = f"{word} "
            else:
                aux_verb = ""

        if not self.percentage:
            return occurrences

        return (100 * occurrences) / total_words

    def inspection_debug_text(self) -> str:
        """
        Return configuration details used during inspection.
        """
        return (
            f"Loaded dictionary: {self.dictionary_name}\n"
            f"Dictionary entries: {len(self.words)}\n"
            "Matches simple verbs and haber + participle/periphrastic entries"
        )