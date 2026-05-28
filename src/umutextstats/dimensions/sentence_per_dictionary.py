# src/umutextstats/dimensions/sentence_per_dictionary.py

import regex as re

from umutextstats.dictionaries import DictionaryLoader
from umutextstats.dimensions.base import BaseDimension
from umutextstats.text.patterns import SENTENCE_SPAN_REGEX


class SentencePerDictionary(BaseDimension):
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

        entries = self.dictionary_loader.load(dictionary_name)

        self.entries = entries.words
        self.exceptions = entries.exceptions

        self.patterns = self._compile_patterns(self.entries, kind="word")

    def _compile_patterns(self, entries: list[str], kind: str):
        patterns = []

        for line_number, entry in enumerate(entries, start=1):
            pattern = rf"(?<!\p{{L}}){entry}(?!\p{{L}})"

            try:
                patterns.append(re.compile(pattern, re.IGNORECASE))
            except re.error as exc:
                raise ValueError(
                    f"Invalid regex in dictionary '{self.dictionary_name}' "
                    f"({kind}) at line {line_number}: {entry!r}. "
                    f"Compiled pattern: {pattern!r}. "
                    f"Regex error: {exc}"
                ) from exc

        return patterns

    def compute(self, df):
        return (
            df[self.input_column]
            .fillna("")
            .astype(str)
            .apply(self._compute_text)
        )

    def _compute_text(self, text: str) -> float:
        sentences = self._split_sentences(text)
        total_sentences = len(sentences)

        if total_sentences == 0:
            return 0.0

        occurrences = 0

        for sentence in sentences:
            for pattern in self.patterns:
                occurrences += len(pattern.findall(sentence))

        if not self.percentage:
            return occurrences

        return (100 * occurrences) / total_sentences

    def _split_sentences(self, text: str) -> list[str]:
        return [
            match.group(0).strip()
            for match in SENTENCE_SPAN_REGEX.finditer(text)
            if match.group(0).strip()
        ]