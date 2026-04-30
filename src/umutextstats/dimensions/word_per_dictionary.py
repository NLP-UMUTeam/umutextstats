# src/umutextstats/dimensions/word_per_dictionary.py

import regex as re

from umutextstats.dictionaries import DictionaryLoader
from umutextstats.dimensions.base import BaseDimension
from umutextstats.dimensions.word_count import WORD_REGEX


class WordPerDictionary(BaseDimension):
    def __init__(
        self,
        key: str,
        dictionary_name: str,
        input_column: str = "text_norm",
        percentage: bool = True,
        use_regex: bool = True,
        dictionary_loader: DictionaryLoader | None = None,
    ):
        super().__init__(key=key, input_column=input_column)

        self.dictionary_name = dictionary_name
        self.percentage = percentage
        self.use_regex = use_regex
        self.dictionary_loader = dictionary_loader or DictionaryLoader()

        entries = self.dictionary_loader.load(dictionary_name)

        self.entries = entries.words
        self.exceptions = entries.exceptions

        if self.use_regex:
            self.patterns = self._compile_patterns(self.entries, kind="word")
            self.exception_patterns = self._compile_patterns(
                self.exceptions,
                kind="exception",
            )
            self.words = None
            self.exception_words = None
        else:
            self.patterns = None
            self.exception_patterns = None
            self.words = set(self.entries)
            self.exception_words = set(self.exceptions)

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

    def _count_regex_patterns(self, text: str, patterns) -> int:
        return sum(len(pattern.findall(text)) for pattern in patterns)

    def _count_plain_words(self, text: str, words: set[str]) -> int:
        source_words = WORD_REGEX.findall(text.lower())
        return sum(1 for word in source_words if word in words)

    def _count_text(self, text: str) -> int:
        if not text:
            return 0

        if self.use_regex:
            count = self._count_regex_patterns(text, self.patterns)
            count -= self._count_regex_patterns(text, self.exception_patterns)
        else:
            count = self._count_plain_words(text, self.words)
            count -= self._count_plain_words(text, self.exception_words)

        return max(0, count)

    def compute(self, df): 
        texts = df[self.input_column].fillna("").astype(str)

        counts = texts.apply(self._count_text)

        if not self.percentage:
            return counts

        total_words = texts.apply(lambda text: len(WORD_REGEX.findall(text)))

        return counts.where(
            total_words == 0,
            (100 * counts / total_words),
        ).fillna(0)