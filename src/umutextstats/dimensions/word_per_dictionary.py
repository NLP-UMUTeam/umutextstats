import regex as re
import numpy as np
import pandas as pd

from umutextstats.dictionaries import DictionaryLoader
from umutextstats.dimensions.base import BaseDimension
from umutextstats.text.tokenization import get_lexical_tokens


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

        dictionary_names = [
            name.strip()
            for name in dictionary_name.split("|")
            if name.strip()
        ]

        words = []
        exceptions = []

        for name in dictionary_names:
            entries = self.dictionary_loader.load(name)
            words.extend(entries.words)
            exceptions.extend(entries.exceptions)

        self.entries = words
        self.exceptions = exceptions

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
        source_words = get_lexical_tokens(text)
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

        if "word_count" in df.columns:
            total_words = df["word_count"]
        else:
            total_words = texts.apply(lambda text: len(get_lexical_tokens(text)))

        counts_array = counts.to_numpy(dtype=float)
        total_words_array = total_words.to_numpy(dtype=float)

        percentages = np.zeros_like(counts_array, dtype=float)

        np.divide(
            100.0 * counts_array,
            total_words_array,
            out=percentages,
            where=total_words_array != 0,
        )

        return pd.Series(percentages, index=counts.index)