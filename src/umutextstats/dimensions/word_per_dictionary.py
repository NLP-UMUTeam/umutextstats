from collections import Counter

import numpy as np
import pandas as pd
import regex as re

from umutextstats.dictionaries import DictionaryLoader
from umutextstats.dimensions.base import BaseDimension
from umutextstats.text.patterns import LEXICAL_TOKEN_REGEX, POS_ITEM_REGEX
from umutextstats.text.tokenization import get_lexical_tokens


class WordPerDictionary(BaseDimension):
    def __init__(
        self,
        key: str,
        dictionary_name: str,
        input_column: str = "text_norm",
        pos_tag: str | list[str] | None = None,
        pos_input_column: str | None = "tagged_pos",
        percentage: bool = True,
        use_regex: bool = True,
        dictionary_loader: DictionaryLoader | None = None,
    ):
        super().__init__(key=key, input_column=input_column)

        self.dictionary_name = dictionary_name
        self.percentage = percentage
        self.use_regex = use_regex
        self.pos_input_column = pos_input_column
        self.dictionary_loader = dictionary_loader or DictionaryLoader()

        if isinstance(pos_tag, str):
            pos_tag = [pos_tag]

        self.pos_tag = pos_tag or None

        dictionary_names = [
            name.strip()
            for name in dictionary_name.split("|")
            if name.strip()
        ]

        entries = []
        exceptions = []

        for name in dictionary_names:
            dictionary_entries = self.dictionary_loader.load(name)
            entries.extend(dictionary_entries.words)
            exceptions.extend(dictionary_entries.exceptions)


        self.entries = entries
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
            self.words = {word.lower() for word in self.entries}
            self.exception_words = {word.lower() for word in self.exceptions}

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
        return sum(1 for word in source_words if word.lower() in words)

    def _count_text(self, text: str) -> int:
        if not text:
            return 0

        if self.use_regex:
            positive_count = self._count_regex_patterns(text, self.patterns)
            exception_count = self._count_regex_patterns(
                text,
                self.exception_patterns,
            )
        else:
            positive_count = self._count_plain_words(text, self.words)
            exception_count = self._count_plain_words(
                text,
                self.exception_words,
            )

        return max(positive_count - exception_count, 0)

    def compute(self, df):
        texts = df[self.input_column].fillna("").astype(str)

        if self.pos_tag:
            tagged_texts = df[self.pos_input_column].fillna("").astype(str)
            counts = [
                self._count_text_with_pos(text, tagged_pos)
                for text, tagged_pos in zip(texts, tagged_texts)
            ]
            counts = pd.Series(counts, index=df.index)
        else:
            counts = texts.apply(self._count_text)

        if not self.percentage:
            return counts

        if "word_count" in df.columns:
            word_totals = df["word_count"]
        else:
            word_totals = texts.apply(lambda text: len(get_lexical_tokens(text)))

        counts_array = counts.to_numpy(dtype=float)
        word_totals_array = word_totals.to_numpy(dtype=float)

        percentages = np.zeros_like(counts_array, dtype=float)

        np.divide(
            100.0 * counts_array,
            word_totals_array,
            out=percentages,
            where=word_totals_array != 0,
        )

        return pd.Series(percentages, index=counts.index)

    def _count_text_with_pos(self, text: str, tagged_pos: str) -> int:
        if not text or not tagged_pos:
            return 0

        allowed_words = [
            item["word"].lower()
            for item in self._parse_tagged_pos(tagged_pos)
            if item["tag"] in self.pos_tag
        ]

        if not allowed_words:
            return 0

        available = Counter(allowed_words)

        if self.use_regex:
            positive_count = self._count_regex_pos_matches(
                text,
                self.patterns,
                available.copy(),
            )
            exception_count = self._count_regex_pos_matches(
                text,
                self.exception_patterns,
                available.copy(),
            )
        else:
            positive_count = self._count_plain_pos_matches(
                text,
                self.words,
                available.copy(),
            )
            exception_count = self._count_plain_pos_matches(
                text,
                self.exception_words,
                available.copy(),
            )

        return max(positive_count - exception_count, 0)

    def _count_regex_pos_matches(self, text: str, patterns, available: Counter) -> int:
        count = 0

        for pattern in patterns:
            for match in pattern.finditer(text):
                matched = match.group(0).lower()

                if available[matched] <= 0:
                    continue

                available[matched] -= 1
                count += 1

        return count

    def _count_plain_pos_matches(
        self,
        text: str,
        words: set[str],
        available: Counter,
    ) -> int:
        count = 0

        for word in get_lexical_tokens(text):
            word = word.lower()

            if available[word] <= 0:
                continue

            available[word] -= 1

            if word in words:
                count += 1

        return count

    def _parse_tagged_pos(self, tagged_pos: str) -> list[dict[str, str]]:
        if not tagged_pos:
            return []

        items = []

        for sentence in tagged_pos.split(" || "):
            for raw_item in sentence.split(", "):
                match = POS_ITEM_REGEX.fullmatch(raw_item.strip())

                if not match:
                    continue

                items.append(
                    {
                        "word": match.group("word") or "",
                        "tag": match.group("tag") or "",
                        "feats": match.group("feats") or "",
                    }
                )

        return items

    def iter_positive_matches(self, text: str):
        text = "" if text is None else str(text)

        if self.use_regex:
            for pattern in self.patterns:
                yield from pattern.finditer(text)
            return

        for match in LEXICAL_TOKEN_REGEX.finditer(text):
            if match.group(0).lower() in self.words:
                yield match

    def iter_exception_matches(self, text: str):
        text = "" if text is None else str(text)

        if self.use_regex:
            for pattern in self.exception_patterns:
                yield from pattern.finditer(text)
            return

        for match in LEXICAL_TOKEN_REGEX.finditer(text):
            if match.group(0).lower() in self.exception_words:
                yield match

    def iter_matches(self, text: str):
        yield from self.iter_positive_matches(text)

    def iter_discarded_matches(self, text: str):
        yield from self.iter_exception_matches(text)

    def iter_positive_matches_with_pos(self, text: str, tagged_pos: str):
        if not self.pos_tag:
            yield from self.iter_positive_matches(text)
            return

        allowed = self._allowed_pos_counter(tagged_pos)

        for match in self.iter_positive_matches(text):
            word = match.group(0).lower()

            if allowed[word] <= 0:
                continue

            allowed[word] -= 1
            yield match


    def iter_exception_matches_with_pos(self, text: str, tagged_pos: str):
        if not self.pos_tag:
            yield from self.iter_exception_matches(text)
            return

        allowed = self._allowed_pos_counter(tagged_pos)

        for match in self.iter_exception_matches(text):
            word = match.group(0).lower()

            if allowed[word] <= 0:
                continue

            allowed[word] -= 1
            yield match


    def iter_matches_with_context(self, text: str, *, tagged_pos: str | None = None):
        if self.pos_tag:
            yield from self.iter_positive_matches_with_pos(text, tagged_pos or "")
            return

        yield from self.iter_positive_matches(text)


    def iter_discarded_matches_with_context(
        self,
        text: str,
        *,
        tagged_pos: str | None = None,
    ):
        if self.pos_tag:
            yield from self.iter_exception_matches_with_pos(text, tagged_pos or "")
            return

        yield from self.iter_exception_matches(text)


    def _allowed_pos_counter(self, tagged_pos: str):
        from collections import Counter

        return Counter(
            item["word"].lower()
            for item in self._parse_tagged_pos(tagged_pos)
            if item["tag"] in self.pos_tag
        )