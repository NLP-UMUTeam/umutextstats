import regex as re
import numpy as np
import pandas as pd

from umutextstats.dictionaries import DictionaryLoader
from umutextstats.dimensions.base import BaseDimension
from umutextstats.text.tokenization import get_lexical_tokens
from umutextstats.text.patterns import POS_ITEM_REGEX


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
            word_totals  = df["word_count"]
        else:
            word_totals  = texts.apply(lambda text: len(get_lexical_tokens(text)))

        counts_array = counts.to_numpy(dtype=float)
        word_totals_array = word_totals .to_numpy(dtype=float)

        percentages = np.zeros_like(counts_array, dtype=float)

        np.divide(
            100.0 * counts_array,
            word_totals_array,
            out=percentages,
            where=word_totals_array != 0,
        )

        return pd.Series(percentages, index=counts.index)
        
        
    def _count_text_with_pos(
        self,
        text: str,
        tagged_pos: str,
    ) -> int:
        if not text or not tagged_pos:
            return 0

        allowed_words = [
            item["word"].lower()
            for item in self._parse_tagged_pos(tagged_pos)
            if item["tag"] in self.pos_tag
        ]

        if not allowed_words:
            return 0

        if self.use_regex:
            positive_count = 0
            exception_count = 0

            positive_words = allowed_words.copy()
            exception_words = allowed_words.copy()

            for pattern in self.patterns:
                for match in pattern.finditer(text):
                    matched_text = match.group(0).lower()

                    if matched_text in positive_words:
                        positive_count += 1
                        positive_words.remove(matched_text)

            for pattern in self.exception_patterns:
                for match in pattern.finditer(text):
                    matched_text = match.group(0).lower()

                    if matched_text in exception_words:
                        exception_count += 1
                        exception_words.remove(matched_text)

            return max(0, positive_count - exception_count)

        source_words = get_lexical_tokens(text)
        available_words = allowed_words.copy()
        count = 0

        for word in source_words:
            if word not in available_words:
                continue

            available_words.remove(word)

            if word in self.words:
                count += 1

            if word in self.exception_words:
                count -= 1

        return max(0, count)
        
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