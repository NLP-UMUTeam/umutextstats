import regex as re

from umutextstats.dimensions.base import BaseDimension
from umutextstats.text.patterns import URL_REGEX, LEADING_MENTION_REGEX, WORD_TOKEN_REGEX


class WordCase(BaseDimension):
    def __init__(
        self,
        key: str,
        comparator: str = "upper",
        input_column: str = "text_raw",
    ):
        super().__init__(key=key, input_column=input_column)
        self.comparator = comparator or "upper"

    def compute(self, df):
        return (
            df[self.input_column]
            .fillna("")
            .astype(str)
            .apply(self._compute_text)
        )

    def _compute_text(self, text: str) -> float:
        text = URL_REGEX.sub("", text)
        text = LEADING_MENTION_REGEX.sub("", text).strip()

        words = WORD_TOKEN_REGEX.findall(text)

        if self.comparator == "title":
            words = [
                word
                for index, word in enumerate(words)
                if index == 0 or len(word) > 3
            ]

        total_words = 0
        fit_words = 0

        for word in words:
            if word.isdigit():
                continue

            if word.startswith("@"):
                continue

            total_words += 1

            if self._fits(word):
                fit_words += 1

        if total_words == 0:
            return 0.0

        return (100 * fit_words) / total_words

    def _fits(self, word: str) -> bool:
        if self.comparator == "lower":
            return word == word.lower()

        if self.comparator == "title":
            return word == word.title()

        # default: upper
        return word == word.upper()