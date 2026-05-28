# src/umutextstats/dimensions/twitter_reply_to.py

import unicodedata
import regex as re

from umutextstats.dictionaries import DictionaryLoader
from umutextstats.dimensions.base import BaseDimension
from umutextstats.text.patterns import FIRST_TOKEN_REGEX


def remove_accents(text: str) -> str:
    text = unicodedata.normalize("NFD", text)
    return "".join(ch for ch in text if unicodedata.category(ch) != "Mn")


class TwitterReplyToDimension(BaseDimension):
    def __init__(
        self,
        key: str,
        dictionary_name: str,
        input_column: str = "text_raw",
        dictionary_loader: DictionaryLoader | None = None,
    ):
        super().__init__(key=key, input_column=input_column)

        self.dictionary_name = dictionary_name
        self.dictionary_loader = dictionary_loader or DictionaryLoader()

        entries = self.dictionary_loader.load(dictionary_name)

        self.words = [
            remove_accents(word.lower())
            for word in entries.words
        ]

    def compute(self, df):
        return (
            df[self.input_column]
            .fillna("")
            .astype(str)
            .apply(self._compute_text)
        )

    def _compute_text(self, text: str) -> int:
        first_token = self._first_token(text)

        if not first_token:
            return 0

        first_token = remove_accents(first_token.lower())

        if not first_token.startswith("@"):
            return 0

        return int(any(word in first_token for word in self.words))

    def _first_token(self, text: str) -> str | None:
        match = FIRST_TOKEN_REGEX.search(text.strip())

        if not match:
            return None

        return match.group(0)