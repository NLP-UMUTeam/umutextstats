# src/umutextstats/dimensions/rtie.py

import statistics
import regex as re

from umutextstats.dimensions.base import BaseDimension
from umutextstats.text.tokenization import get_lexical_tokens
from umutextstats.text.patterns import SENTENCE_SPAN_REGEX


class RTIEBaseDimension(BaseDimension):
    def __init__(
        self,
        key: str,
        input_column: str = "text_norm",
        separator: str = "by-chunks",
        chunk_size: int = 1000,
    ):
        super().__init__(key=key, input_column=input_column)
        self.separator = separator or "by-chunks"
        self.chunk_size = chunk_size

    def _ratios(self, text: str) -> list[float]:
        if self.separator == "whole":
            return [self._ttr_whole(text)]

        if self.separator == "by-sentence":
            return self._ttr_by_sentence(text)

        return self._ttr_by_chunks(text)

    def _ttr_whole(self, text: str) -> float:
        words = self._words(text)
        return len(set(words)) / len(words) if words else 0.0

    def _ttr_by_chunks(self, text: str) -> list[float]:
        words = self._words(text)

        if not words:
            return []

        ratios = []

        for start in range(0, len(words), self.chunk_size):
            chunk = words[start:start + self.chunk_size]
            if chunk:
                ratios.append(len(set(chunk)) / len(chunk))

        return ratios

    def _ttr_by_sentence(self, text: str) -> list[float]:
        ratios = []

        for sentence in self._sentences(text):
            words = self._words(sentence)
            ratios.append(len(set(words)) / len(words) if words else 0.0)

        return ratios

    def _words(self, text: str) -> list[str]:
        return get_lexical_tokens (text)

    def _sentences(self, text: str) -> list[str]:
        return [
            match.group(0).strip()
            for match in SENTENCE_SPAN_REGEX.finditer(text)
            if match.group(0).strip()
        ]


class RTIEDimension(RTIEBaseDimension):
    def compute(self, df):
        return (
            df[self.input_column]
            .fillna("")
            .astype(str)
            .apply(self._compute_text)
        )

    def _compute_text(self, text: str) -> float:
        ratios = self._ratios(text)
        return sum(ratios) / len(ratios) if ratios else 0.0


class RTIEDeviationDimension(RTIEBaseDimension):
    def compute(self, df):
        return (
            df[self.input_column]
            .fillna("")
            .astype(str)
            .apply(self._compute_text)
        )

    def _compute_text(self, text: str) -> float:
        ratios = self._ratios(text)

        if len(ratios) <= 1:
            return 0.0

        return statistics.pstdev(ratios)