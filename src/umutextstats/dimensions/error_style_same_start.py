import regex as re
from collections import Counter

from umutextstats.dimensions.base import BaseDimension
from umutextstats.text.patterns import SENTENCE_SPAN_REGEX
from umutextstats.text.patterns import INITIAL_TOKEN_EXCLUSING_NUMBERS_REGEX


class ErrorStyleSentencesStartingWithTheSameWord(BaseDimension):
    def __init__(
        self,
        key: str,
        input_column: str = "text_raw",
    ):
        super().__init__(key=key, input_column=input_column)

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

        first_words = [
            first_word
            for sentence in sentences
            if (first_word := self._first_word(sentence)) is not None
        ]

        stats = Counter(first_words)

        occurrences = sum(
            count
            for count in stats.values()
            if count > 1
        )

        return (100 * occurrences) / total_sentences

    def _split_sentences(self, text: str) -> list[str]:
        sentences = []

        for match in SENTENCE_SPAN_REGEX.finditer(text):
            sentence = match.group(0).strip()

            if not sentence:
                continue

            if not any(char.isalpha() for char in sentence):
                continue

            sentences.append(sentence)

        return sentences

    def _first_word(self, sentence: str) -> str | None:
        match = INITIAL_TOKEN_EXCLUSING_NUMBERS_REGEX.search(sentence)

        if not match:
            return None

        return match.group(0).lower()