import regex as re

from umutextstats.dimensions.base import BaseDimension

from umutextstats.text.patterns import SENTENCE_SPAN_REGEX
from umutextstats.text.patterns import INITIAL_TOKEN_REGEX


class ErrorStyleSentencesStartingWithNumbers(BaseDimension):
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

        occurrences = sum(
            1
            for sentence in sentences
            if self._starts_with_number(sentence)
        )

        return (100 * occurrences) / total_sentences

    def _split_sentences(self, text: str) -> list[str]:
        sentences = []

        for match in SENTENCE_SPAN_REGEX.finditer(text):
            sentence = match.group(0).strip()

            if not sentence:
                continue

            if not any(char.isalnum() for char in sentence):
                continue

            sentences.append(sentence)

        return sentences

    def _starts_with_number(self, sentence: str) -> bool:
        match = INITIAL_TOKEN_REGEX.search(sentence)

        if not match:
            return False

        token = match.group(0)

        return token[0].isdigit()