import regex as re

from umutextstats.dimensions.base import BaseDimension
from umutextstats.text.patterns import SENTENCE_SPAN_REGEX, MENTION_REGEX


class ErrorCapitalizationStartingWithLowerCaseDimension(BaseDimension):
    START_SYMBOLS = {"¿", "¡", "[", '"', "'", "-", "—", "_"}

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

        errors = sum(1 for sentence in sentences if self._starts_with_lowercase(sentence))

        return (100 * errors) / total_sentences

    def _split_sentences(self, text: str) -> list[str]:
        sentences = []

        for match in SENTENCE_SPAN_REGEX.finditer(text):
            sentence = match.group(0).strip()

            if not sentence:
                continue

            # Evita contar fragmentos como solo comillas, símbolos, etc.
            if not any(char.isalpha() for char in sentence):
                continue

            sentences.append(sentence)

        return sentences

    def _starts_with_lowercase(self, sentence: str) -> bool:
        sentence = MENTION_REGEX.sub("", sentence).strip()

        if not sentence:
            return False

        for char in sentence:
            if char in self.START_SYMBOLS or char.isspace():
                continue

            if not char.isalpha():
                return False

            return char == char.lower() and char != char.upper()

        return False