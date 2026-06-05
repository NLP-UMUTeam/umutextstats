from statistics import pstdev
from dataclasses import dataclass

from umutextstats.dimensions.base import BaseDimension
from umutextstats.text.paragraph import split_paragraphs, paragraph_lengths, iter_paragraph_spans
from umutextstats.text.patterns import DIALOGUE_PARAGRAPH_REGEX


@dataclass(frozen=True)
class SimpleMatch:
    text: str
    start_pos: int
    end_pos: int

    def group(self, index=0):
        return self.text

    def start(self):
        return self.start_pos

    def end(self):
        return self.end_pos



class ParagraphCountDimension(BaseDimension):
    """Count non-empty paragraphs separated by one or more blank lines."""

    def compute(self, df):

        if "paragraph_count" in df.columns:
            return df["paragraph_count"]

        text = df[self.input_column].fillna("").astype(str)

        return text.apply(lambda value: len(split_paragraphs(value)))

    def iter_matches(self, text: str):
        for paragraph, start, end in iter_paragraph_spans(text):
            yield SimpleMatch(paragraph, start, end)

class AverageParagraphLengthDimension(BaseDimension):
    """Compute the average number of words per paragraph."""

    def compute(self, df):
        if "paragraph_length_avg" in df.columns:
            return df["paragraph_length_avg"]

        text = df[self.input_column].fillna("").astype(str)

        return text.apply(self._average_paragraph_length)

    def iter_matches(self, text: str):
        for paragraph, start, end in iter_paragraph_spans(text):
            yield SimpleMatch(paragraph, start, end)

    @staticmethod
    def _average_paragraph_length(text: str) -> float:
        lengths = paragraph_lengths(text)

        if not lengths:
            return 0.0

        return sum(lengths) / len(lengths)


class ParagraphLengthDeviationDimension(BaseDimension):
    """Compute the population standard deviation of paragraph lengths."""

    def compute(self, df):
        if "paragraph_length_std" in df.columns:
            return df["paragraph_length_std"]

        text = df[self.input_column].fillna("").astype(str)

        return text.apply(self._paragraph_length_std)

    def iter_matches(self, text: str):
        for paragraph, start, end in iter_paragraph_spans(text):
            yield SimpleMatch(paragraph, start, end)

    @staticmethod
    def _paragraph_length_std(text: str) -> float:
        lengths = paragraph_lengths(text)

        if len(lengths) < 2:
            return 0.0

        return pstdev(lengths)
    

class DialogueParagraphPercentageDimension(BaseDimension):
    """
    Percentage of paragraphs that begin with a dialogue dash.

    Examples:
        —Hola.
        - Hola.
        – Hola.
    """

    def compute(self, df):
        return (
            df[self.input_column]
            .fillna("")
            .astype(str)
            .apply(self._compute_text)
        )

    def iter_matches(self, text: str):
        for paragraph, start, end in iter_paragraph_spans(text):
            if DIALOGUE_PARAGRAPH_REGEX.match(paragraph):
                yield SimpleMatch(paragraph, start, end)

    @staticmethod
    def _compute_text(text: str) -> float:
        paragraphs = split_paragraphs(text)

        if not paragraphs:
            return 0.0

        dialogue_count = sum(
            1
            for paragraph in paragraphs
            if DIALOGUE_PARAGRAPH_REGEX.match(paragraph)
        )

        return 100.0 * dialogue_count / len(paragraphs)

