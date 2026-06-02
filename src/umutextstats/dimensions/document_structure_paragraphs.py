from statistics import pstdev

from umutextstats.dimensions.base import BaseDimension
from umutextstats.text.paragraph import split_paragraphs, paragraph_lengths


class ParagraphCountDimension(BaseDimension):
    """Count non-empty paragraphs separated by one or more blank lines."""

    def compute(self, df):

        if "paragraph_count" in df.columns:
            return df["paragraph_count"]

        text = df[self.input_column].fillna("").astype(str)

        return text.apply(lambda value: len(split_paragraphs(value)))


class AverageParagraphLengthDimension(BaseDimension):
    """Compute the average number of words per paragraph."""

    def compute(self, df):
        if "paragraph_length_avg" in df.columns:
            return df["paragraph_length_avg"]

        text = df[self.input_column].fillna("").astype(str)

        return text.apply(self._average_paragraph_length)

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

    @staticmethod
    def _paragraph_length_std(text: str) -> float:
        lengths = paragraph_lengths(text)

        if len(lengths) < 2:
            return 0.0

        return pstdev(lengths)