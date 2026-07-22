from dataclasses import dataclass
from statistics import pstdev

import pandas as pd

from umutextstats.dimensions.results import DimensionComputation
from umutextstats.inspection.iterable_inspectable_dimension import IterableInspectableDimension
from umutextstats.text.paragraph import (
    iter_paragraph_spans,
    paragraph_lengths,
    split_paragraphs,
)
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


@dataclass(frozen=True)
class ParagraphEvidence:
    text: str
    start: int
    end: int
    word_count: int


def analyze_paragraphs(
    text: str,
) -> list[ParagraphEvidence]:
    """
    Return non-empty paragraphs with offsets and word counts.
    """
    lengths = paragraph_lengths(text)
    spans = list(iter_paragraph_spans(text))

    return [
        ParagraphEvidence(
            text=paragraph,
            start=start,
            end=end,
            word_count=length,
        )
        for (
            paragraph,
            start,
            end,
        ), length in zip(
            spans,
            lengths,
        )
    ]


def paragraph_to_evidence(
    paragraph: ParagraphEvidence,
) -> dict:
    return {
        "text": paragraph.text,
        "start": paragraph.start,
        "end": paragraph.end,
        "word_count": paragraph.word_count,
    }

class ParagraphCountDimension(IterableInspectableDimension):
    """Count non-empty paragraphs separated by one or more blank lines."""

    def compute_single(
        self,
        row: pd.Series,
    ) -> int:
        return len(
            analyze_paragraphs(
                self.get_text(row)
            )
        )

    def compute(
        self,
        df: pd.DataFrame,
    ) -> pd.Series:
        if "paragraph_count" in df.columns:
            return df["paragraph_count"]

        return self.get_text_series(df).apply(
            lambda value: len(split_paragraphs(value))
        )

    def iter_matches(self, text: str):
        for paragraph, start, end in iter_paragraph_spans(text):
            yield SimpleMatch(paragraph, start, end)

    def compute_result(
        self,
        df: pd.DataFrame,
    ) -> DimensionComputation:
        texts = self.get_text_series(df)

        analyses = texts.apply(
            analyze_paragraphs
        )

        counts = analyses.apply(len)

        evidence = analyses.apply(
            lambda paragraphs: [
                paragraph_to_evidence(paragraph)
                for paragraph in paragraphs
            ]
        )

        if "paragraph_count" in df.columns:
            values = df["paragraph_count"].copy()
        else:
            values = counts

        return DimensionComputation(
            values=values,
            numerators=counts,
            evidence=evidence,
            metadata={
                "measure": "count",
                "unit": "paragraphs",
                "evidence_offset_unit": "text_characters",
            },
        )


class AverageParagraphLengthDimension(IterableInspectableDimension):
    """Compute the average number of words per paragraph."""

    def compute_single(
        self,
        row: pd.Series,
    ) -> float:
        return self._average_paragraph_length(self.get_text(row))

    def compute(
        self,
        df: pd.DataFrame,
    ) -> pd.Series:
        if "paragraph_length_avg" in df.columns:
            return df["paragraph_length_avg"]

        return self.get_text_series(df).apply(
            self._average_paragraph_length
        )

    def iter_matches(self, text: str):
        for paragraph, start, end in iter_paragraph_spans(text):
            yield SimpleMatch(paragraph, start, end)

    @staticmethod
    def _average_paragraph_length(text: str) -> float:
        lengths = paragraph_lengths(text)

        if not lengths:
            return 0.0

        return sum(lengths) / len(lengths)

    def compute_result(
        self,
        df: pd.DataFrame,
    ) -> DimensionComputation:
        texts = self.get_text_series(df)

        analyses = texts.apply(
            analyze_paragraphs
        )

        numerators = analyses.apply(
            lambda paragraphs: sum(
                paragraph.word_count
                for paragraph in paragraphs
            )
        )

        denominators = analyses.apply(len)

        computed_values = pd.Series(
            [
                (
                    numerator / denominator
                    if denominator
                    else 0.0
                )
                for numerator, denominator in zip(
                    numerators,
                    denominators,
                )
            ],
            index=df.index,
            dtype=float,
        )

        if "paragraph_length_avg" in df.columns:
            values = df[
                "paragraph_length_avg"
            ].copy()
        else:
            values = computed_values

        evidence = analyses.apply(
            lambda paragraphs: [
                paragraph_to_evidence(paragraph)
                for paragraph in paragraphs
            ]
        )

        return DimensionComputation(
            values=values,
            numerators=numerators,
            denominators=denominators,
            evidence=evidence,
            metadata={
                "measure": "mean",
                "unit": "words_per_paragraph",
                "numerator_unit": "words",
                "normalization_unit": "paragraphs",
                "evidence_offset_unit": "text_characters",
            },
        )

class ParagraphLengthDeviationDimension(IterableInspectableDimension):
    """Compute the population standard deviation of paragraph lengths."""

    def compute_single(
        self,
        row: pd.Series,
    ) -> float:
        return self._paragraph_length_std(self.get_text(row))

    def compute(
        self,
        df: pd.DataFrame,
    ) -> pd.Series:
        if "paragraph_length_std" in df.columns:
            return df["paragraph_length_std"]

        return self.get_text_series(df).apply(
            self._paragraph_length_std
        )

    def iter_matches(self, text: str):
        for paragraph, start, end in iter_paragraph_spans(text):
            yield SimpleMatch(paragraph, start, end)

    def compute_result(
        self,
        df: pd.DataFrame,
    ) -> DimensionComputation:
        texts = self.get_text_series(df)

        analyses = texts.apply(
            analyze_paragraphs
        )

        computed_values = analyses.apply(
            lambda paragraphs: (
                pstdev(
                    [
                        paragraph.word_count
                        for paragraph in paragraphs
                    ]
                )
                if len(paragraphs) >= 2
                else 0.0
            )
        ).astype(float)

        if "paragraph_length_std" in df.columns:
            values = df[
                "paragraph_length_std"
            ].copy()
        else:
            values = computed_values

        evidence = analyses.apply(
            lambda paragraphs: [
                paragraph_to_evidence(paragraph)
                for paragraph in paragraphs
            ]
        )

        return DimensionComputation(
            values=values,
            evidence=evidence,
            metadata={
                "measure": "population_standard_deviation",
                "unit": "words_per_paragraph",
                "evidence_offset_unit": "text_characters",
            },
        )

    @staticmethod
    def _paragraph_length_std(text: str) -> float:
        lengths = paragraph_lengths(text)

        if len(lengths) < 2:
            return 0.0

        return pstdev(lengths)


class DialogueParagraphPercentageDimension(IterableInspectableDimension):
    """
    Percentage of paragraphs that begin with a dialogue dash.

    Examples:
        —Hola.
        - Hola.
        – Hola.
    """

    def compute_single(
        self,
        row: pd.Series,
    ) -> float:
        return self._compute_text(self.get_text(row))

    def compute(
        self,
        df: pd.DataFrame,
    ) -> pd.Series:
        return self.get_text_series(df).apply(self._compute_text)

    def iter_matches(
        self,
        text: str,
    ):
        _, dialogue_paragraphs = (
            self._analyze_text(text)
        )

        for paragraph in dialogue_paragraphs:
            yield SimpleMatch(
                paragraph.text,
                paragraph.start,
                paragraph.end,
            )

    def _compute_text(
        self,
        text: str,
    ) -> float:
        paragraphs, dialogue_paragraphs = (
            self._analyze_text(text)
        )

        if not paragraphs:
            return 0.0

        return (
            100.0
            * len(dialogue_paragraphs)
            / len(paragraphs)
        )
    
    def _analyze_text(
        self,
        text: str,
    ) -> tuple[
        list[ParagraphEvidence],
        list[ParagraphEvidence],
    ]:
        paragraphs = analyze_paragraphs(text)

        dialogue_paragraphs = [
            paragraph
            for paragraph in paragraphs
            if DIALOGUE_PARAGRAPH_REGEX.match(
                paragraph.text
            )
        ]

        return paragraphs, dialogue_paragraphs
    
    def compute_result(
        self,
        df: pd.DataFrame,
    ) -> DimensionComputation:
        texts = self.get_text_series(df)

        analyses = texts.apply(
            self._analyze_text
        )

        numerators = analyses.apply(
            lambda analysis: len(analysis[1])
        )

        denominators = analyses.apply(
            lambda analysis: len(analysis[0])
        )

        values = pd.Series(
            [
                (
                    100.0 * numerator / denominator
                    if denominator
                    else 0.0
                )
                for numerator, denominator in zip(
                    numerators,
                    denominators,
                )
            ],
            index=df.index,
            dtype=float,
        )

        evidence = analyses.apply(
            lambda analysis: [
                paragraph_to_evidence(paragraph)
                for paragraph in analysis[1]
            ]
        )

        return DimensionComputation(
            values=values,
            numerators=numerators,
            denominators=denominators,
            evidence=evidence,
            metadata={
                "measure": "rate",
                "normalization_unit": "paragraphs",
                "numerator_unit": "dialogue_paragraphs",
                "scale": 100.0,
                "evidence_offset_unit": "text_characters",
            },
        )