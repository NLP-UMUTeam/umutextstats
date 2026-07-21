from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from umutextstats.dimensions.mixins import TextComputeMixin
from umutextstats.dimensions.results import DimensionComputation
from umutextstats.inspection.iterable_inspectable_dimension import (
    IterableInspectableDimension,
)
from umutextstats.text.patterns import DEPENDENCY_ITEM_REGEX


PASSIVE_DEPRELS = {
    "aux:pass",
    "nsubj:pass",
    "expl:pass",
}


@dataclass(frozen=True)
class DependencyItem:
    """
    Parsed dependency item with offsets in the serialized dependency input.
    """

    word: str
    deprel: str
    start: int
    end: int


@dataclass(frozen=True)
class DependencySentence:
    """
    Dependency-tagged sentence with absolute serialized-text offsets.
    """

    text: str
    start: int
    end: int
    items: tuple[DependencyItem, ...]


@dataclass(frozen=True)
class PassiveSentence:
    """
    Sentence containing at least one passive dependency relation.
    """

    sentence: DependencySentence
    triggers: tuple[DependencyItem, ...]


@dataclass(frozen=True)
class DependencyMatch:
    """
    Regex-like sentence match used by the inspection layer.
    """

    text: str
    start_pos: int
    end_pos: int

    def group(
        self,
        index: int = 0,
    ) -> str:
        if index != 0:
            raise IndexError(index)

        return self.text

    def start(self) -> int:
        return self.start_pos

    def end(self) -> int:
        return self.end_pos


class PassiveVoiceDependencyDimension(
    TextComputeMixin,
    IterableInspectableDimension,
):
    """
    Compute the percentage of dependency-tagged sentences containing
    passive voice dependency labels.

    Evidence offsets refer to the serialized dependency input.
    """

    def __init__(
        self,
        key: str,
        input_column: str = "tagged_dep",
    ):
        super().__init__(
            key=key,
            input_column=input_column,
        )

    @classmethod
    def from_config(
        cls,
        dimension,
        input_column: str = "tagged_dep",
    ):
        """
        Build the dimension from configuration.
        """
        return cls(
            key=dimension.key,
            input_column=input_column,
        )

    def _analyze_text(
        self,
        tagged_text: str,
    ) -> tuple[
        list[DependencySentence],
        list[PassiveSentence],
    ]:
        """
        Parse dependency sentences and detect passive sentences.

        This is the shared source of truth for computation,
        extraction, and inspection.
        """
        tagged_text = (
            ""
            if tagged_text is None
            else str(tagged_text)
        )

        sentences = self._parse_sentences(
            tagged_text
        )

        passive_sentences = []

        for sentence in sentences:
            triggers = tuple(
                item
                for item in sentence.items
                if item.deprel in PASSIVE_DEPRELS
            )

            if triggers:
                passive_sentences.append(
                    PassiveSentence(
                        sentence=sentence,
                        triggers=triggers,
                    )
                )

        return sentences, passive_sentences

    def _compute_text(
        self,
        tagged_text: str,
    ) -> float:
        """
        Compute the percentage of passive sentences.
        """
        sentences, passive_sentences = (
            self._analyze_text(tagged_text)
        )

        if not sentences:
            return 0.0

        return (
            100.0
            * len(passive_sentences)
            / len(sentences)
        )

    def compute_result(
        self,
        df: pd.DataFrame,
    ) -> DimensionComputation:
        """
        Compute values, numerators, denominators, and evidence.
        """
        tagged_texts = self.get_text_series(df)

        analyses = tagged_texts.apply(
            self._analyze_text
        )

        numerators = analyses.apply(
            lambda analysis: len(analysis[1])
        )

        denominators = analyses.apply(
            lambda analysis: len(analysis[0])
        )

        evidence = analyses.apply(
            lambda analysis: self._to_evidence(
                analysis[1]
            )
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
        )

        return DimensionComputation(
            values=values,
            numerators=numerators,
            denominators=denominators,
            evidence=evidence,
            metadata={
                "measure": "rate",
                "normalization_unit": "dependency_sentences",
                "scale": 100.0,
                "evidence_offset_unit": (
                    "serialized_dependency_characters"
                ),
            },
        )

    def iter_matches(
        self,
        tagged_text: str,
    ):
        """
        Yield one match for each passive sentence.
        """
        _, passive_sentences = self._analyze_text(
            tagged_text
        )

        for passive in passive_sentences:
            sentence = passive.sentence

            yield DependencyMatch(
                text=sentence.text,
                start_pos=sentence.start,
                end_pos=sentence.end,
            )

    def _parse_sentences(
        self,
        tagged_text: str,
    ) -> list[DependencySentence]:
        """
        Parse sentences with absolute offsets in the serialized input.
        """
        if not tagged_text:
            return []

        sentences = []
        cursor = 0
        separator = " || "

        for raw_sentence in tagged_text.split(separator):
            raw_start = cursor
            cursor += len(raw_sentence) + len(separator)

            sentence_text = raw_sentence.strip()

            if not sentence_text:
                continue

            leading_spaces = (
                len(raw_sentence)
                - len(raw_sentence.lstrip())
            )

            sentence_start = (
                raw_start + leading_spaces
            )
            sentence_end = (
                sentence_start + len(sentence_text)
            )

            items = self._parse_sentence_items(
                sentence_text=sentence_text,
                sentence_start=sentence_start,
            )

            sentences.append(
                DependencySentence(
                    text=sentence_text,
                    start=sentence_start,
                    end=sentence_end,
                    items=tuple(items),
                )
            )

        return sentences

    def _parse_sentence_items(
        self,
        sentence_text: str,
        sentence_start: int,
    ) -> list[DependencyItem]:
        """
        Parse dependency items with absolute serialized-text offsets.
        """
        items = []
        cursor = 0

        for raw_item in sentence_text.split(", "):
            raw_start = sentence_text.find(
                raw_item,
                cursor,
            )

            if raw_start < 0:
                continue

            cursor = raw_start + len(raw_item)

            match = DEPENDENCY_ITEM_REGEX.fullmatch(
                raw_item.strip()
            )

            if not match:
                continue

            word = match.group("word") or ""
            deprel = match.group("deprel") or ""

            leading_spaces = (
                len(raw_item)
                - len(raw_item.lstrip())
            )

            local_start, local_end = match.span("word")

            items.append(
                DependencyItem(
                    word=word,
                    deprel=deprel,
                    start=(
                        sentence_start
                        + raw_start
                        + leading_spaces
                        + local_start
                    ),
                    end=(
                        sentence_start
                        + raw_start
                        + leading_spaces
                        + local_end
                    ),
                )
            )

        return items


    @staticmethod
    def _to_evidence(
        passive_sentences: list[PassiveSentence],
    ) -> list[dict]:
        """
        Convert passive sentences into serializable evidence.
        """
        return [
            {
                "text": passive.sentence.text,
                "start": passive.sentence.start,
                "end": passive.sentence.end,
                "triggers": [
                    {
                        "text": trigger.word,
                        "deprel": trigger.deprel,
                        "start": trigger.start,
                        "end": trigger.end,
                    }
                    for trigger in passive.triggers
                ],
            }
            for passive in passive_sentences
        ]

    def inspection_debug_text(self) -> str:
        """
        Return the dependency labels considered passive.
        """
        return (
            "Passive dependency labels: "
            "aux:pass, nsubj:pass, expl:pass"
        )