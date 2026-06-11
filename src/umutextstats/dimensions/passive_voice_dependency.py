from __future__ import annotations

from dataclasses import dataclass

from umutextstats.dimensions.mixins import TextComputeMixin

import pandas as pd

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
class DependencyMatch:
    """
    Regex-like match object used by the inspection layer.
    """

    text: str
    start_pos: int
    end_pos: int

    def group(self, index: int = 0) -> str:
        if index != 0:
            raise IndexError(index)

        return self.text

    def start(self) -> int:
        return self.start_pos

    def end(self) -> int:
        return self.end_pos


class PassiveVoiceDependencyDimension(TextComputeMixin, IterableInspectableDimension):
    """
    Compute the percentage of dependency-tagged sentences that contain
    passive voice dependency labels.
    """

    def __init__(
        self,
        key: str,
        input_column: str = "tagged_dep",
    ):
        super().__init__(key=key, input_column=input_column)

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

    def _compute_text(
        self,
        tagged_text: str,
    ) -> float:
        """
        Compute the percentage of sentences containing passive dependency labels.
        """
        sentences = self._split_sentences(tagged_text)

        if not sentences:
            return 0.0

        passive_sentences = sum(
            1
            for sentence in sentences
            if self._has_passive_voice(
                self._parse_sentence(sentence)
            )
        )

        return (100 * passive_sentences) / len(sentences)

    def _has_passive_voice(
        self,
        items: list[dict],
    ) -> bool:
        """
        Return True if any dependency item is marked as passive.
        """
        return any(
            item["deprel"] in PASSIVE_DEPRELS
            for item in items
        )

    def _split_sentences(
        self,
        tagged_text: str,
    ) -> list[str]:
        """
        Split dependency-tagged text into sentence chunks.
        """
        if not tagged_text:
            return []

        return [
            sentence.strip()
            for sentence in tagged_text.split(" || ")
            if sentence.strip()
        ]

    def _parse_sentence(
        self,
        sentence: str,
    ) -> list[dict]:
        """
        Parse a dependency-tagged sentence into dependency items.
        """
        items = []

        for raw_item in sentence.split(", "):
            match = DEPENDENCY_ITEM_REGEX.fullmatch(raw_item.strip())

            if not match:
                continue

            items.append(
                {
                    "word": match.group("word") or "",
                    "deprel": match.group("deprel") or "",
                }
            )

        return items

    def iter_matches(
        self,
        tagged_text: str,
    ):
        """
        Yield passive dependency matches for inspection.
        """
        tagged_text = "" if tagged_text is None else str(tagged_text)

        search_start = 0

        for sentence in self._split_sentences(tagged_text):
            sentence_start = tagged_text.find(sentence, search_start)

            if sentence_start == -1:
                sentence_start = 0

            search_start = sentence_start + len(sentence)

            for match in DEPENDENCY_ITEM_REGEX.finditer(sentence):
                word = match.group("word") or ""
                deprel = match.group("deprel") or ""

                if deprel not in PASSIVE_DEPRELS:
                    continue

                start = sentence_start + match.start()
                end = sentence_start + match.end()

                yield DependencyMatch(
                    text=word,
                    start_pos=start,
                    end_pos=end,
                )

    def inspection_debug_text(self) -> str:
        """
        Return the dependency labels considered passive.
        """
        return (
            "Passive dependency labels: "
            "aux:pass, nsubj:pass, expl:pass"
        )