from __future__ import annotations

import regex as re

from umutextstats.dimensions.base import BaseDimension
from umutextstats.text.patterns import DEPENDENCY_ITEM_REGEX



class PassiveVoiceDependencyDimension(BaseDimension):
    def __init__(
        self,
        key: str,
        input_column: str = "tagged_dep",
    ):
        super().__init__(key=key, input_column=input_column)

    def compute(self, df):
        return (
            df[self.input_column]
            .fillna("")
            .astype(str)
            .apply(self._compute_text)
        )

    def _compute_text(self, tagged_text: str) -> float:
        sentences = self._split_sentences(tagged_text)

        if not sentences:
            return 0.0

        passive_sentences = 0

        for sentence in sentences:
            items = self._parse_sentence(sentence)

            if self._has_passive_voice(items):
                passive_sentences += 1

        return (100 * passive_sentences) / len(sentences)

    def _has_passive_voice(self, items: list[dict]) -> bool:
        for item in items:
            deprel = item["deprel"]

            if (
                deprel == "aux:pass"
                or deprel == "nsubj:pass"
                or deprel == "expl:pass"
            ):
                return True

        return False

    def _split_sentences(self, tagged_text: str) -> list[str]:
        if not tagged_text:
            return []

        return [
            sentence.strip()
            for sentence in tagged_text.split(" || ")
            if sentence.strip()
        ]

    def _parse_sentence(self, sentence: str) -> list[dict]:
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