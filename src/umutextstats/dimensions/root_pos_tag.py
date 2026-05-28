from __future__ import annotations

import regex as re

from umutextstats.dimensions.base import BaseDimension
from umutextstats.text.patterns import POS_ITEM_REGEX
from umutextstats.text.patterns import DEPENDENCY_ITEM_REGEX


class RootPOSTagDimension(BaseDimension):
    def __init__(
        self,
        key: str,
        input_column: str = "tagged_pos",
        tagged_dep_column: str = "tagged_dep",
        tag: str | None = None,
    ):
        super().__init__(key=key, input_column=input_column)
        self.tagged_dep_column = tagged_dep_column
        self.tag = tag

    def compute(self, df):
        return df.apply(self._compute_row, axis=1)

    def _compute_row(self, row) -> float:
        tagged_pos = str(row.get(self.input_column, "") or "")
        tagged_dep = str(row.get(self.tagged_dep_column, "") or "")

        pos_sentences = self._split_sentences(tagged_pos)
        dep_sentences = self._split_sentences(tagged_dep)

        total_roots = 0
        matches = 0

        for pos_sentence, dep_sentence in zip(pos_sentences, dep_sentences):
            pos_items = self._parse_pos_sentence(pos_sentence)
            dep_items = self._parse_dep_sentence(dep_sentence)

            if not pos_items or not dep_items:
                continue

            root_indices = [
                index
                for index, item in enumerate(dep_items)
                if item["head"] == 0
            ]

            for root_index in root_indices:
                if root_index >= len(pos_items):
                    continue

                total_roots += 1

                if self.tag and pos_items[root_index]["tag"] == self.tag:
                    matches += 1

        if total_roots == 0:
            return 0.0

        return (100 * matches) / total_roots

    def _split_sentences(self, tagged_text: str) -> list[str]:
        if not tagged_text:
            return []

        return [
            sentence.strip()
            for sentence in tagged_text.split(" || ")
            if sentence.strip()
        ]

    def _parse_pos_sentence(self, sentence: str) -> list[dict[str, str]]:
        items = []

        for raw_item in sentence.split(", "):
            match = POS_ITEM_REGEX.fullmatch(raw_item.strip())

            if not match:
                continue

            items.append(
                {
                    "word": match.group("word") or "",
                    "tag": match.group("tag") or "",
                    "feats": match.group("feats") or "",
                }
            )

        return items

    def _parse_dep_sentence(self, sentence: str) -> list[dict[str, str | int]]:
        items = []

        for raw_item in sentence.split(", "):
            match = DEPENDENCY_ITEM_REGEX.fullmatch(raw_item.strip())

            if not match:
                continue

            try:
                head = int(match.group("head") or 0)
            except ValueError:
                head = 0

            items.append(
                {
                    "word": match.group("word") or "",
                    "deprel": match.group("deprel") or "",
                    "head": head,
                }
            )

        return items