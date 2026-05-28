from __future__ import annotations

import regex as re

from umutextstats.dimensions.base import BaseDimension
from umutextstats.text.patterns import DEPENDENCY_ITEM_REGEX


class DependencyDistanceDimension(BaseDimension):
    def __init__(
        self,
        key: str,
        input_column: str = "tagged_dep",
        mode: str = "mean",
    ):
        super().__init__(key=key, input_column=input_column)
        self.mode = mode

    def compute(self, df):
        return (
            df[self.input_column]
            .fillna("")
            .astype(str)
            .apply(self._compute_text)
        )

    def _compute_text(self, tagged_text: str) -> float:
        distances = []

        for sentence in tagged_text.split(" || "):
            items = self._parse_tagged_dep(sentence)

            distances.extend(
                abs(item["id"] - item["head"])
                for item in items
                if item["head"] > 0
            )

        if not distances:
            return 0.0

        if self.mode == "max":
            return float(max(distances))

        if self.mode == "sum":
            return float(sum(distances))

        return float(sum(distances) / len(distances))

    def _parse_tagged_dep(self, tagged_text: str) -> list[dict]:
        if not tagged_text:
            return []

        items = []

        for index, raw_item in enumerate(tagged_text.split(", "), start=1):
            match = DEPENDENCY_ITEM_REGEX.fullmatch(raw_item.strip())

            if not match:
                continue

            try:
                head = int(match.group("head") or 0)
            except ValueError:
                head = 0

            items.append(
                {
                    "id": index,
                    "word": match.group("word") or "",
                    "deprel": match.group("deprel") or "",
                    "head": head,
                }
            )

        return items