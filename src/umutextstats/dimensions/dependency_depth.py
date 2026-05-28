from __future__ import annotations

import regex as re

from umutextstats.dimensions.base import BaseDimension
from umutextstats.text.patterns import DEPENDENCY_ITEM_REGEX

class DependencyDepthDimension(BaseDimension):
    def __init__(
        self,
        key: str,
        input_column: str = "tagged_dep",
        mode: str = "max",
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
        all_values = []

        for sentence in tagged_text.split(" || "):
            items = self._parse_tagged_dep(sentence)

            if not items:
                continue

            depths = self._compute_depths(items)

            if depths:
                all_values.extend(depths.values())

        if not all_values:
            return 0.0

        if self.mode == "mean":
            return sum(all_values) / len(all_values)

        if self.mode == "sum":
            return float(sum(all_values))

        return float(max(all_values))

    def _compute_depths(self, items: list[dict]) -> dict[int, int]:
        heads = {
            item["id"]: item["head"]
            for item in items
        }

        depths = {}

        for item_id in heads:
            depths[item_id] = self._depth(item_id, heads, seen=set())

        return depths

    def _depth(
        self,
        item_id: int,
        heads: dict[int, int],
        seen: set[int],
    ) -> int:
        if item_id in seen:
            return 0

        seen.add(item_id)

        head = heads.get(item_id, 0)

        if head == 0:
            return 0

        if head not in heads:
            return 0

        return 1 + self._depth(head, heads, seen)

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