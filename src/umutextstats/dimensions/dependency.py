from __future__ import annotations

import pandas as pd

from umutextstats.dimensions.mixins import TextComputeMixin

from umutextstats.config.params import param
from umutextstats.inspection.scalar_inspectable_dimension import (
    ScalarInspectableDimension,
)
from umutextstats.text.patterns import DEPENDENCY_ITEM_REGEX


class DependencyDepthDimension(TextComputeMixin, ScalarInspectableDimension):
    def __init__(
        self,
        key: str,
        input_column: str = "tagged_dep",
        mode: str = "max",
    ):
        super().__init__(key=key, input_column=input_column)
        self.mode = mode

    @classmethod
    def from_config(
        cls,
        dimension,
        input_column: str = "tagged_dep",
    ):
        return cls(
            key=dimension.key,
            input_column=input_column,
            mode=param(dimension, "mode", "max"),
        )
    
    def _compute_text(
        self,
        tagged_text: str,
    ) -> float:
        all_values = []

        for sentence in tagged_text.split(" || "):
            items = parse_tagged_dep(sentence, with_id=True)

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

    def _compute_depths(
        self,
        items: list[dict],
    ) -> dict[int, int]:
        heads = {
            item["id"]: item["head"]
            for item in items
        }

        depths = {}

        for item_id in heads:
            depths[item_id] = self._depth(
                item_id=item_id,
                heads=heads,
                seen=set(),
            )

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

        return 1 + self._depth(
            item_id=head,
            heads=heads,
            seen=seen,
        )


class DependencyDistanceDimension(ScalarInspectableDimension):
    def __init__(
        self,
        key: str,
        input_column: str = "tagged_dep",
        mode: str = "mean",
    ):
        super().__init__(key=key, input_column=input_column)
        self.mode = mode

    @classmethod
    def from_config(
        cls,
        dimension,
        input_column: str = "tagged_dep",
    ):
        return cls(
            key=dimension.key,
            input_column=input_column,
            mode=param(dimension, "mode", "mean"),
        )

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

    def _compute_text(
        self,
        tagged_text: str,
    ) -> float:
        distances = []

        for sentence in tagged_text.split(" || "):
            items = parse_tagged_dep(sentence, with_id=True)

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


class DependencyTag(TextComputeMixin, ScalarInspectableDimension):
    def __init__(
        self,
        key: str,
        input_column: str = "tagged_dep",
        deprel: str | None = None,
    ):
        super().__init__(key=key, input_column=input_column)
        self.deprel = deprel

    @classmethod
    def from_config(
        cls,
        dimension,
        input_column: str = "tagged_dep",
    ):
        return cls(
            key=dimension.key,
            input_column=input_column,
            deprel=param(dimension, "deprel"),
        )

    def _compute_text(
        self,
        tagged_text: str,
    ) -> float:
        items = parse_tagged_dep(tagged_text, with_id=False)
        total_words = len(items)

        if total_words == 0:
            return 0.0

        matches = sum(
            1
            for item in items
            if self._matches(item)
        )

        return (100 * matches) / total_words

    def _matches(
        self,
        item: dict[str, str],
    ) -> bool:
        if self.deprel:
            return item["deprel"] == self.deprel

        return False


def parse_tagged_dep(
    tagged_text: str,
    with_id: bool = False,
) -> list[dict]:
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

        item = {
            "word": match.group("word") or "",
            "deprel": match.group("deprel") or "",
            "head": head,
        }

        if with_id:
            item["id"] = index

        items.append(item)

    return items