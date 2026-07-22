from __future__ import annotations

import pandas as pd

from umutextstats.config.params import param
from umutextstats.dimensions.mixins import TextComputeMixin
from umutextstats.dimensions.results import DimensionComputation
from umutextstats.inspection.scalar_inspectable_dimension import (
    ScalarInspectableDimension,
)
from umutextstats.text.patterns import DEPENDENCY_ITEM_REGEX


class DependencyDepthDimension(
    TextComputeMixin,
    ScalarInspectableDimension,
):
    """
    Compute dependency-tree depth statistics.

    Supported modes:
    - max: maximum dependency depth
    - mean: average dependency depth
    - sum: sum of dependency depths
    """

    def __init__(
        self,
        key: str,
        input_column: str = "tagged_dep",
        mode: str = "max",
    ):
        super().__init__(
            key=key,
            input_column=input_column,
        )

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
            mode=param(
                dimension,
                "mode",
                "max",
            ),
        )

    def _compute_text(
        self,
        tagged_text: str,
    ) -> float:
        values = self._get_depth_values(
            tagged_text
        )

        return self._aggregate(values)

    def compute_result(
        self,
        df: pd.DataFrame,
    ) -> DimensionComputation:
        """
        Compute dependency-depth values and aggregation components.
        """
        texts = self.get_text_series(df)

        observations = texts.apply(
            self._get_depth_values
        )

        values = observations.apply(
            self._aggregate
        ).astype(float)

        metadata = {
            "measure": (
                "dependency_depth"
            ),
            "aggregation": self.mode,
            "observation_unit": (
                "dependency_items"
            ),
            "unit": "dependency_edges",
        }

        if self.mode == "mean":
            numerators = observations.apply(sum)
            denominators = observations.apply(len)

            return DimensionComputation(
                values=values,
                numerators=numerators,
                denominators=denominators,
                metadata=metadata,
            )

        return DimensionComputation(
            values=values,
            metadata=metadata,
        )

    def _get_depth_values(
        self,
        tagged_text: str,
    ) -> list[int]:
        """
        Return the depth of each parsed dependency item.
        """
        tagged_text = (
            ""
            if tagged_text is None
            else str(tagged_text)
        )

        all_values = []

        for sentence in tagged_text.split(
            " || "
        ):
            items = parse_tagged_dep(
                sentence,
                with_id=True,
            )

            if not items:
                continue

            depths = self._compute_depths(
                items
            )

            all_values.extend(
                depths.values()
            )

        return all_values

    def _aggregate(
        self,
        values: list[int],
    ) -> float:
        """
        Aggregate dependency depths according to the configured mode.
        """
        if not values:
            return 0.0

        if self.mode == "mean":
            return float(
                sum(values) / len(values)
            )

        if self.mode == "sum":
            return float(sum(values))

        return float(max(values))

    def _compute_depths(
        self,
        items: list[dict],
    ) -> dict[int, int]:
        """
        Compute dependency depth for each item.
        """
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
        """
        Recursively compute an item's distance from the root.
        """
        if item_id in seen:
            return 0

        seen.add(item_id)

        head = heads.get(
            item_id,
            0,
        )

        if head == 0:
            return 0

        if head not in heads:
            return 0

        return 1 + self._depth(
            item_id=head,
            heads=heads,
            seen=seen,
        )


class DependencyDistanceDimension(
    ScalarInspectableDimension
):
    """
    Compute dependency-distance statistics.

    A dependency distance is the absolute difference between a token ID
    and the ID of its syntactic head.

    Supported modes:
    - max: maximum dependency distance
    - mean: average dependency distance
    - sum: sum of dependency distances
    """

    def __init__(
        self,
        key: str,
        input_column: str = "tagged_dep",
        mode: str = "mean",
    ):
        super().__init__(
            key=key,
            input_column=input_column,
        )

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
            mode=param(
                dimension,
                "mode",
                "mean",
            ),
        )

    def compute_single(
        self,
        row: pd.Series,
    ) -> float:
        return self._compute_text(
            self.get_text(row)
        )

    def compute(
        self,
        df: pd.DataFrame,
    ) -> pd.Series:
        return self.get_text_series(df).apply(
            self._compute_text
        )

    def compute_result(
        self,
        df: pd.DataFrame,
    ) -> DimensionComputation:
        """
        Compute dependency-distance values and aggregation components.
        """
        texts = self.get_text_series(df)

        observations = texts.apply(
            self._get_distances
        )

        values = observations.apply(
            self._aggregate
        ).astype(float)

        metadata = {
            "measure": (
                "dependency_distance"
            ),
            "aggregation": self.mode,
            "observation_unit": (
                "dependency_relations"
            ),
            "unit": "token_positions",
        }

        if self.mode == "mean":
            numerators = observations.apply(
                sum
            )

            denominators = observations.apply(
                len
            )

            return DimensionComputation(
                values=values,
                numerators=numerators,
                denominators=denominators,
                metadata=metadata,
            )

        return DimensionComputation(
            values=values,
            metadata=metadata,
        )

    def _compute_text(
        self,
        tagged_text: str,
    ) -> float:
        distances = self._get_distances(
            tagged_text
        )

        return self._aggregate(
            distances
        )

    def _get_distances(
        self,
        tagged_text: str,
    ) -> list[int]:
        """
        Return dependency distances for all non-root items.
        """
        tagged_text = (
            ""
            if tagged_text is None
            else str(tagged_text)
        )

        distances = []

        for sentence in tagged_text.split(
            " || "
        ):
            items = parse_tagged_dep(
                sentence,
                with_id=True,
            )

            distances.extend(
                abs(
                    item["id"]
                    - item["head"]
                )
                for item in items
                if item["head"] > 0
            )

        return distances

    def _aggregate(
        self,
        distances: list[int],
    ) -> float:
        """
        Aggregate dependency distances according to the configured mode.
        """
        if not distances:
            return 0.0

        if self.mode == "max":
            return float(
                max(distances)
            )

        if self.mode == "sum":
            return float(
                sum(distances)
            )

        return float(
            sum(distances)
            / len(distances)
        )


class DependencyTag(
    TextComputeMixin,
    ScalarInspectableDimension,
):
    """
    Compute the percentage of dependency items matching a relation.
    """

    def __init__(
        self,
        key: str,
        input_column: str = "tagged_dep",
        deprel: str | None = None,
    ):
        super().__init__(
            key=key,
            input_column=input_column,
        )

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
            deprel=param(
                dimension,
                "deprel",
            ),
        )

    def _compute_text(
        self,
        tagged_text: str,
    ) -> float:
        total_items, matches = (
            self._analyze_text(
                tagged_text
            )
        )

        if total_items == 0:
            return 0.0

        return (
            100.0
            * matches
            / total_items
        )

    def compute_result(
        self,
        df: pd.DataFrame,
    ) -> DimensionComputation:
        """
        Compute dependency-tag percentages and their components.
        """
        texts = self.get_text_series(df)

        analyses = texts.apply(
            self._analyze_text
        )

        denominators = analyses.apply(
            lambda analysis: analysis[0]
        )

        numerators = analyses.apply(
            lambda analysis: analysis[1]
        )

        values = pd.Series(
            [
                (
                    100.0
                    * numerator
                    / denominator
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

        return DimensionComputation(
            values=values,
            numerators=numerators,
            denominators=denominators,
            metadata={
                "measure": "rate",
                "numerator_unit": (
                    "matching_dependency_items"
                ),
                "normalization_unit": (
                    "dependency_items"
                ),
                "scale": 100.0,
                "dependency_relation": (
                    self.deprel
                ),
            },
        )

    def _analyze_text(
        self,
        tagged_text: str,
    ) -> tuple[int, int]:
        """
        Return total dependency items and matching items.
        """
        items = parse_tagged_dep(
            tagged_text,
            with_id=False,
        )

        matches = sum(
            1
            for item in items
            if self._matches(item)
        )

        return len(items), matches

    def _matches(
        self,
        item: dict[str, str],
    ) -> bool:
        if not self.deprel:
            return False

        return (
            item["deprel"]
            == self.deprel
        )


def parse_tagged_dep(
    tagged_text: str,
    with_id: bool = False,
) -> list[dict]:
    """
    Parse dependency-tagged text.

    Items use the serialized form:

        word__(deprel)(head)

    Sentence blocks can be separated by ` || `. Token IDs restart at 1
    for every sentence.
    """
    if not tagged_text:
        return []

    tagged_text = str(tagged_text)
    items = []

    for sentence in tagged_text.split(
        " || "
    ):
        for index, raw_item in enumerate(
            sentence.split(", "),
            start=1,
        ):
            match = (
                DEPENDENCY_ITEM_REGEX.fullmatch(
                    raw_item.strip()
                )
            )

            if not match:
                continue

            try:
                head = int(
                    match.group("head")
                    or 0
                )
            except ValueError:
                head = 0

            item = {
                "word": (
                    match.group("word")
                    or ""
                ),
                "deprel": (
                    match.group("deprel")
                    or ""
                ),
                "head": head,
            }

            if with_id:
                item["id"] = index

            items.append(item)

    return items