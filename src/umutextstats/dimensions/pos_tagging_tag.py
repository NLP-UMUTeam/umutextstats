from __future__ import annotations

import numpy as np
import pandas as pd

from umutextstats.config.params import param
from umutextstats.dimensions.mixins import TextComputeMixin
from umutextstats.dimensions.results import DimensionComputation
from umutextstats.inspection.iterable_inspectable_dimension import (
    IterableInspectableDimension,
)
from umutextstats.text.pos import (
    POSItem,
    parse_tagged_pos_with_offsets,
    pos_item_matches,
)


class POSTaggingTag(
    TextComputeMixin,
    IterableInspectableDimension,
):
    """
    Compute the percentage of POS-tagged items matching a configured
    POS tag or universal feature filter.

    Evidence offsets refer to the serialized `tagged_pos` input.
    """

    def __init__(
        self,
        key: str,
        input_column: str = "tagged_pos",
        postagger_tag: str | None = None,
        postagger_universal: str | None = None,
    ):
        super().__init__(
            key=key,
            input_column=input_column,
        )

        self.postagger_tag = postagger_tag
        self.postagger_universal = postagger_universal

    @classmethod
    def from_config(
        cls,
        dimension,
        input_column: str = "tagged_pos",
    ):
        """
        Build the dimension from configuration.
        """
        return cls(
            key=dimension.key,
            input_column=input_column,
            postagger_tag=param(
                dimension,
                "tag",
            ),
            postagger_universal=param(
                dimension,
                "universal",
            ),
        )

    def _analyze_text(
        self,
        tagged_text: str,
    ) -> tuple[list[POSItem], list[POSItem]]:
        """
        Parse a POS annotation and return all items and matching items.

        This is the single source of truth for computation, inspection,
        and structured extraction.
        """
        items = list(
            parse_tagged_pos_with_offsets(
                tagged_text
            )
        )

        matches = [
            item
            for item in items
            if self._matches(item)
        ]

        return items, matches

    def _compute_text(
        self,
        tagged_text: str,
    ) -> float:
        """
        Compute the percentage of matching POS items.
        """
        items, matches = self._analyze_text(
            tagged_text
        )

        if not items:
            return 0.0

        return (
            100.0
            * len(matches)
            / len(items)
        )

    def compute_result(
        self,
        df: pd.DataFrame,
    ) -> DimensionComputation:
        """
        Compute values, numerators, denominators, and POS evidence.
        """
        tagged_texts = self.get_text_series(df)

        analyses = tagged_texts.apply(
            self._analyze_text
        )

        denominators = analyses.apply(
            lambda analysis: len(analysis[0])
        )

        matched_items = analyses.apply(
            lambda analysis: analysis[1]
        )

        numerators = matched_items.apply(len)

        evidence = matched_items.apply(
            self._items_to_evidence
        )

        numerator_array = numerators.to_numpy(
            dtype=float
        )

        denominator_array = denominators.to_numpy(
            dtype=float
        )

        percentages = np.zeros_like(
            numerator_array,
            dtype=float,
        )

        np.divide(
            100.0 * numerator_array,
            denominator_array,
            out=percentages,
            where=denominator_array != 0,
        )

        return DimensionComputation(
            values=pd.Series(
                percentages,
                index=df.index,
            ),
            numerators=numerators,
            denominators=denominators,
            evidence=evidence,
            metadata={
                "measure": "rate",
                "normalization_unit": "pos_items",
                "scale": 100.0,
                "evidence_offset_unit": (
                    "serialized_pos_characters"
                ),
            },
        )

    def iter_matches(
        self,
        tagged_text: str,
    ):
        """
        Yield the same POS matches used for computation.
        """
        _, matches = self._analyze_text(
            tagged_text
        )

        for item in matches:
            yield _POSMatch(item)

    @staticmethod
    def _items_to_evidence(
        items: list[POSItem],
    ) -> list[dict]:
        """
        Convert matching POS items to serializable evidence.
        """
        return [
            {
                "text": item.word,
                "start": (
                    item.start
                    if item.start is not None
                    else 0
                ),
                "end": (
                    item.end
                    if item.end is not None
                    else 0
                ),
            }
            for item in items
        ]

    def _matches(
        self,
        item: POSItem,
    ) -> bool:
        """
        Check whether a POS item matches the configured filters.
        """
        return pos_item_matches(
            item=item,
            tag=self.postagger_tag,
            universal=self.postagger_universal,
        )

    def inspection_debug_text(
        self,
    ) -> str:
        """
        Return configuration details used during inspection.
        """
        parts = []

        if self.postagger_tag:
            parts.append(
                f"POS tag: {self.postagger_tag}"
            )

        if self.postagger_universal:
            parts.append(
                "Universal features: "
                f"{self.postagger_universal}"
            )

        return (
            "\n".join(parts)
            or "No POS filter configured"
        )


class _POSMatch:
    """
    Regex-like wrapper around a POSItem for inspection rendering.
    """

    def __init__(
        self,
        item: POSItem,
    ):
        self.item = item

    def group(
        self,
        index=0,
    ):
        if index != 0:
            raise IndexError(index)

        return self.item.word

    def start(self):
        return (
            self.item.start
            if self.item.start is not None
            else 0
        )

    def end(self):
        return (
            self.item.end
            if self.item.end is not None
            else 0
        )