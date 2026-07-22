from __future__ import annotations

import pandas as pd

from umutextstats.config.params import param
from umutextstats.dimensions.results import DimensionComputation
from umutextstats.inspection.scalar_inspectable_dimension import (
    ScalarInspectableDimension,
)
from umutextstats.text.patterns import (
    DEPENDENCY_ITEM_REGEX,
    POS_ITEM_REGEX,
)


class RootPOSTagDimension(ScalarInspectableDimension):
    """
    Compute the percentage of dependency roots whose POS tag matches
    a configured tag.

    This dimension needs two annotation columns:

    - `input_column`: POS-tagged text, usually "tagged_pos".
    - `tagged_dep_column`: dependency-tagged text, usually "tagged_dep".
    """

    def __init__(
        self,
        key: str,
        input_column: str = "tagged_pos",
        tagged_dep_column: str = "tagged_dep",
        tag: str | None = None,
    ):
        super().__init__(
            key=key,
            input_column=input_column,
        )

        self.tagged_dep_column = tagged_dep_column
        self.tag = tag

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
            tagged_dep_column=param(
                dimension,
                "tagged_dep_column",
                "tagged_dep",
            ),
            tag=param(
                dimension,
                "tag",
            ),
        )

    def compute_single(
        self,
        row: pd.Series,
    ) -> float:
        """
        Compute the root POS tag percentage for a single row.
        """
        matches, total_roots = self._analyze_row(
            row
        )

        if total_roots == 0:
            return 0.0

        return (
            100.0
            * matches
            / total_roots
        )

    def compute(
        self,
        df: pd.DataFrame,
    ) -> pd.Series:
        """
        Compute the root POS tag percentage for all rows.
        """
        return df.apply(
            self._compute_row,
            axis=1,
        )

    def compute_result(
        self,
        df: pd.DataFrame,
    ) -> DimensionComputation:
        """
        Compute root POS tag percentages and their components.
        """
        analyses = df.apply(
            self._analyze_row,
            axis=1,
        )

        numerators = analyses.apply(
            lambda analysis: analysis[0]
        )

        denominators = analyses.apply(
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
                    "roots_matching_pos_tag"
                ),
                "normalization_unit": (
                    "aligned_dependency_roots"
                ),
                "scale": 100.0,
                "target_pos_tag": self.tag,
                "pos_column": self.input_column,
                "dependency_column": (
                    self.tagged_dep_column
                ),
                "alignment_method": (
                    "sentence_and_token_position"
                ),
            },
        )

    def _compute_row(
        self,
        row: pd.Series,
    ) -> float:
        """
        Compute the dimension from a DataFrame row.
        """
        matches, total_roots = self._analyze_row(
            row
        )

        if total_roots == 0:
            return 0.0

        return (
            100.0
            * matches
            / total_roots
        )

    def _analyze_row(
        self,
        row: pd.Series,
    ) -> tuple[int, int]:
        """
        Return matching and total aligned dependency-root counts.
        """
        tagged_pos = self.get_text(
            row
        )

        tagged_dep = self.get_text(
            row=row,
            column=self.tagged_dep_column,
        )

        return self._analyze_text(
            tagged_pos=tagged_pos,
            tagged_dep=tagged_dep,
        )

    def _analyze_text(
        self,
        tagged_pos: str,
        tagged_dep: str,
    ) -> tuple[int, int]:
        """
        Return matching and total dependency roots.

        POS and dependency sentences are aligned by sentence position.
        Tokens within each sentence are aligned by token position.
        """
        pos_sentences = self._split_sentences(
            tagged_pos
        )

        dep_sentences = self._split_sentences(
            tagged_dep
        )

        total_roots = 0
        matches = 0

        for pos_sentence, dep_sentence in zip(
            pos_sentences,
            dep_sentences,
        ):
            pos_items = self._parse_pos_sentence(
                pos_sentence
            )

            dep_items = self._parse_dep_sentence(
                dep_sentence
            )

            if not pos_items or not dep_items:
                continue

            root_indices = [
                index
                for index, item in enumerate(
                    dep_items
                )
                if item["head"] == 0
            ]

            for root_index in root_indices:
                if root_index >= len(pos_items):
                    continue

                total_roots += 1

                if (
                    self.tag
                    and pos_items[root_index]["tag"]
                    == self.tag
                ):
                    matches += 1

        return matches, total_roots

    @staticmethod
    def _split_sentences(
        tagged_text: str,
    ) -> list[str]:
        """
        Split tagged text into sentence chunks.
        """
        if not tagged_text:
            return []

        return [
            sentence.strip()
            for sentence in str(
                tagged_text
            ).split(" || ")
            if sentence.strip()
        ]

    @staticmethod
    def _parse_pos_sentence(
        sentence: str,
    ) -> list[dict[str, str]]:
        """
        Parse a POS-tagged sentence into token dictionaries.
        """
        items = []

        for raw_item in sentence.split(", "):
            match = POS_ITEM_REGEX.fullmatch(
                raw_item.strip()
            )

            if not match:
                continue

            items.append(
                {
                    "word": (
                        match.group("word")
                        or ""
                    ),
                    "tag": (
                        match.group("tag")
                        or ""
                    ),
                    "feats": (
                        match.group("feats")
                        or ""
                    ),
                }
            )

        return items

    @staticmethod
    def _parse_dep_sentence(
        sentence: str,
    ) -> list[dict[str, str | int]]:
        """
        Parse a dependency-tagged sentence into token dictionaries.
        """
        items = []

        for raw_item in sentence.split(", "):
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

            items.append(
                {
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
            )

        return items

    def inspection_debug_text(
        self,
    ) -> str:
        """
        Return configuration details used during inspection.
        """
        return (
            f"Root POS tag filter: {self.tag}\n"
            f"POS column: {self.input_column}\n"
            f"Dependency column: {self.tagged_dep_column}"
        )