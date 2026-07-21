from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from umutextstats.config.params import param
from umutextstats.dimensions.mixins import TextComputeMixin
from umutextstats.dimensions.results import DimensionComputation
from umutextstats.inspection.iterable_inspectable_dimension import (
    IterableInspectableDimension,
)
from umutextstats.text.patterns import NER_ITEM_REGEX


NER_NORMALIZER_WORDS = "words"
NER_NORMALIZER_ENTITIES = "entities"


@dataclass(frozen=True)
class NEREntity:
    """
    Parsed NER entity with offsets in the serialized NER input.
    """

    text: str
    tag: str
    start: int
    end: int


@dataclass(frozen=True)
class NERMatch:
    """
    Regex-like match object used by the inspection layer.
    """

    entity: NEREntity

    def group(
        self,
        index: int = 0,
    ) -> str:
        if index != 0:
            raise IndexError(index)

        return self.entity.text

    def start(self) -> int:
        return self.entity.start

    def end(self) -> int:
        return self.entity.end


class NERTaggingTag(
    TextComputeMixin,
    IterableInspectableDimension,
):
    """
    Compute the percentage of NER entities matching a configured tag.

    The denominator can be either:

    - the number of detected entities;
    - the number of whitespace-separated words in the serialized
      tagged NER string.

    Evidence offsets refer to the serialized `tagged_ner` input.
    """

    def __init__(
        self,
        key: str,
        tag: str | None = None,
        input_column: str = "tagged_ner",
        normalizer: str = NER_NORMALIZER_ENTITIES,
    ):
        super().__init__(
            key=key,
            input_column=input_column,
        )

        self.tag = tag
        self.normalizer = normalizer

    @classmethod
    def from_config(
        cls,
        dimension,
        input_column: str = "tagged_ner",
    ):
        """
        Build the dimension from configuration.
        """
        return cls(
            key=dimension.key,
            tag=param(
                dimension,
                "tag",
            ),
            input_column=input_column,
            normalizer=param(
                dimension,
                "normalizer",
                NER_NORMALIZER_ENTITIES,
            ),
        )

    def _analyze_text(
        self,
        tagged_ner: str,
    ) -> tuple[list[NEREntity], list[NEREntity]]:
        """
        Parse all entities and select those matching the configured tag.

        This is the single source of truth for computation, inspection,
        and structured extraction.
        """
        entities = self._parse_entities(
            tagged_ner
        )

        if not self.tag:
            return entities, []

        matches = [
            entity
            for entity in entities
            if entity.tag == self.tag
        ]

        return entities, matches

    def _compute_text(
        self,
        tagged_ner: str,
    ) -> float:
        """
        Compute the percentage of entities matching the configured tag.
        """
        tagged_ner = (
            ""
            if tagged_ner is None
            else str(tagged_ner)
        )

        entities, matches = self._analyze_text(
            tagged_ner
        )

        denominator = self._get_denominator(
            tagged_ner=tagged_ner,
            entities=entities,
        )

        if denominator == 0:
            return 0.0

        return (
            100.0
            * len(matches)
            / denominator
        )

    def compute_result(
        self,
        df: pd.DataFrame,
    ) -> DimensionComputation:
        """
        Compute values, numerators, denominators, and NER evidence.
        """
        tagged_texts = self.get_text_series(df)

        analyses = tagged_texts.apply(
            self._analyze_text
        )

        entities = analyses.apply(
            lambda analysis: analysis[0]
        )

        matched_entities = analyses.apply(
            lambda analysis: analysis[1]
        )

        numerators = matched_entities.apply(
            len
        )

        denominators = pd.Series(
            (
                self._get_denominator(
                    tagged_ner=tagged_ner,
                    entities=document_entities,
                )
                for tagged_ner, document_entities in zip(
                    tagged_texts,
                    entities,
                )
            ),
            index=df.index,
        )

        evidence = matched_entities.apply(
            self._entities_to_evidence
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

        normalization_unit = {
            NER_NORMALIZER_ENTITIES: "ner_entities",
            NER_NORMALIZER_WORDS: (
                "serialized_ner_whitespace_words"
            ),
        }[self.normalizer]

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
                "normalization_unit": normalization_unit,
                "scale": 100.0,
                "evidence_offset_unit": (
                    "serialized_ner_characters"
                ),
            },
        )

    def iter_matches(
        self,
        tagged_ner: str,
    ):
        """
        Yield the same matching entities used by computation.
        """
        _, matches = self._analyze_text(
            tagged_ner
        )

        for entity in matches:
            yield NERMatch(entity)

    def _get_denominator(
        self,
        tagged_ner: str,
        entities: list[NEREntity],
    ) -> int:
        """
        Return the denominator according to the configured normalizer.
        """
        if self.normalizer == NER_NORMALIZER_ENTITIES:
            return len(entities)

        if self.normalizer == NER_NORMALIZER_WORDS:
            return len(tagged_ner.split())

        raise ValueError(
            f"Unknown NER normalizer: {self.normalizer}"
        )

    @staticmethod
    def _entities_to_evidence(
        entities: list[NEREntity],
    ) -> list[dict]:
        """
        Convert matching entities to serializable evidence.
        """
        return [
            {
                "text": entity.text,
                "tag": entity.tag,
                "start": entity.start,
                "end": entity.end,
            }
            for entity in entities
        ]

    @staticmethod
    def _parse_entities(
        tagged_ner: str,
    ) -> list[NEREntity]:
        """
        Parse NER entities with absolute offsets in `tagged_ner`.
        """
        if not tagged_ner:
            return []

        entities = []

        for match in NER_ITEM_REGEX.finditer(
            tagged_ner
        ):
            text = match.group("text") or ""
            tag = match.group("tag") or ""

            start, end = match.span("text")

            entities.append(
                NEREntity(
                    text=text,
                    tag=tag,
                    start=start,
                    end=end,
                )
            )

        return entities

    def inspection_debug_text(
        self,
    ) -> str:
        """
        Return configuration details used during inspection.
        """
        return (
            f"NER tag: {self.tag}\n"
            f"Input column: {self.input_column}\n"
            f"Normalizer: {self.normalizer}"
        )