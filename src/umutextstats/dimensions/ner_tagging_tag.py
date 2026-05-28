import regex as re

from umutextstats.dimensions.base import BaseDimension
from umutextstats.text.patterns import NER_ITEM_REGEX


class NERTaggingTag(BaseDimension):
    def __init__(
        self,
        key: str,
        tag: str | None = None,
        input_column: str = "tagged_ner",
    ):
        super().__init__(key=key, input_column=input_column)
        self.tag = tag

    def compute(self, df):
        return (
            df[self.input_column]
            .fillna("")
            .astype(str)
            .apply(self._compute_text)
        )

    def _compute_text(self, tagged_ner: str) -> float:
        entities = self._parse_entities(tagged_ner)

        total_entities = len(entities)

        if total_entities == 0:
            return 0.0

        if not self.tag:
            return 0.0

        matches = sum(1 for entity in entities if entity["tag"] == self.tag)

        return (100 * matches) / total_entities

    def _parse_entities(self, tagged_ner: str) -> list[dict[str, str]]:
        if not tagged_ner:
            return []

        entities = []

        for match in NER_ITEM_REGEX.finditer(tagged_ner):
            entities.append(
                {
                    "tag": match.group("tag"),
                    "text": match.group("text"),
                }
            )

        return entities