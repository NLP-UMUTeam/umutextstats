import regex as re

from umutextstats.dimensions.base import BaseDimension
from umutextstats.text.patterns import DEPENDENCY_ITEM_REGEX


class DependencyTag(BaseDimension):
    def __init__(
        self,
        key: str,
        input_column: str = "tagged_dep",
        deprel: str | None = None,
    ):
        super().__init__(key=key, input_column=input_column)
        self.deprel = deprel

    def compute(self, df):
        return (
            df[self.input_column]
            .fillna("")
            .astype(str)
            .apply(self._compute_text)
        )

    def _compute_text(self, tagged_text: str) -> float:
        items = self._parse_tagged_dep(tagged_text)

        total_words = len(items)

        if total_words == 0:
            return 0.0

        matches = sum(1 for item in items if self._matches(item))

        return (100 * matches) / total_words

    def _matches(self, item: dict[str, str]) -> bool:
        if self.deprel:
            return item["deprel"] == self.deprel

        return False

    def _parse_tagged_dep(self, tagged_text: str) -> list[dict[str, str]]:
        if not tagged_text:
            return []

        items = []

        for raw_item in tagged_text.split(", "):
            match = DEPENDENCY_ITEM_REGEX.fullmatch(raw_item.strip())

            if not match:
                continue

            items.append(
                {
                    "word": match.group("word") or "",
                    "deprel": match.group("deprel") or "",
                    "head": match.group("head") or "",
                }
            )

        return items