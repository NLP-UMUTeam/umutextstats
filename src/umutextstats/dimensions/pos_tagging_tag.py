from umutextstats.dimensions.base import BaseDimension
from umutextstats.text.pos import (
    POSItem,
    parse_tagged_pos,
    parse_tagged_pos_with_offsets,
    pos_item_matches,
)


class POSTaggingTag(BaseDimension):
    def __init__(
        self,
        key: str,
        input_column: str = "tagged_pos",
        postagger_tag: str | None = None,
        postagger_universal: str | None = None,
    ):
        super().__init__(key=key, input_column=input_column)
        self.postagger_tag = postagger_tag
        self.postagger_universal = postagger_universal

    def compute(self, df):
        return (
            df[self.input_column]
            .fillna("")
            .astype(str)
            .apply(self._compute_text)
        )

    def _compute_text(self, tagged_text: str) -> float:
        items = parse_tagged_pos(tagged_text)

        total_words = len(items)

        if total_words == 0:
            return 0.0

        matches = sum(1 for item in items if self._matches(item))

        return (100 * matches) / total_words

    def _matches(self, item: POSItem) -> bool:
        return pos_item_matches(
            item=item,
            tag=self.postagger_tag,
            universal=self.postagger_universal,
        )

    def iter_matches(self, tagged_text: str):
        for item in parse_tagged_pos_with_offsets(tagged_text):
            if self._matches(item):
                yield _POSMatch(item)


class _POSMatch:
    def __init__(self, item: POSItem):
        self.item = item

    def group(self, index=0):
        return self.item.word

    def start(self):
        return self.item.start or 0

    def end(self):
        return self.item.end or 0