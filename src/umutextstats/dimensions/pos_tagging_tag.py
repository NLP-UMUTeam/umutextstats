import regex as re

from umutextstats.dimensions.base import BaseDimension


POS_ITEM_REGEX = re.compile(
    r"(?P<word>.+?)__\((?P<tag>[^)]*)\)(?:\((?P<feats>[^)]*)\))?"
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
        items = self._parse_tagged_pos(tagged_text)

        total_words = len(items)

        if total_words == 0:
            return 0.0

        matches = sum(1 for item in items if self._matches(item))

        return (100 * matches) / total_words

    def _split_feats(self, feats: str | None) -> set[str]:
        if not feats:
            return set()

        return {
            feat.strip()
            for feat in feats.split("|")
            if feat.strip()
        }


    def _matches(self, item: dict[str, str]) -> bool:
        tag = item["tag"]
        feats = self._split_feats(item["feats"])
        required_feats = self._split_feats(self.postagger_universal)

        if self.postagger_tag and tag != self.postagger_tag:
            return False

        if required_feats:
            return required_feats.issubset(feats)

        return bool(self.postagger_tag)

    def _parse_tagged_pos(self, tagged_text: str) -> list[dict[str, str]]:
        if not tagged_text:
            return []

        items = []

        for sentence in tagged_text.split(" || "):
            for raw_item in sentence.split(", "):
                match = POS_ITEM_REGEX.fullmatch(raw_item.strip())

                if not match:
                    continue

                items.append(
                    {
                        "word": match.group("word"),
                        "tag": match.group("tag") or "",
                        "feats": match.group("feats") or "",
                    }
                )

        return items