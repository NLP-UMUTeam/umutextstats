from dataclasses import dataclass
from dataclasses import replace

from umutextstats.text.patterns import POS_ITEM_REGEX


@dataclass(frozen=True)
class POSItem:
    word: str
    tag: str
    feats: str
    start: int | None = None
    end: int | None = None


def split_feats(feats: str | None) -> set[str]:
    if not feats:
        return set()

    return {
        feat.strip()
        for feat in feats.split("|")
        if feat.strip()
    }


def pos_item_matches(
    item: POSItem,
    tag: str | None,
    universal: str | None,
) -> bool:
    item_feats = split_feats(item.feats)
    required_feats = split_feats(universal)

    if tag and item.tag != tag:
        return False

    if required_feats:
        return required_feats.issubset(item_feats)

    return bool(tag)


def parse_tagged_pos(tagged_text: str) -> list[POSItem]:
    if not tagged_text:
        return []

    items: list[POSItem] = []

    for sentence in tagged_text.split(" || "):
        for raw_item in sentence.split(", "):
            match = POS_ITEM_REGEX.fullmatch(raw_item.strip())

            if not match:
                continue

            items.append(
                POSItem(
                    word=match.group("word"),
                    tag=match.group("tag") or "",
                    feats=match.group("feats") or "",
                )
            )

    return items


def plain_text_from_tagged_pos(tagged_pos: str) -> str:
    words = [item.word for item in parse_tagged_pos(tagged_pos)]

    text = " ".join(words)

    return (
        text.replace(" .", ".")
        .replace(" ,", ",")
        .replace(" ;", ";")
        .replace(" :", ":")
        .replace(" ?", "?")
        .replace(" !", "!")
    )


def parse_tagged_pos_with_offsets(
    tagged_text: str,
) -> list[POSItem]:
    """
    Parse serialized POS items preserving absolute word offsets.

    Offsets refer to the word inside the complete serialized tagged text.
    """
    if not tagged_text:
        return []

    items = []

    for match in POS_ITEM_REGEX.finditer(tagged_text):
        word = match.group("word") or ""
        tag = match.group("tag") or ""
        feats = match.group("feats") or ""

        word_start, word_end = match.span("word")

        items.append(
            POSItem(
                word=word,
                tag=tag,
                feats=feats,
                start=word_start,
                end=word_end,
            )
        )

    return items


def looks_like_tagged_pos(text: str) -> bool:
    return "__(" in text and ")" in text

def parse_tagged_pos_with_offsets(
    tagged_text: str,
) -> list[POSItem]:
    """
    Parse POS items and attach absolute word offsets.

    The semantic parsing is delegated to `parse_tagged_pos()`.
    Words are then located sequentially in the complete serialized
    annotation.
    """
    if not tagged_text:
        return []

    parsed_items = parse_tagged_pos(tagged_text)

    items_with_offsets = []
    cursor = 0

    for item in parsed_items:
        start = tagged_text.find(
            item.word,
            cursor,
        )

        if start < 0:
            items_with_offsets.append(
                replace(
                    item,
                    start=None,
                    end=None,
                )
            )
            continue

        end = start + len(item.word)

        items_with_offsets.append(
            replace(
                item,
                start=start,
                end=end,
            )
        )

        cursor = end

    return items_with_offsets