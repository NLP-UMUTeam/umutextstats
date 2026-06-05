from dataclasses import dataclass

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
    tagged_pos: str,
    text: str | None = None,
) -> list[POSItem]:
    if not tagged_pos:
        return []

    if text is None:
        text = plain_text_from_tagged_pos(tagged_pos)

    items: list[POSItem] = []
    search_start = 0

    for item in parse_tagged_pos(tagged_pos):
        start = text.find(item.word, search_start)

        if start == -1:
            start = text.find(item.word)

        if start == -1:
            continue

        end = start + len(item.word)
        search_start = end

        items.append(
            POSItem(
                word=item.word,
                tag=item.tag,
                feats=item.feats,
                start=start,
                end=end,
            )
        )

    return items


def looks_like_tagged_pos(text: str) -> bool:
    return "__(" in text and ")" in text