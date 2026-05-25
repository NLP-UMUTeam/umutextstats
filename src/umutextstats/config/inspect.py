from __future__ import annotations

import regex as re
from rich.console import Group
from rich.text import Text
from dataclasses import dataclass

from umutextstats.config.explain import find_dimension
from umutextstats.config.models import UMUTextStatsConfig
from umutextstats.dictionaries import DictionaryLoader
from umutextstats.dimensions.pos_tagging_tag import POS_ITEM_REGEX
from umutextstats.nlp.stanza_annotator import StanzaAnnotator, format_tagged_pos

@dataclass(frozen=True)
class InspectMatch:
    match: str
    start: int
    end: int


@dataclass(frozen=True)
class DimensionInspection:
    key: str
    class_name: str | None
    pattern: str | None
    dictionary: str | None
    matches: list[InspectMatch]
    discarded_matches: list[InspectMatch] | None = None
    debug_text: str | None = None
    internal_representation: str | None = None
    description: str | None = None

def inspect_dimension_text(
    config: UMUTextStatsConfig,
    key: str,
    text: str,
) -> DimensionInspection:
    explanation = find_dimension(config, key)

    if explanation is None:
        raise ValueError(f"Dimension not found: {key}")

    dimension = explanation.dimension

    if dimension.class_name == "PatternDimension":
        return _inspect_pattern_dimension(dimension, text)

    if dimension.class_name in {"WordPerDictionary", "VerbPerDictionary"}:
        return _inspect_dictionary_dimension(dimension, text)

    if dimension.class_name == "POSTaggingTag":
        return _inspect_pos_tagging_dimension(dimension, text)

    if dimension.children:
        return _inspect_composite_dimension(config, dimension, text)

    raise ValueError(
        f"Dimension class is not inspectable yet: {dimension.class_name}"
    )

    raise ValueError(
        f"Dimension class is not inspectable yet: {dimension.class_name}"
    )


def _inspect_pattern_dimension(dimension, text: str) -> DimensionInspection:
    if not dimension.pattern:
        raise ValueError(f"PatternDimension without pattern: {dimension.key}")

    matches = [
        InspectMatch(
            match=match.group(0),
            start=match.start(),
            end=match.end(),
        )
        for match in re.finditer(dimension.pattern, text)
    ]

    return DimensionInspection(
        key=dimension.key,
        class_name=dimension.class_name,
        pattern=None,
        dictionary=None,
        matches=matches,
        discarded_matches=[],
        debug_text=dimension.pattern,
    )


def _inspect_dictionary_dimension(dimension, text: str) -> DimensionInspection:
    dictionary_name = (
        dimension.dictionary
        or dimension.params.get("dictionary")
        or dimension.params.get("dictionaries")
    )

    if not dictionary_name:
        raise ValueError(f"Dictionary dimension without dictionary: {dimension.key}")

    pos_tag = getattr(dimension, "pos_tag", None) or dimension.params.get("pos_tag")

    if _looks_like_tagged_pos(text):
        tagged_pos = text
        source_text = _plain_text_from_tagged_pos(tagged_pos)
    else:
        tagged_pos = None
        source_text = text

    use_regex = not dimension.disabled_regexp

    matches: list[InspectMatch] = []
    discarded_matches: list[InspectMatch] = []

    for name in _split_dictionary_names(dictionary_name):
        entries = DictionaryLoader().load(name)

        if use_regex:
            positive_matches = _find_dictionary_regex_matches(
                text=source_text,
                entries=entries.words,
                dictionary_name=name,
            )

            exception_matches = _find_dictionary_regex_matches(
                text=source_text,
                entries=entries.exceptions,
                dictionary_name=name,
            )
        else:
            positive_matches = _find_dictionary_plain_matches(
                text=source_text,
                entries=entries.words,
            )

            exception_matches = _find_dictionary_plain_matches(
                text=source_text,
                entries=entries.exceptions,
            )

        if pos_tag:
            if not tagged_pos:
                annotator = StanzaAnnotator()
                doc = annotator.annotate_texts([source_text])[0]
                tagged_pos = format_tagged_pos(doc)

            positive_matches = _filter_matches_by_pos_tag(
                matches=positive_matches,
                tagged_pos=tagged_pos,
                text=source_text,
                pos_tag=pos_tag,
            )

            exception_matches = _filter_matches_by_pos_tag(
                matches=exception_matches,
                tagged_pos=tagged_pos,
                text=source_text,
                pos_tag=pos_tag,
            )

        kept, discarded = _remove_exception_matches(
            positive_matches,
            exception_matches,
        )

        matches.extend(kept)
        discarded_matches.extend(discarded)

    return DimensionInspection(
        key=dimension.key,
        class_name=dimension.class_name,
        pattern=None,
        dictionary=dictionary_name,
        matches=matches,
        discarded_matches=discarded_matches,
        debug_text=tagged_pos or f"Loaded dictionary: {dictionary_name}",
        internal_representation=tagged_pos,
    )

def render_inspection(
    inspection: DimensionInspection,
    text: str,
):
    lines = [
        Text(f"Key: {inspection.key}", style="bold"),
    ]

    if inspection.class_name:
        lines.append(Text(f"Class: {inspection.class_name}"))

    if inspection.pattern:
        lines.append(Text(f"Pattern: {inspection.pattern}"))

    if inspection.dictionary:
        lines.append(Text(f"Dictionary: {inspection.dictionary}"))

    lines.append(Text(""))
    lines.append(Text(f"Matches: {len(inspection.matches)}"))

    for match in inspection.matches:
        lines.append(
            Text(f"  - {match.match} [{match.start}:{match.end}]")
        )

    discarded_matches = inspection.discarded_matches or []

    if discarded_matches:
        lines.append(Text(""))
        lines.append(
            Text(
                f"Discarded by exceptions: "
                f"{len(discarded_matches)}"
            )
        )

        for match in discarded_matches:
            lines.append(
                Text(
                    f"  - {match.match} "
                    f"[{match.start}:{match.end}]"
                )
            )

    if inspection.debug_text:
        lines.append(Text(""))
        lines.append(Text("Internal representation:", style="bold"))
        lines.append(Text(inspection.debug_text))

    if inspection.matches:
        lines.append(Text(""))
        lines.append(Text("Highlighted:", style="bold"))
        lines.append(highlight_matches(text, inspection.matches))

    return Group(*lines)


def highlight_matches(text: str, matches: list[InspectMatch]) -> Text:
    highlighted = Text(text)

    for match in matches:
        highlighted.stylize(
            "bold red",
            match.start,
            match.end,
        )

    return highlighted


def _split_dictionary_names(dictionary_name: str) -> list[str]:
    return [
        name.strip()
        for name in dictionary_name.split("|")
        if name.strip()
    ]


def _find_dictionary_plain_matches(
    text: str,
    entries: list[str],
) -> list[InspectMatch]:
    matches: list[InspectMatch] = []

    for entry in entries:
        pattern = rf"(?<!\p{{L}}){re.escape(entry)}(?!\p{{L}})"

        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            matches.append(
                InspectMatch(
                    match=match.group(0),
                    start=match.start(),
                    end=match.end(),
                )
            )

    return matches

def _split_dictionary_names(dictionary_name: str) -> list[str]:
    return [
        name.strip()
        for name in dictionary_name.split("|")
        if name.strip()
    ]


def _find_dictionary_regex_matches(
    text: str,
    entries: list[str],
    dictionary_name: str,
) -> list[InspectMatch]:
    matches: list[InspectMatch] = []

    for line_number, entry in enumerate(entries, start=1):
        pattern = rf"(?<!\p{{L}}){entry}(?!\p{{L}})"

        try:
            compiled = re.compile(pattern, re.IGNORECASE)
        except re.error as exc:
            raise ValueError(
                f"Invalid regex in dictionary '{dictionary_name}' "
                f"at line {line_number}: {entry!r}. "
                f"Compiled pattern: {pattern!r}. "
                f"Regex error: {exc}"
            ) from exc

        for match in compiled.finditer(text):
            matches.append(
                InspectMatch(
                    match=match.group(0),
                    start=match.start(),
                    end=match.end(),
                )
            )

    return matches


def _find_dictionary_plain_matches(
    text: str,
    entries: list[str],
) -> list[InspectMatch]:
    matches: list[InspectMatch] = []

    for entry in entries:
        pattern = rf"(?<!\p{{L}}){re.escape(entry)}(?!\p{{L}})"

        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            matches.append(
                InspectMatch(
                    match=match.group(0),
                    start=match.start(),
                    end=match.end(),
                )
            )

    return matches


def _remove_exception_matches(
    positive_matches: list[InspectMatch],
    exception_matches: list[InspectMatch],
) -> tuple[list[InspectMatch], list[InspectMatch]]:
    if not exception_matches:
        return positive_matches, []

    kept = []
    discarded = []

    for positive in positive_matches:
        is_exception = any(
            positive.start >= exception.start
            and positive.end <= exception.end
            for exception in exception_matches
        )

        if is_exception:
            discarded.append(positive)
        else:
            kept.append(positive)

    return kept, discarded


def _looks_like_tagged_pos(text: str) -> bool:
    return "__(" in text


def _plain_text_from_tagged_pos(tagged_pos: str) -> str:
    words = []

    for sentence in tagged_pos.split(" || "):
        for raw_item in sentence.split(", "):
            raw_item = raw_item.strip()

            if not raw_item:
                continue

            word = raw_item.split("__(", 1)[0]
            words.append(word)

    text = " ".join(words)

    return (
        text.replace(" .", ".")
        .replace(" ,", ",")
        .replace(" ;", ";")
        .replace(" :", ":")
        .replace(" ?", "?")
        .replace(" !", "!")
    )

def _inspect_pos_tagging_dimension(
    dimension,
    text: str,
) -> DimensionInspection:
    tag = dimension.params.get("tag")
    universal = dimension.universal or dimension.params.get("universal")

    if _looks_like_tagged_pos(text):
        tagged_pos = text
        source_text = _plain_text_from_tagged_pos(tagged_pos)
    else:
        source_text = text
        annotator = StanzaAnnotator()
        doc = annotator.annotate_texts([text])[0]
        tagged_pos = format_tagged_pos(doc)

    parsed_items = _parse_tagged_pos_with_offsets(
        tagged_pos=tagged_pos,
        text=source_text,
    )

    matches = []

    for item in parsed_items:
        if _pos_item_matches(
            item=item,
            tag=tag,
            universal=universal,
        ):
            matches.append(
                InspectMatch(
                    match=item["word"],
                    start=item["start"],
                    end=item["end"],
                )
            )

    return DimensionInspection(
        key=dimension.key,
        class_name=dimension.class_name,
        pattern=None,
        dictionary=None,
        matches=matches,
        discarded_matches=[],
        debug_text=tagged_pos,
        internal_representation=tagged_pos,
    )


def _parse_tagged_pos_with_offsets(
    tagged_pos: str,
    text: str,
) -> list[dict]:
    items = []
    search_start = 0

    if not tagged_pos:
        return items

    for sentence in tagged_pos.split(" || "):
        for raw_item in sentence.split(", "):
            match = POS_ITEM_REGEX.fullmatch(raw_item.strip())

            if not match:
                continue

            word = match.group("word")
            tag = match.group("tag") or ""
            feats = match.group("feats") or ""

            start = text.find(word, search_start)

            if start == -1:
                start = text.find(word)

            if start == -1:
                continue

            end = start + len(word)
            search_start = end

            items.append(
                {
                    "word": word,
                    "tag": tag,
                    "feats": feats,
                    "start": start,
                    "end": end,
                }
            )

    return items


def _split_feats(feats: str | None) -> set[str]:
    if not feats:
        return set()

    return {
        feat.strip()
        for feat in feats.split("|")
        if feat.strip()
    }


def _pos_item_matches(
    item: dict,
    tag: str | None,
    universal: str | None,
) -> bool:
    item_tag = item["tag"]
    item_feats = _split_feats(item["feats"])
    required_feats = _split_feats(universal)

    if tag and item_tag != tag:
        return False

    if required_feats:
        return required_feats.issubset(item_feats)

    return bool(tag)
    
    
def _inspect_composite_dimension(
    config: UMUTextStatsConfig,
    dimension: DimensionConfig,
    text: str,
) -> DimensionInspection:
    matches = []
    discarded_matches = []
    internal_representation = []

    for child in dimension.children:
        try:
            child_inspection = inspect_dimension_text(
                config=config,
                key=child.key,
                text=text,
            )
        except ValueError:
            continue

        matches.extend(child_inspection.matches)

        if child_inspection.discarded_matches:
            discarded_matches.extend(child_inspection.discarded_matches)

        if child_inspection.internal_representation:
            internal_representation.append(
                f"{child.key}:\n{child_inspection.internal_representation}"
            )

    matches = _deduplicate_matches(matches)
    discarded_matches = _deduplicate_matches(discarded_matches)

    return DimensionInspection(
        key=dimension.key,
        class_name=dimension.class_name or "CompositeDimension",
        pattern=None,
        dictionary=None,
        matches=matches,
        discarded_matches=discarded_matches,
        debug_text=None,
        internal_representation="\n\n".join(internal_representation) or None,
        description=dimension.description,
    )
    
    
def _deduplicate_matches(matches: list[InspectMatch]) -> list[InspectMatch]:
    seen = set()
    unique = []

    for match in matches:
        key = (match.start, match.end, match.match)

        if key in seen:
            continue

        seen.add(key)
        unique.append(match)

    return sorted(unique, key=lambda item: (item.start, item.end, item.match))
    
    
def _filter_matches_by_pos_tag(
    matches: list[InspectMatch],
    tagged_pos: str,
    text: str,
    pos_tag: str,
) -> list[InspectMatch]:
    parsed_items = _parse_tagged_pos_with_offsets(
        tagged_pos=tagged_pos,
        text=text,
    )

    allowed_spans = {
        (item["start"], item["end"])
        for item in parsed_items
        if item["tag"] == pos_tag
    }

    return [
        match
        for match in matches
        if (match.start, match.end) in allowed_spans
    ]