from __future__ import annotations

import regex as re
from rich.console import Group
from rich.text import Text
from dataclasses import dataclass

from umutextstats.io.text import ensure_text
from umutextstats.config.explain import find_dimension
from umutextstats.dimensions.factory import build_runtime_dimension
from umutextstats.config.models import UMUTextStatsConfig
from umutextstats.text.patterns import POS_ITEM_REGEX

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


def _inspect_not_supported_dimension(
    dimension,
    text: str,
) -> DimensionInspection:
    return DimensionInspection(
        key=dimension.key,
        class_name=dimension.class_name,
        pattern=None,
        dictionary=None,
        matches=[],
        discarded_matches=[],
        debug_text=f"Inspection not implemented for {dimension.class_name}",
    )



def inspect_dimension_text(
    config: UMUTextStatsConfig,
    key: str,
    text: str,
    annotations: dict | None = None,
) -> DimensionInspection:
    text = ensure_text(text)

    explanation = find_dimension(config, key)

    if explanation is None:
        raise ValueError(f"Dimension not found: {key}")

    dimension = explanation.dimension
    if dimension.children:
        return _inspect_composite_dimension(config, dimension, text)

    runtime_dimension = build_runtime_dimension(dimension)

    tagged_pos = None
    if annotations:
        tagged_pos = annotations.get("tagged_pos")
    

    if runtime_dimension is not None and hasattr(runtime_dimension, "iter_matches"):
        return _inspect_iterable_dimension(
            dimension=dimension,
            runtime_dimension=runtime_dimension,
            text=text,
            tagged_pos=tagged_pos,
        )    


    return _inspect_not_supported_dimension(dimension, text)


def _inspect_iterable_dimension(
    dimension,
    runtime_dimension,
    text: str,
    tagged_pos: str | None = None,
) -> DimensionInspection:
    if hasattr(runtime_dimension, "iter_matches_with_context"):
        match_iter = runtime_dimension.iter_matches_with_context(
            text,
            tagged_pos=tagged_pos,
        )
    else:
        match_iter = runtime_dimension.iter_matches(text)

    matches = [
        InspectMatch(
            match=match.group(0),
            start=match.start(),
            end=match.end(),
        )
        for match in match_iter
    ]

    if hasattr(runtime_dimension, "iter_discarded_matches_with_context"):
        discarded_iter = runtime_dimension.iter_discarded_matches_with_context(
            text,
            tagged_pos=tagged_pos,
        )
    elif hasattr(runtime_dimension, "iter_discarded_matches"):
        discarded_iter = runtime_dimension.iter_discarded_matches(text)
    else:
        discarded_iter = []

    discarded_matches = [
        InspectMatch(
            match=match.group(0),
            start=match.start(),
            end=match.end(),
        )
        for match in discarded_iter
    ]

    return DimensionInspection(
        key=dimension.key,
        class_name=dimension.class_name,
        pattern=dimension.pattern,
        dictionary=(
            dimension.dictionary
            or dimension.params.get("dictionary")
            or dimension.params.get("dictionaries")
        ),
        matches=matches,
        discarded_matches=discarded_matches,
        debug_text=_inspection_debug_text(dimension),
    )


def _inspection_debug_text(dimension) -> str:
    if dimension.pattern:
        return dimension.pattern

    dictionary = (
        dimension.dictionary
        or dimension.params.get("dictionary")
        or dimension.params.get("dictionaries")
    )

    if dictionary:
        return f"Loaded dictionary: {dictionary}"

    return f"Inspection not implemented for {dimension.class_name}"


def _inspect_paragraph_count_dimension(dimension, text: str) -> DimensionInspection:
    matches = []

    return DimensionInspection(
        key=dimension.key,
        class_name=dimension.class_name,
        pattern=None,
        dictionary=None,
        matches=matches,
        discarded_matches=[],
        debug_text="paragraphs separated by one or more blank lines",
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
    pos_tag: str | list[str],
) -> list[InspectMatch]:
    parsed_items = _parse_tagged_pos_with_offsets(
        tagged_pos=tagged_pos,
        text=text,
    )

    allowed_tags = (
        pos_tag
        if isinstance(pos_tag, list)
        else [pos_tag]
    )

    allowed_spans = {
        (item["start"], item["end"])
        for item in parsed_items
        if item["tag"] in allowed_tags
    }

    return [
        match
        for match in matches
        if (match.start, match.end) in allowed_spans
    ]

