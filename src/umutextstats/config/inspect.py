from __future__ import annotations

from typing import Any

import pandas as pd
from rich.console import Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from umutextstats.config.explain import find_dimension
from umutextstats.config.models import UMUTextStatsConfig
from umutextstats.dimensions.factory import build_runtime_dimension
from umutextstats.evidence import (
    EvidenceOccurrence,
    PositionalDistribution,
    build_positional_distribution,
)
from umutextstats.inspection.models import (
    DimensionInspection,
    InspectMatch,
)
from umutextstats.io.text import ensure_text


_SPARK_LEVELS = "▁▂▃▄▅▆▇█"


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
        debug_text=(
            f"Inspection not implemented for "
            f"{dimension.class_name}"
        ),
    )


def inspect_dimension_text(
    config: UMUTextStatsConfig,
    key: str,
    text: str,
    annotations: dict | None = None,
) -> DimensionInspection:
    """
    Inspect a configured dimension for a single text.

    The text and optional annotations are converted into a pandas Series,
    which is the single-row runtime context used by dimensions.
    """
    text = ensure_text(text)

    explanation = find_dimension(
        config,
        key,
    )

    if explanation is None:
        raise ValueError(
            f"Dimension not found: {key}"
        )

    dimension = explanation.dimension
    runtime_dimension = build_runtime_dimension(
        dimension
    )

    row_data = {
        "text": text,
        "text_raw": text,
        "text_norm": text,
    }

    if annotations:
        row_data.update(
            annotations
        )

    row = pd.Series(
        row_data
    )

    if runtime_dimension is not None:
        inspection = runtime_dimension.inspect(
            row
        )

        if inspection is not None:
            return inspection

    return _inspect_not_supported_dimension(
        dimension=dimension,
        text=text,
    )


def build_match_distribution(
    inspection: DimensionInspection,
    text: str,
    annotations: dict[str, Any] | None = None,
    segments: int = 3,
) -> tuple[PositionalDistribution, str]:
    """
    Adapt inspection matches to the common evidence model.

    Returns the positional distribution and the reference text used
    by the evidence offsets.
    """
    reference_text = resolve_inspection_reference(
        inspection=inspection,
        text=text,
        annotations=annotations,
    )

    occurrences = [
        EvidenceOccurrence(
            label=match.match,
            start=match.start,
            end=match.end,
            offset_source=inspection.offset_source,
            offset_unit=inspection.offset_unit,
        )
        for match in inspection.matches
    ]

    distribution = build_positional_distribution(
        occurrences=occurrences,
        reference_length=len(reference_text),
        segments=segments,
        offset_source=(
            inspection.offset_source
            or "text"
        ),
        offset_unit=inspection.offset_unit,
    )

    return distribution, reference_text


def render_inspection(
    inspection: DimensionInspection,
    text: str,
):
    lines = [
        Text(
            f"Key: {inspection.key}",
            style="bold",
        ),
    ]

    if inspection.class_name:
        lines.append(
            Text(
                f"Class: "
                f"{inspection.class_name}"
            )
        )

    if inspection.pattern:
        lines.append(
            Text(
                f"Pattern: "
                f"{inspection.pattern}"
            )
        )

    if inspection.dictionary:
        lines.append(
            Text(
                f"Dictionary: "
                f"{inspection.dictionary}"
            )
        )

    lines.append(
        Text("")
    )

    lines.append(
        Text(
            f"Matches: "
            f"{len(inspection.matches)}"
        )
    )

    for match in inspection.matches:
        lines.append(
            Text(
                f"  - {match.match} "
                f"[{match.start}:{match.end}]"
            )
        )

    discarded_matches = (
        inspection.discarded_matches
        or []
    )

    if discarded_matches:
        lines.append(
            Text("")
        )

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
        lines.append(
            Text("")
        )

        lines.append(
            Text(
                "Internal representation:",
                style="bold",
            )
        )

        lines.append(
            Text(
                inspection.debug_text
            )
        )

    if inspection.matches:
        lines.append(
            Text("")
        )

        lines.append(
            Text(
                "Highlighted:",
                style="bold",
            )
        )

        lines.append(
            highlight_matches(
                text,
                inspection.matches,
            )
        )

    return Group(
        *lines
    )


def build_distribution_sparkline(
    distribution: PositionalDistribution,
) -> str:
    """
    Build a compact Unicode sparkline from segment counts.
    """
    counts = [
        segment.count
        for segment in distribution.segments
    ]

    if not counts:
        return ""

    maximum = max(counts)

    if maximum == 0:
        return _SPARK_LEVELS[0] * len(counts)

    last_level = len(_SPARK_LEVELS) - 1

    characters: list[str] = []

    for count in counts:
        if count == 0:
            level = 0
        else:
            level = round(
                count
                / maximum
                * last_level
            )

        characters.append(
            _SPARK_LEVELS[level]
        )

    return "".join(
        characters
    )


def render_match_distribution(
    distribution: PositionalDistribution,
):
    """
    Render a positional distribution as a Rich table.
    """
    table = Table(
        title="Positional distribution",
    )

    table.add_column(
        "Segment",
        justify="right",
    )

    table.add_column(
        "Range",
    )

    table.add_column(
        "Offsets",
    )

    table.add_column(
        "Matches",
        justify="right",
    )

    table.add_column(
        "Share",
        justify="right",
    )

    table.add_column(
        "Distribution",
    )

    bar_width = 20

    for segment in distribution.segments:
        start_percent = (
            segment.start_ratio
            * 100
        )

        end_percent = (
            segment.end_ratio
            * 100
        )

        if segment.share is None:
            share_text = "—"
            bar = ""
        else:
            share_text = (
                f"{segment.share:.1%}"
            )

            filled = round(
                segment.share
                * bar_width
            )

            bar = (
                "█" * filled
            )

        table.add_row(
            str(
                segment.index + 1
            ),
            (
                f"{start_percent:.0f}–"
                f"{end_percent:.0f}%"
            ),
            (
                f"{segment.start}:"
                f"{segment.end}"
            ),
            str(
                segment.count
            ),
            share_text,
            bar,
        )

    delta_text = (
        f"{distribution.delta:+.3f}"
        if distribution.delta is not None
        else "N/A"
    )

    reference = Text(
        "Positional reference: "
        f"{distribution.offset_source or 'unknown'} "
        f"({distribution.offset_unit})"
    )

    profile = Text()

    profile.append(
        "Profile: ",
        style="bold",
    )

    profile.append(
        build_distribution_sparkline(
            distribution
        ),
        style="cyan",
    )

    return Group(
        reference,
        Text(""),
        table,
        Text(""),
        profile,
        Text(
            "Delta final - initial: "
            f"{delta_text}"
        ),
    )


def render_segmented_text(
    distribution: PositionalDistribution,
    text: str,
    matches: list[InspectMatch],
):
    """
    Render the inspected reference text divided into relative segments.

    Match highlighting is preserved inside each segment when offsets
    are expressed in characters.
    """
    if distribution.offset_unit != "characters":
        return Group(
            Text(
                "Segmented text",
                style="bold",
            ),
            Text(""),
            Text(
                "Segmented text is unavailable for "
                f"offset unit '{distribution.offset_unit}'."
            ),
        )

    items = [
        Text(
            "Segmented text",
            style="bold",
        ),
        Text(""),
    ]

    for segment in distribution.segments:
        start_percent = (
            segment.start_ratio
            * 100
        )

        end_percent = (
            segment.end_ratio
            * 100
        )

        segment_text = Text(
            text[
                segment.start:
                segment.end
            ]
        )

        for match in matches:
            overlap_start = max(
                match.start,
                segment.start,
            )

            overlap_end = min(
                match.end,
                segment.end,
            )

            if overlap_start >= overlap_end:
                continue

            local_start = (
                overlap_start
                - segment.start
            )

            local_end = (
                overlap_end
                - segment.start
            )

            segment_text.stylize(
                "bold red",
                local_start,
                local_end,
            )

        title = (
            f"Segment {segment.index + 1} · "
            f"{start_percent:.0f}–"
            f"{end_percent:.0f}% · "
            f"{segment.count} matches"
        )

        items.append(
            Panel(
                segment_text,
                title=title,
                subtitle=(
                    f"Offsets "
                    f"{segment.start}:"
                    f"{segment.end}"
                ),
                expand=False,
            )
        )

        if (
            segment.index
            < len(distribution.segments) - 1
        ):
            items.append(
                Text("")
            )

    return Group(
        *items
    )


def highlight_matches(
    text: str,
    matches: list[InspectMatch],
) -> Text:
    highlighted = Text(
        text
    )

    for match in matches:
        highlighted.stylize(
            "bold red",
            match.start,
            match.end,
        )

    return highlighted


def resolve_inspection_reference(
    inspection: DimensionInspection,
    text: str,
    annotations: dict[str, Any] | None = None,
) -> str:
    """
    Resolve the representation whose coordinate system is used by
    the inspection matches.
    """
    offset_source = (
        inspection.offset_source
        or "text"
    )

    if offset_source in {
        "text",
        "text_raw",
        "text_norm",
    }:
        return ensure_text(
            text
        )

    if annotations is None:
        raise ValueError(
            "Positional reference "
            f"'{offset_source}' requires annotations."
        )

    reference = annotations.get(
        offset_source
    )

    if reference is None:
        available = ", ".join(
            sorted(
                annotations
            )
        )

        raise ValueError(
            "Positional reference "
            f"'{offset_source}' is not available. "
            f"Available annotations: {available}"
        )

    return ensure_text(
        reference
    )