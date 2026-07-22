from __future__ import annotations

import pandas as pd
from rich.console import Group
from rich.table import Table
from rich.text import Text
from rich.table import Table
from rich.panel import Panel

from umutextstats.config.explain import find_dimension
from umutextstats.config.models import UMUTextStatsConfig
from umutextstats.dimensions.factory import build_runtime_dimension
from umutextstats.inspection.models import (
    DimensionInspection,
    InspectDistribution,
    InspectMatch,
    InspectSegment,
)
from umutextstats.io.text import ensure_text


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
    segments: int = 3,
) -> InspectDistribution:
    """
    Distribute inspected matches across relative text segments.

    Matches are assigned using their midpoint character offset.
    """
    if segments < 1:
        raise ValueError(
            "segments must be greater than or equal to 1"
        )

    text = ensure_text(
        text
    )

    text_length = len(
        text
    )

    matches = inspection.matches or []
    total_matches = len(
        matches
    )

    counts = [
        0
        for _ in range(segments)
    ]

    if text_length > 0:
        for match in matches:
            midpoint = (
                match.start
                + match.end
            ) / 2

            relative_position = (
                midpoint
                / text_length
            )

            segment_index = min(
                int(
                    relative_position
                    * segments
                ),
                segments - 1,
            )

            segment_index = max(
                segment_index,
                0,
            )

            counts[
                segment_index
            ] += 1

    segment_items: list[
        InspectSegment
    ] = []

    for index, count in enumerate(
        counts
    ):
        start_ratio = (
            index
            / segments
        )

        end_ratio = (
            (index + 1)
            / segments
        )

        start = round(
            start_ratio
            * text_length
        )

        end = round(
            end_ratio
            * text_length
        )

        share = (
            count
            / total_matches
            if total_matches > 0
            else None
        )

        segment_items.append(
            InspectSegment(
                index=index,
                start=start,
                end=end,
                start_ratio=start_ratio,
                end_ratio=end_ratio,
                count=count,
                share=share,
            )
        )

    delta = None

    if total_matches > 0:
        first_share = (
            segment_items[0].share
            or 0.0
        )

        last_share = (
            segment_items[-1].share
            or 0.0
        )

        delta = (
            last_share
            - first_share
        )

    return InspectDistribution(
        text_length=text_length,
        total_matches=total_matches,
        segments=segment_items,
        delta=delta,
    )


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


def render_match_distribution(
    distribution: InspectDistribution,
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

    return Group(
        table,
        Text(""),
        Text(
            "Delta final - initial: "
            f"{delta_text}"
        ),
    )

def render_segmented_text(
    distribution: InspectDistribution,
    text: str,
    matches: list[InspectMatch],
):
    """
    Render the inspected text divided into relative segments.

    Match highlighting is preserved inside each segment.
    """
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