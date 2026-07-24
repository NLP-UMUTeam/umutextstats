from __future__ import annotations

import json
from pathlib import Path
from typing import TextIO

import pandas as pd

from umutextstats.evidence.jsonl import (
    dimension_distribution_from_document,
    dimension_evidence_descriptor,
)


SUPPORTED_SCHEMA_VERSIONS = {
    "1.0",
    "1.1",
}


def read_structured_jsonl(
    path: str | Path,
    *,
    position_dimensions: list[str] | None = None,
    position_segments: int = 5,
) -> pd.DataFrame:
    """
    Read a structured UMUTextStats JSONL file.

    Global dimension values are flattened into ordinary columns.
    Optionally, positional features are generated for selected
    dimensions in the same pass.
    """
    path = Path(path)

    with path.open(
        "r",
        encoding="utf-8",
    ) as stream:
        return read_structured_jsonl_stream(
            stream,
            position_dimensions=position_dimensions,
            position_segments=position_segments,
        )


def read_structured_jsonl_stream(
    stream: TextIO,
    *,
    position_dimensions: list[str] | None = None,
    position_segments: int = 5,
) -> pd.DataFrame:
    """
    Read structured UMUTextStats JSONL from an open text stream.

    The metadata record defines the preferred global dimension order.
    Document records are converted into one DataFrame row each.

    When positional dimensions are requested, each selected dimension
    produces:

    - ``<dimension>__position_total``
    - ``<dimension>__position_share_1``
    - ...
    - ``<dimension>__position_share_N``
    """
    requested_positions = _normalize_position_dimensions(
        position_dimensions
    )

    if position_segments <= 0:
        raise ValueError(
            "position_segments must be greater than zero"
        )

    metadata = None
    dimension_order: list[str] = []
    positional_columns: list[str] = []
    documents: list[dict] = []
    has_id = False

    for line_number, raw_line in enumerate(
        stream,
        start=1,
    ):
        line = raw_line.strip()

        if not line:
            continue

        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Invalid JSON on line {line_number}: {exc}"
            ) from exc

        if not isinstance(record, dict):
            raise ValueError(
                f"Expected a JSON object on line {line_number}"
            )

        record_type = record.get("_type")

        if record_type == "metadata":
            if metadata is not None:
                raise ValueError(
                    "Structured JSONL contains more than one "
                    "metadata record"
                )

            metadata = record

            dimension_order = _parse_metadata_dimensions(
                metadata=metadata,
                line_number=line_number,
            )

            _validate_position_dimensions(
                metadata=metadata,
                dimension_order=dimension_order,
                position_dimensions=requested_positions,
                line_number=line_number,
            )

            positional_columns = _build_position_columns(
                dimensions=requested_positions,
                segments=position_segments,
            )

            continue

        if record_type == "document":
            if metadata is None:
                raise ValueError(
                    "Document record found before metadata "
                    f"on line {line_number}"
                )

            document = _parse_document(
                record=record,
                line_number=line_number,
                dimension_order=dimension_order,
            )

            if "id" in record:
                document["id"] = record["id"]
                has_id = True

            positional_values = _parse_document_positions(
                metadata=metadata,
                record=record,
                line_number=line_number,
                position_dimensions=requested_positions,
                position_segments=position_segments,
            )

            document.update(
                positional_values
            )

            documents.append(document)
            continue

        raise ValueError(
            f"Unknown record type on line {line_number}: "
            f"{record_type!r}"
        )

    if metadata is None:
        raise ValueError(
            "Structured JSONL does not contain a metadata record"
        )

    columns = []

    if has_id:
        columns.append("id")

    columns.extend(
        dimension_order
    )

    columns.extend(
        positional_columns
    )

    rows = []

    for document in documents:
        row = {}

        if has_id:
            row["id"] = document.get("id")

        for key in dimension_order:
            row[key] = document.get(key)

        for key in positional_columns:
            row[key] = document.get(key)

        rows.append(row)

    return pd.DataFrame(
        rows,
        columns=columns,
    )


def _normalize_position_dimensions(
    dimensions: list[str] | None,
) -> list[str]:
    """
    Normalize and deduplicate requested positional dimensions.
    """
    if not dimensions:
        return []

    normalized = []

    for dimension in dimensions:
        if not isinstance(dimension, str):
            raise TypeError(
                "Position dimension keys must be strings"
            )

        key = dimension.strip()

        if not key:
            raise ValueError(
                "Position dimension keys cannot be empty"
            )

        if key not in normalized:
            normalized.append(key)

    return normalized


def _validate_position_dimensions(
    *,
    metadata: dict,
    dimension_order: list[str],
    position_dimensions: list[str],
    line_number: int,
) -> None:
    """
    Ensure requested dimensions exist and have evidence descriptors.
    """
    for dimension_key in position_dimensions:
        if dimension_key not in dimension_order:
            raise ValueError(
                f"Position dimension {dimension_key!r} "
                f"is not declared in metadata "
                f"on line {line_number}"
            )

        try:
            dimension_evidence_descriptor(
                metadata,
                dimension_key,
            )
        except (
            KeyError,
            TypeError,
            ValueError,
        ) as exc:
            raise ValueError(
                f"Position dimension {dimension_key!r} "
                "does not provide usable positional evidence: "
                f"{exc}"
            ) from exc


def _build_position_columns(
    *,
    dimensions: list[str],
    segments: int,
) -> list[str]:
    """
    Build deterministic positional feature column names.
    """
    columns = []

    for dimension_key in dimensions:
        columns.append(
            f"{dimension_key}__position_total"
        )

        for segment_index in range(
            1,
            segments + 1,
        ):
            columns.append(
                f"{dimension_key}"
                f"__position_share_{segment_index}"
            )

    return columns


def _parse_document_positions(
    *,
    metadata: dict,
    record: dict,
    line_number: int,
    position_dimensions: list[str],
    position_segments: int,
) -> dict[str, float | int]:
    """
    Build positional ML features for one document.
    """
    values: dict[str, float | int] = {}

    for dimension_key in position_dimensions:
        try:
            distribution = (
                dimension_distribution_from_document(
                    metadata_record=metadata,
                    document_record=record,
                    dimension_key=dimension_key,
                    segments=position_segments,
                )
            )
        except (
            KeyError,
            TypeError,
            ValueError,
        ) as exc:
            raise ValueError(
                f"Could not vectorize positional dimension "
                f"{dimension_key!r} on line {line_number}: {exc}"
            ) from exc

        values[
            f"{dimension_key}__position_total"
        ] = distribution.total_occurrences

        for segment_number, segment in enumerate(
            distribution.segments,
            start=1,
        ):
            values[
                f"{dimension_key}"
                f"__position_share_{segment_number}"
            ] = (
                float(segment.share)
                if segment.share is not None
                else 0.0
            )

    return values


def _parse_metadata_dimensions(
    metadata: dict,
    line_number: int,
) -> list[str]:
    """
    Validate metadata and return the configured dimension order.
    """
    schema_version = metadata.get("schema_version")

    if schema_version not in SUPPORTED_SCHEMA_VERSIONS:
        raise ValueError(
            f"Unsupported schema version on line {line_number}: "
            f"{schema_version!r}"
        )

    dimensions = metadata.get("dimensions")

    if not isinstance(dimensions, list):
        raise ValueError(
            f"Metadata dimensions must be a list "
            f"on line {line_number}"
        )

    dimension_order = []

    for key in dimensions:
        if not isinstance(key, str) or not key:
            raise ValueError(
                f"Invalid dimension key in metadata "
                f"on line {line_number}: {key!r}"
            )

        if key in dimension_order:
            raise ValueError(
                f"Duplicated dimension key in metadata: {key!r}"
            )

        dimension_order.append(key)

    return dimension_order


def _parse_document(
    record: dict,
    line_number: int,
    dimension_order: list[str],
) -> dict:
    """
    Extract global dimension values from one document record.
    """
    dimensions = record.get("dimensions")

    if not isinstance(dimensions, dict):
        raise ValueError(
            f"Document dimensions must be an object "
            f"on line {line_number}"
        )

    unknown_dimensions = [
        key
        for key in dimensions
        if key not in dimension_order
    ]

    if unknown_dimensions:
        formatted = ", ".join(
            repr(key)
            for key in unknown_dimensions
        )

        raise ValueError(
            f"Document on line {line_number} contains dimensions "
            f"not declared in metadata: {formatted}"
        )

    values = {}

    for key in dimension_order:
        if key not in dimensions:
            values[key] = None
            continue

        dimension = dimensions[key]

        if not isinstance(dimension, dict):
            raise ValueError(
                f"Dimension {key!r} on line {line_number} "
                "must be an object"
            )

        values[key] = dimension.get("value")

    return values