# src/umutextstats/vectorization/jsonl.py

from __future__ import annotations

import json
from pathlib import Path
from typing import TextIO

import pandas as pd


SUPPORTED_SCHEMA_VERSIONS = {
    "1.0",
}


def read_structured_jsonl(
    path: str | Path,
) -> pd.DataFrame:
    """
    Read a structured UMUTextStats JSONL file and flatten its global
    dimension values into a DataFrame.
    """
    path = Path(path)

    with path.open(
        "r",
        encoding="utf-8",
    ) as stream:
        return read_structured_jsonl_stream(stream)


def read_structured_jsonl_stream(
    stream: TextIO,
) -> pd.DataFrame:
    """
    Read structured UMUTextStats JSONL from an open text stream.

    The metadata record defines the preferred dimension order. Document
    records are converted into one DataFrame row each.
    """
    metadata = None
    dimension_order: list[str] = []
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

    columns.extend(dimension_order)

    rows = []

    for document in documents:
        row = {}

        if has_id:
            row["id"] = document.get("id")

        for key in dimension_order:
            row[key] = document.get(key)

        rows.append(row)

    return pd.DataFrame(
        rows,
        columns=columns,
    )


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