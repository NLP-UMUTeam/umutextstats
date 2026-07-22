from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO


@dataclass
class StructuredExtractionData:
    """
    Structured extraction records loaded from JSONL.
    """

    metadata: dict[str, Any]
    documents: list[dict[str, Any]]


def read_structured_jsonl(
    source: str | Path | TextIO,
) -> StructuredExtractionData:
    """
    Read a structured extraction JSONL source.

    The source must contain exactly one metadata record and zero or more
    document records.
    """
    if hasattr(source, "read"):
        return _read_structured_jsonl_stream(
            source
        )

    path = Path(source)

    with path.open(
        "r",
        encoding="utf-8",
    ) as stream:
        return _read_structured_jsonl_stream(
            stream
        )


def _read_structured_jsonl_stream(
    stream: TextIO,
) -> StructuredExtractionData:
    """
    Read structured extraction records from an open text stream.
    """
    metadata: dict[str, Any] | None = None
    documents: list[dict[str, Any]] = []

    for line_number, line in enumerate(
        stream,
        start=1,
    ):
        line = line.strip()

        if not line:
            continue

        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Invalid JSON at line {line_number}: {exc}"
            ) from exc

        if not isinstance(record, dict):
            raise ValueError(
                f"JSONL record at line {line_number} "
                "must be an object."
            )

        record_type = record.get("_type")

        if record_type == "metadata":
            if metadata is not None:
                raise ValueError(
                    "Structured JSONL contains more than one "
                    "metadata record."
                )

            metadata = record
            continue

        if record_type == "document":
            documents.append(record)
            continue

        raise ValueError(
            f"Unknown structured JSONL record type at "
            f"line {line_number}: {record_type!r}"
        )

    if metadata is None:
        raise ValueError(
            "Structured JSONL does not contain a metadata record."
        )

    dimensions = metadata.get(
        "dimensions",
        [],
    )

    if not isinstance(dimensions, list):
        raise ValueError(
            "Structured JSONL metadata field "
            "'dimensions' must be a list."
        )

    dimension_metadata = metadata.get(
        "dimension_metadata",
        {},
    )

    if not isinstance(
        dimension_metadata,
        dict,
    ):
        raise ValueError(
            "Structured JSONL metadata field "
            "'dimension_metadata' must be a mapping."
        )

    return StructuredExtractionData(
        metadata=metadata,
        documents=documents,
    )