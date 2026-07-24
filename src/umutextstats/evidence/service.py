from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from umutextstats.evidence.aggregation import (
    AggregatedPositionalDistribution,
    aggregate_positional_distributions,
)
from umutextstats.evidence.descriptors import (
    EvidenceDescriptor,
)
from umutextstats.evidence.jsonl import (
    dimension_distribution_from_document,
    dimension_evidence_descriptor,
)
from umutextstats.evidence.models import (
    PositionalDistribution,
)


@dataclass(frozen=True)
class PositionalAggregationResult:
    """
    Result of aggregating one positional dimension from JSONL.
    """

    dimension_key: str
    descriptor: EvidenceDescriptor
    segments: int
    document_count: int
    distribution: AggregatedPositionalDistribution


def aggregate_dimension_from_jsonl(
    path: str | Path,
    dimension_key: str,
    *,
    segments: int = 3,
) -> PositionalAggregationResult:
    """
    Aggregate one positional dimension from a structured JSONL file.

    The JSONL file must contain:

    - a metadata record as its first non-empty line;
    - document records after the metadata;
    - an evidence descriptor for the selected dimension;
    - the reference length required by that descriptor.

    Parameters
    ----------
    path:
        Structured UMUTextStats JSONL file.
    dimension_key:
        Dimension whose evidence will be distributed positionally.
    segments:
        Number of equal-width positional segments.

    Returns
    -------
    PositionalAggregationResult
        Metadata and aggregated positional distribution.

    Raises
    ------
    ValueError
        If the file is malformed, the dimension is unknown, the dimension
        has no positional evidence descriptor, or a document is inconsistent.
    """
    input_path = Path(path)

    if segments <= 0:
        raise ValueError(
            "segments must be greater than zero"
        )

    with input_path.open(
        encoding="utf-8",
    ) as stream:
        metadata, metadata_line_number = (
            _read_first_record(
                stream,
                path=input_path,
            )
        )

        _validate_metadata_record(
            metadata,
            path=input_path,
            line_number=metadata_line_number,
        )

        descriptor = dimension_evidence_descriptor(
            metadata,
            dimension_key,
        )

        distributions = list(
            _iter_dimension_distributions(
                stream=stream,
                metadata=metadata,
                dimension_key=dimension_key,
                segments=segments,
                path=input_path,
                start_line_number=(
                    metadata_line_number + 1
                ),
            )
        )

    aggregated = aggregate_positional_distributions(
        distributions
    )

    return PositionalAggregationResult(
        dimension_key=dimension_key,
        descriptor=descriptor,
        segments=segments,
        document_count=len(distributions),
        distribution=aggregated,
    )


def iter_dimension_distributions_from_jsonl(
    path: str | Path,
    dimension_key: str,
    *,
    segments: int = 3,
) -> Iterator[PositionalDistribution]:
    """
    Yield document-level positional distributions from JSONL.

    This function is useful when callers want to process each document
    separately instead of immediately aggregating the entire corpus.
    """
    input_path = Path(path)

    if segments <= 0:
        raise ValueError(
            "segments must be greater than zero"
        )

    with input_path.open(
        encoding="utf-8",
    ) as stream:
        metadata, metadata_line_number = (
            _read_first_record(
                stream,
                path=input_path,
            )
        )

        _validate_metadata_record(
            metadata,
            path=input_path,
            line_number=metadata_line_number,
        )

        # Validate the requested dimension before starting iteration.
        dimension_evidence_descriptor(
            metadata,
            dimension_key,
        )

        yield from _iter_dimension_distributions(
            stream=stream,
            metadata=metadata,
            dimension_key=dimension_key,
            segments=segments,
            path=input_path,
            start_line_number=(
                metadata_line_number + 1
            ),
        )


def _iter_dimension_distributions(
    *,
    stream,
    metadata: dict[str, Any],
    dimension_key: str,
    segments: int,
    path: Path,
    start_line_number: int,
) -> Iterator[PositionalDistribution]:
    for line_number, line in enumerate(
        stream,
        start=start_line_number,
    ):
        if not line.strip():
            continue

        document = _read_json_record(
            line,
            path=path,
            line_number=line_number,
        )

        try:
            yield dimension_distribution_from_document(
                metadata_record=metadata,
                document_record=document,
                dimension_key=dimension_key,
                segments=segments,
            )
        except (
            KeyError,
            TypeError,
            ValueError,
        ) as exc:
            raise ValueError(
                f"{path}:{line_number}: "
                f"could not build positional distribution "
                f"for dimension {dimension_key!r}: {exc}"
            ) from exc


def _read_first_record(
    stream,
    *,
    path: Path,
) -> tuple[dict[str, Any], int]:
    for line_number, line in enumerate(
        stream,
        start=1,
    ):
        if not line.strip():
            continue

        return (
            _read_json_record(
                line,
                path=path,
                line_number=line_number,
            ),
            line_number,
        )

    raise ValueError(
        f"{path}: JSONL file is empty"
    )


def _read_json_record(
    line: str,
    *,
    path: Path,
    line_number: int,
) -> dict[str, Any]:
    try:
        record = json.loads(
            line
        )
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"{path}:{line_number}: invalid JSON: {exc}"
        ) from exc

    if not isinstance(
        record,
        dict,
    ):
        raise ValueError(
            f"{path}:{line_number}: "
            "JSONL record must be an object"
        )

    return record


def _validate_metadata_record(
    metadata: dict[str, Any],
    *,
    path: Path,
    line_number: int,
) -> None:
    if metadata.get("_type") != "metadata":
        raise ValueError(
            f"{path}:{line_number}: "
            "first JSONL record must have "
            "'_type' equal to 'metadata'"
        )