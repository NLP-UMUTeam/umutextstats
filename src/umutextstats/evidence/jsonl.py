from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from umutextstats.evidence.distribution import (
    build_positional_distribution,
)
from umutextstats.evidence.models import (
    EvidenceOccurrence,
    PositionalDistribution,
)


def evidence_occurrences_from_dimension_record(
    dimension_record: Mapping[str, Any],
    *,
    offset_source: str | None = None,
    offset_unit: str = "characters",
) -> list[EvidenceOccurrence]:
    """
    Convert a structured JSONL dimension record into occurrences.

    Expected dimension record:

    {
        "value": ...,
        "numerator": ...,
        "denominator": ...,
        "evidence": [
            {
                "text": "...",
                "start": 10,
                "end": 15
            }
        ]
    }

    Missing or null evidence is treated as an empty list.
    """
    evidence = dimension_record.get(
        "evidence"
    )

    if evidence is None:
        return []

    if not isinstance(
        evidence,
        Sequence,
    ) or isinstance(
        evidence,
        (str, bytes),
    ):
        raise ValueError(
            "Dimension evidence must be a list."
        )

    occurrences: list[
        EvidenceOccurrence
    ] = []

    for index, item in enumerate(
        evidence
    ):
        if not isinstance(
            item,
            Mapping,
        ):
            raise ValueError(
                "Evidence item at position "
                f"{index} must be an object."
            )

        occurrences.append(
            evidence_occurrence_from_record(
                item,
                offset_source=offset_source,
                offset_unit=offset_unit,
                position=index,
            )
        )

    return occurrences


def evidence_occurrence_from_record(
    record: Mapping[str, Any],
    *,
    offset_source: str | None = None,
    offset_unit: str = "characters",
    position: int | None = None,
) -> EvidenceOccurrence:
    """
    Convert one JSONL evidence object into an EvidenceOccurrence.
    """
    prefix = (
        f"Evidence item at position {position}"
        if position is not None
        else "Evidence item"
    )

    if "start" not in record:
        raise ValueError(
            f"{prefix} has no 'start' field."
        )

    if "end" not in record:
        raise ValueError(
            f"{prefix} has no 'end' field."
        )

    start = _read_integer_offset(
        record["start"],
        field="start",
        prefix=prefix,
    )

    end = _read_integer_offset(
        record["end"],
        field="end",
        prefix=prefix,
    )

    label = _read_evidence_label(
        record
    )

    item_offset_source = record.get(
        "offset_source",
        offset_source,
    )

    item_offset_unit = record.get(
        "offset_unit",
        offset_unit,
    )

    if (
        item_offset_source is not None
        and not isinstance(
            item_offset_source,
            str,
        )
    ):
        raise ValueError(
            f"{prefix} field 'offset_source' "
            "must be a string or null."
        )

    if not isinstance(
        item_offset_unit,
        str,
    ):
        raise ValueError(
            f"{prefix} field 'offset_unit' "
            "must be a string."
        )

    return EvidenceOccurrence(
        label=label,
        start=start,
        end=end,
        offset_source=item_offset_source,
        offset_unit=item_offset_unit,
    )


def dimension_position_metadata(
    metadata_record: Mapping[str, Any],
    dimension_key: str,
) -> tuple[str | None, str]:
    """
    Read positional metadata for a dimension from the metadata record.

    Returns:

        (offset_source, offset_unit)
    """
    dimension_metadata = metadata_record.get(
        "dimension_metadata",
        {},
    )

    if not isinstance(
        dimension_metadata,
        Mapping,
    ):
        raise ValueError(
            "Structured JSONL metadata field "
            "'dimension_metadata' must be a mapping."
        )

    metadata = dimension_metadata.get(
        dimension_key,
        {},
    )

    if not isinstance(
        metadata,
        Mapping,
    ):
        raise ValueError(
            "Metadata for dimension "
            f"{dimension_key!r} must be a mapping."
        )

    offset_source = metadata.get(
        "offset_source"
    )

    if offset_source is None:
        # Backwards-compatible fallback. Older JSONL files may only
        # contain the configured input column.
        offset_source = metadata.get(
            "input_column"
        )

    offset_unit = metadata.get(
        "offset_unit",
        "characters",
    )

    if (
        offset_source is not None
        and not isinstance(
            offset_source,
            str,
        )
    ):
        raise ValueError(
            "Dimension metadata field "
            "'offset_source' must be a string or null."
        )

    if not isinstance(
        offset_unit,
        str,
    ):
        raise ValueError(
            "Dimension metadata field "
            "'offset_unit' must be a string."
        )

    return (
        offset_source,
        offset_unit,
    )


def dimension_occurrences_from_document(
    metadata_record: Mapping[str, Any],
    document_record: Mapping[str, Any],
    dimension_key: str,
) -> list[EvidenceOccurrence]:
    """
    Extract one dimension's evidence from a JSONL document record.

    Positional metadata is obtained from the JSONL metadata record.
    """
    dimensions = document_record.get(
        "dimensions",
        {},
    )

    if not isinstance(
        dimensions,
        Mapping,
    ):
        raise ValueError(
            "Structured document field "
            "'dimensions' must be a mapping."
        )

    dimension_record = dimensions.get(
        dimension_key
    )

    if dimension_record is None:
        raise ValueError(
            f"Dimension {dimension_key!r} "
            "is not present in the document record."
        )

    if not isinstance(
        dimension_record,
        Mapping,
    ):
        raise ValueError(
            f"Dimension record {dimension_key!r} "
            "must be a mapping."
        )

    offset_source, offset_unit = (
        dimension_position_metadata(
            metadata_record,
            dimension_key,
        )
    )

    return evidence_occurrences_from_dimension_record(
        dimension_record,
        offset_source=offset_source,
        offset_unit=offset_unit,
    )


def dimension_distribution_from_document(
    metadata_record: Mapping[str, Any],
    document_record: Mapping[str, Any],
    dimension_key: str,
    *,
    reference_length: int,
    segments: int = 3,
) -> PositionalDistribution:
    """
    Build a positional distribution for one JSONL document dimension.

    `reference_length` must correspond to the representation named by
    the dimension's offset_source, such as text_norm or tagged_pos.
    """
    offset_source, offset_unit = (
        dimension_position_metadata(
            metadata_record,
            dimension_key,
        )
    )

    occurrences = dimension_occurrences_from_document(
        metadata_record=metadata_record,
        document_record=document_record,
        dimension_key=dimension_key,
    )

    return build_positional_distribution(
        occurrences=occurrences,
        reference_length=reference_length,
        segments=segments,
        offset_source=offset_source,
        offset_unit=offset_unit,
    )


def _read_integer_offset(
    value: Any,
    *,
    field: str,
    prefix: str,
) -> int:
    """
    Validate and normalize one evidence offset.
    """
    if isinstance(
        value,
        bool,
    ):
        raise ValueError(
            f"{prefix} field {field!r} "
            "must be an integer."
        )

    try:
        result = int(
            value
        )
    except (
        TypeError,
        ValueError,
    ) as exc:
        raise ValueError(
            f"{prefix} field {field!r} "
            "must be an integer."
        ) from exc

    if result != value:
        raise ValueError(
            f"{prefix} field {field!r} "
            "must be an integer."
        )

    return result


def _read_evidence_label(
    record: Mapping[str, Any],
) -> str | None:
    """
    Read the most common evidence label fields.
    """
    for field in (
        "label",
        "text",
        "match",
        "word",
    ):
        value = record.get(
            field
        )

        if value is not None:
            return str(
                value
            )

    return None