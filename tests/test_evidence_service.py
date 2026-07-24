from __future__ import annotations

import json
from pathlib import Path

import pytest

from umutextstats.evidence.descriptors import (
    EvidenceDescriptor,
)
from umutextstats.evidence.service import (
    aggregate_dimension_from_jsonl,
    iter_dimension_distributions_from_jsonl,
)


DIMENSION_KEY = "lexical-negative"


def segment_counts(
    distribution,
) -> list[int]:
    return [
        segment.count
        for segment in distribution.segments
    ]


def write_jsonl(
    path: Path,
    records: list[dict],
) -> None:
    with path.open(
        "w",
        encoding="utf-8",
    ) as stream:
        for record in records:
            stream.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
            )
            stream.write("\n")


def metadata_record() -> dict:
    return {
        "_type": "metadata",
        "schema_version": "1.1",
        "dimensions": [
            DIMENSION_KEY,
        ],
        "dimension_metadata": {
            DIMENSION_KEY: {
                "kind": "atomic",
                "class_name": "WordPerDictionary",
                "measure": "rate",
                "evidence": {
                    "kind": "text_span",
                    "source": "text_norm",
                    "unit": "characters",
                },
            }
        },
    }


def document_record(
    document_id: str,
    *,
    reference_length: int,
    evidence: list[dict],
) -> dict:
    return {
        "_type": "document",
        "id": document_id,
        "reference_lengths": {
            "text_norm": reference_length,
        },
        "dimensions": {
            DIMENSION_KEY: {
                "value": float(
                    len(evidence)
                ),
                "numerator": len(
                    evidence
                ),
                "evidence": evidence,
            }
        },
    }


def test_aggregate_dimension_from_jsonl(
    tmp_path: Path,
):
    path = tmp_path / "features.jsonl"

    write_jsonl(
        path,
        [
            metadata_record(),
            document_record(
                "doc-1",
                reference_length=12,
                evidence=[
                    {
                        "text": "a",
                        "start": 0,
                        "end": 1,
                    },
                    {
                        "text": "b",
                        "start": 10,
                        "end": 11,
                    },
                ],
            ),
            document_record(
                "doc-2",
                reference_length=12,
                evidence=[
                    {
                        "text": "c",
                        "start": 5,
                        "end": 6,
                    }
                ],
            ),
            document_record(
                "doc-3",
                reference_length=12,
                evidence=[],
            ),
        ],
    )

    result = aggregate_dimension_from_jsonl(
        path,
        DIMENSION_KEY,
        segments=3,
    )

    assert result.dimension_key == (
        DIMENSION_KEY
    )

    assert result.descriptor == (
        EvidenceDescriptor(
            kind="text_span",
            source="text_norm",
            unit="characters",
        )
    )

    assert result.segments == 3
    assert result.document_count == 3

    aggregated = result.distribution

    assert aggregated.document_count == 3
    assert (
        aggregated.documents_with_evidence
        == 2
    )
    assert aggregated.total_occurrences == 3

    assert aggregated.occurrence_counts == [
        1,
        1,
        1,
    ]

    assert aggregated.occurrence_shares == [
        pytest.approx(
            1 / 3
        ),
        pytest.approx(
            1 / 3
        ),
        pytest.approx(
            1 / 3
        ),
    ]

    assert aggregated.mean_document_shares == [
        pytest.approx(
            0.25
        ),
        pytest.approx(
            0.5
        ),
        pytest.approx(
            0.25
        ),
    ]

    assert aggregated.occurrence_delta == (
        pytest.approx(
            0.0
        )
    )

    assert aggregated.mean_document_delta == (
        pytest.approx(
            0.0
        )
    )


def test_iter_dimension_distributions_from_jsonl(
    tmp_path: Path,
):
    path = tmp_path / "features.jsonl"

    write_jsonl(
        path,
        [
            metadata_record(),
            document_record(
                "doc-1",
                reference_length=9,
                evidence=[
                    {
                        "text": "a",
                        "start": 0,
                        "end": 1,
                    }
                ],
            ),
            document_record(
                "doc-2",
                reference_length=9,
                evidence=[
                    {
                        "text": "b",
                        "start": 7,
                        "end": 8,
                    }
                ],
            ),
        ],
    )

    distributions = list(
        iter_dimension_distributions_from_jsonl(
            path,
            DIMENSION_KEY,
            segments=3,
        )
    )

    assert len(distributions) == 2

    assert segment_counts(
        distributions[0]
    ) == [
        1,
        0,
        0,
    ]

    assert segment_counts(
        distributions[1]
    ) == [
        0,
        0,
        1,
    ]

def test_service_accepts_blank_lines(
    tmp_path: Path,
):
    path = tmp_path / "features.jsonl"

    with path.open(
        "w",
        encoding="utf-8",
    ) as stream:
        stream.write("\n")
        stream.write(
            json.dumps(
                metadata_record()
            )
        )
        stream.write("\n\n")
        stream.write(
            json.dumps(
                document_record(
                    "doc-1",
                    reference_length=9,
                    evidence=[],
                )
            )
        )
        stream.write("\n")

    result = aggregate_dimension_from_jsonl(
        path,
        DIMENSION_KEY,
        segments=3,
    )

    assert result.document_count == 1
    assert (
        result.distribution.document_count
        == 1
    )
    assert (
        result.distribution.total_occurrences
        == 0
    )


def test_service_rejects_non_positive_segments(
    tmp_path: Path,
):
    path = tmp_path / "features.jsonl"

    write_jsonl(
        path,
        [
            metadata_record(),
        ],
    )

    with pytest.raises(
        ValueError,
        match="segments must be greater than zero",
    ):
        aggregate_dimension_from_jsonl(
            path,
            DIMENSION_KEY,
            segments=0,
        )


def test_service_rejects_empty_jsonl(
    tmp_path: Path,
):
    path = tmp_path / "empty.jsonl"
    path.write_text(
        "",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="JSONL file is empty",
    ):
        aggregate_dimension_from_jsonl(
            path,
            DIMENSION_KEY,
        )


def test_service_rejects_invalid_metadata_record(
    tmp_path: Path,
):
    path = tmp_path / "features.jsonl"

    write_jsonl(
        path,
        [
            {
                "_type": "document",
                "id": "doc-1",
            }
        ],
    )

    with pytest.raises(
        ValueError,
        match="first JSONL record",
    ):
        aggregate_dimension_from_jsonl(
            path,
            DIMENSION_KEY,
        )


def test_service_rejects_invalid_json(
    tmp_path: Path,
):
    path = tmp_path / "features.jsonl"

    path.write_text(
        (
            json.dumps(
                metadata_record()
            )
            + "\n"
            + "{not-json}\n"
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="invalid JSON",
    ):
        aggregate_dimension_from_jsonl(
            path,
            DIMENSION_KEY,
        )


def test_service_rejects_unknown_dimension(
    tmp_path: Path,
):
    path = tmp_path / "features.jsonl"

    write_jsonl(
        path,
        [
            metadata_record(),
        ],
    )

    with pytest.raises(
        (
            KeyError,
            ValueError,
        )
    ):
        aggregate_dimension_from_jsonl(
            path,
            "unknown-dimension",
        )


def test_service_reports_document_line_number(
    tmp_path: Path,
):
    path = tmp_path / "features.jsonl"

    invalid_document = (
        document_record(
            "doc-1",
            reference_length=5,
            evidence=[
                {
                    "text": "outside",
                    "start": 4,
                    "end": 10,
                }
            ],
        )
    )

    write_jsonl(
        path,
        [
            metadata_record(),
            invalid_document,
        ],
    )

    with pytest.raises(
        ValueError,
        match=r"features\.jsonl:2:",
    ):
        aggregate_dimension_from_jsonl(
            path,
            DIMENSION_KEY,
        )