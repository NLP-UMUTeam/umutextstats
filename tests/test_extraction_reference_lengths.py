from __future__ import annotations

import pytest

from umutextstats.evidence import (
    dimension_distribution_from_document,
    dimension_evidence_descriptor,
    document_reference_length,
)


DIMENSION_KEY = "negative"


def make_metadata() -> dict:
    return {
        "_type": "metadata",
        "schema_version": "1.1",
        "dimensions": [
            DIMENSION_KEY,
        ],
        "dimension_metadata": {
            DIMENSION_KEY: {
                "kind": "atomic",
                "evidence": {
                    "kind": "text_span",
                    "source": "text_norm",
                    "unit": "characters",
                },
            }
        },
    }


def make_document() -> dict:
    return {
        "_type": "document",
        "id": 1,
        "reference_lengths": {
            "text_norm": 90,
        },
        "dimensions": {
            DIMENSION_KEY: {
                "value": 3,
                "numerator": 3,
                "evidence": [
                    {
                        "text": "a",
                        "start": 10,
                        "end": 11,
                    },
                    {
                        "text": "b",
                        "start": 40,
                        "end": 41,
                    },
                    {
                        "text": "c",
                        "start": 80,
                        "end": 81,
                    },
                ],
            }
        },
    }


def test_reads_dimension_evidence_descriptor():
    descriptor = dimension_evidence_descriptor(
        metadata_record=make_metadata(),
        dimension_key=DIMENSION_KEY,
    )

    assert descriptor.kind == "text_span"
    assert descriptor.source == "text_norm"
    assert descriptor.unit == "characters"


def test_reads_document_reference_length():
    assert document_reference_length(
        document_record=make_document(),
        source="text_norm",
    ) == 90


def test_builds_distribution_without_external_length():
    distribution = (
        dimension_distribution_from_document(
            metadata_record=make_metadata(),
            document_record=make_document(),
            dimension_key=DIMENSION_KEY,
            segments=3,
        )
    )

    assert distribution.reference_length == 90
    assert distribution.total_occurrences == 3

    assert [
        segment.count
        for segment in distribution.segments
    ] == [
        1,
        1,
        1,
    ]


def test_rejects_missing_evidence_descriptor():
    metadata = make_metadata()

    del metadata[
        "dimension_metadata"
    ][DIMENSION_KEY]["evidence"]

    with pytest.raises(
        ValueError,
        match="has no evidence descriptor",
    ):
        dimension_evidence_descriptor(
            metadata_record=metadata,
            dimension_key=DIMENSION_KEY,
        )


def test_rejects_missing_reference_length():
    document = make_document()

    document[
        "reference_lengths"
    ] = {}

    with pytest.raises(
        ValueError,
        match="is not present",
    ):
        document_reference_length(
            document_record=document,
            source="text_norm",
        )