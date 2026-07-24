from __future__ import annotations

import pytest

from umutextstats.evidence import (
    dimension_distribution_from_document,
    dimension_occurrences_from_document,
    dimension_position_metadata,
    evidence_occurrences_from_dimension_record,
)


DIMENSION_KEY = (
    "psycholinguistic-processes-negative-general"
)


def make_metadata() -> dict:
    return {
        "_type": "metadata",
        "schema_version": "1.0",
        "dimensions": [
            DIMENSION_KEY,
        ],
        "dimension_metadata": {
            DIMENSION_KEY: {
                "kind": "atomic",
                "class_name": "WordPerDictionary",
                "input_column": "text_norm",
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
            "text_norm": 54,
        },
        "dimensions": {
            DIMENSION_KEY: {
                "value": 44.44444444444444,
                "numerator": 4,
                "denominator": 9,
                "evidence": [
                    {
                        "text": "terrible",
                        "start": 17,
                        "end": 25,
                    },
                    {
                        "text": "miedo",
                        "start": 31,
                        "end": 36,
                    },
                    {
                        "text": "dolor",
                        "start": 38,
                        "end": 43,
                    },
                    {
                        "text": "fracaso",
                        "start": 46,
                        "end": 53,
                    },
                ],
            }
        },
    }


def test_evidence_occurrences_from_dimension_record():
    dimension_record = make_document()[
        "dimensions"
    ][DIMENSION_KEY]

    occurrences = (
        evidence_occurrences_from_dimension_record(
            dimension_record,
            offset_source="text_norm",
            offset_unit="characters",
        )
    )

    assert len(
        occurrences
    ) == 4

    assert occurrences[0].label == "terrible"
    assert occurrences[0].start == 17
    assert occurrences[0].end == 25
    assert occurrences[0].offset_source == "text_norm"
    assert occurrences[0].offset_unit == "characters"


def test_dimension_position_metadata():
    offset_source, offset_unit = (
        dimension_position_metadata(
            make_metadata(),
            DIMENSION_KEY,
        )
    )

    assert offset_source == "text_norm"
    assert offset_unit == "characters"


def test_dimension_position_metadata_rejects_missing_descriptor():
    metadata = make_metadata()

    del metadata[
        "dimension_metadata"
    ][DIMENSION_KEY]["evidence"]

    with pytest.raises(
        ValueError,
        match="has no evidence descriptor",
    ):
        dimension_position_metadata(
            metadata,
            DIMENSION_KEY,
        )

def test_dimension_occurrences_from_document():
    occurrences = (
        dimension_occurrences_from_document(
            metadata_record=make_metadata(),
            document_record=make_document(),
            dimension_key=DIMENSION_KEY,
        )
    )

    assert [
        occurrence.label
        for occurrence in occurrences
    ] == [
        "terrible",
        "miedo",
        "dolor",
        "fracaso",
    ]


def test_dimension_distribution_from_document():
    distribution = (
        dimension_distribution_from_document(
            metadata_record=make_metadata(),
            document_record=make_document(),
            dimension_key=DIMENSION_KEY,
            segments=3,
        )
    )

    assert distribution.reference_length == 54
    assert distribution.total_occurrences == 4
    assert distribution.offset_source == "text_norm"
    assert distribution.offset_unit == "characters"

    assert [
        segment.count
        for segment in distribution.segments
    ] == [
        0,
        2,
        2,
    ]

    assert [
        segment.share
        for segment in distribution.segments
    ] == pytest.approx(
        [
            0.0,
            0.5,
            0.5,
        ]
    )

    assert distribution.delta == pytest.approx(
        0.5
    )


def test_missing_evidence_returns_empty_list():
    occurrences = (
        evidence_occurrences_from_dimension_record(
            {
                "value": 0.0,
                "numerator": 0,
                "denominator": 10,
            },
            offset_source="text_norm",
        )
    )

    assert occurrences == []


def test_rejects_evidence_without_start():
    with pytest.raises(
        ValueError,
        match="has no 'start' field",
    ):
        evidence_occurrences_from_dimension_record(
            {
                "evidence": [
                    {
                        "text": "terrible",
                        "end": 8,
                    }
                ]
            }
        )


def test_rejects_non_list_evidence():
    with pytest.raises(
        ValueError,
        match="evidence must be a list",
    ):
        evidence_occurrences_from_dimension_record(
            {
                "evidence": {
                    "text": "terrible",
                    "start": 0,
                    "end": 8,
                }
            }
        )


def test_rejects_missing_dimension():
    with pytest.raises(
        ValueError,
        match="is not present in the document record",
    ):
        dimension_occurrences_from_document(
            metadata_record=make_metadata(),
            document_record={
                "_type": "document",
                "dimensions": {},
            },
            dimension_key=DIMENSION_KEY,
        )