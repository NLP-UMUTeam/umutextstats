from __future__ import annotations

import io
import json

import pandas as pd

from umutextstats.evidence.descriptors import (
    EvidenceDescriptor,
)
from umutextstats.extraction.models import (
    BatchDimensionResult,
    ExtractionResult,
)
from umutextstats.output.structured_jsonl import (
    write_structured_jsonl_stream,
)


def test_writes_evidence_descriptor_and_reference_lengths():
    result = ExtractionResult(
        ids=pd.Series(
            [
                10,
                11,
            ]
        ),
        dimensions={
            "negative": BatchDimensionResult(
                key="negative",
                values=pd.Series(
                    [
                        1.0,
                        2.0,
                    ]
                ),
                evidence=pd.Series(
                    [
                        [
                            {
                                "text": "miedo",
                                "start": 0,
                                "end": 5,
                            }
                        ],
                        [],
                    ]
                ),
                evidence_descriptor=(
                    EvidenceDescriptor(
                        kind="text_span",
                        source="text_norm",
                        unit="characters",
                    )
                ),
            )
        },
        reference_lengths={
            "text_norm": pd.Series(
                [
                    20,
                    30,
                ]
            )
        },
    )

    stream = io.StringIO()

    write_structured_jsonl_stream(
        result,
        stream,
    )

    records = [
        json.loads(line)
        for line in stream.getvalue().splitlines()
    ]

    metadata = records[0]
    first_document = records[1]
    second_document = records[2]

    assert metadata[
        "schema_version"
    ] == "1.1"

    assert metadata[
        "dimension_metadata"
    ]["negative"]["evidence"] == {
        "kind": "text_span",
        "source": "text_norm",
        "unit": "characters",
    }

    assert first_document[
        "reference_lengths"
    ] == {
        "text_norm": 20,
    }

    assert second_document[
        "reference_lengths"
    ] == {
        "text_norm": 30,
    }

    assert first_document[
        "dimensions"
    ]["negative"]["evidence"] == [
        {
            "text": "miedo",
            "start": 0,
            "end": 5,
        }
    ]