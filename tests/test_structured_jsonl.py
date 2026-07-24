# tests/test_structured_jsonl.py

from __future__ import annotations

import io
import json

import pytest
import numpy as np
import pandas as pd

from umutextstats.output import (
    StructuredJSONLOutputResolver,
    write_output,
    write_structured_jsonl_stream,
)

from umutextstats.extraction import (
    BatchDimensionResult,
    ExtractionResult,
)

def read_jsonl(path) -> list[dict]:
    """
    Read a JSONL file and parse each non-empty line.
    """
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_structured_jsonl_resolver_supports_jsonl_extension():
    resolver = StructuredJSONLOutputResolver()

    assert resolver.supports("analysis.jsonl")
    assert resolver.supports("ANALYSIS.JSONL")
    assert not resolver.supports("analysis.json")
    assert not resolver.supports("analysis.csv")


def test_structured_jsonl_resolver_supports_explicit_format():
    resolver = StructuredJSONLOutputResolver()

    assert resolver.supports(
        "analysis.data",
        output_format="jsonl",
    )

    assert resolver.supports(
        "analysis.data",
        output_format="structured-jsonl",
    )

    assert not resolver.supports(
        "analysis.data",
        output_format="json",
    )


def test_write_output_writes_structured_jsonl(tmp_path):
    features = pd.DataFrame(
        {
            "id": ["doc-1", "doc-2"],
            "feature-a": [1, 2],
            "feature-b": [0.5, 1.5],
        }
    )

    output_path = tmp_path / "analysis.jsonl"

    write_output(
        features,
        output_path,
    )

    records = read_jsonl(output_path)

    assert len(records) == 3

    assert records[0] == {
        "_type": "metadata",
        "schema_version": "1.1",
        "dimensions": [
            "feature-a",
            "feature-b",
        ],
    }

    assert records[1] == {
        "_type": "document",
        "id": "doc-1",
        "dimensions": {
            "feature-a": {
                "value": 1,
            },
            "feature-b": {
                "value": 0.5,
            },
        },
    }

    assert records[2] == {
        "_type": "document",
        "id": "doc-2",
        "dimensions": {
            "feature-a": {
                "value": 2,
            },
            "feature-b": {
                "value": 1.5,
            },
        },
    }


def test_write_output_accepts_explicit_structured_jsonl_format(
    tmp_path,
):
    features = pd.DataFrame(
        {
            "id": ["doc-1"],
            "feature-a": [3],
        }
    )

    output_path = tmp_path / "analysis.data"

    write_output(
        features,
        output_path,
        output_format="structured-jsonl",
    )

    records = read_jsonl(output_path)

    assert records[1]["id"] == "doc-1"
    assert records[1]["dimensions"]["feature-a"]["value"] == 3


def test_structured_jsonl_converts_missing_values_to_null(
    tmp_path,
):
    features = pd.DataFrame(
        {
            "id": ["doc-1", "doc-2"],
            "feature-a": [
                np.nan,
                np.inf,
            ],
            "feature-b": [
                None,
                -np.inf,
            ],
        }
    )

    output_path = tmp_path / "analysis.jsonl"

    write_output(
        features,
        output_path,
    )

    records = read_jsonl(output_path)

    first_document = records[1]
    second_document = records[2]

    assert (
        first_document["dimensions"]["feature-a"]["value"]
        is None
    )
    assert (
        first_document["dimensions"]["feature-b"]["value"]
        is None
    )
    assert (
        second_document["dimensions"]["feature-a"]["value"]
        is None
    )
    assert (
        second_document["dimensions"]["feature-b"]["value"]
        is None
    )


def test_structured_jsonl_preserves_unicode(tmp_path):
    features = pd.DataFrame(
        {
            "id": ["documento-ñ"],
            "feature-text": ["Información lingüística"],
        }
    )

    output_path = tmp_path / "analysis.jsonl"

    write_output(
        features,
        output_path,
    )

    raw_text = output_path.read_text(encoding="utf-8")
    records = read_jsonl(output_path)

    assert "Información lingüística" in raw_text
    assert records[1]["id"] == "documento-ñ"
    assert (
        records[1]["dimensions"]["feature-text"]["value"]
        == "Información lingüística"
    )


def test_structured_jsonl_without_id_uses_row_number(tmp_path):
    features = pd.DataFrame(
        {
            "feature-a": [10, 20],
        },
        index=[5, 12],
    )

    output_path = tmp_path / "analysis.jsonl"

    write_output(
        features,
        output_path,
    )

    records = read_jsonl(output_path)

    assert records[1]["row"] == 1
    assert records[2]["row"] == 2
    assert "id" not in records[1]
    assert "id" not in records[2]


def test_write_structured_jsonl_stream():
    features = pd.DataFrame(
        {
            "id": ["doc-1"],
            "feature-a": [4],
        }
    )

    stream = io.StringIO()

    write_structured_jsonl_stream(
        features,
        stream,
    )

    records = [
        json.loads(line)
        for line in stream.getvalue().splitlines()
        if line.strip()
    ]

    assert len(records) == 2

    assert records[0]["_type"] == "metadata"
    assert records[1] == {
        "_type": "document",
        "id": "doc-1",
        "dimensions": {
            "feature-a": {
                "value": 4,
            },
        },
    }


def test_structured_jsonl_empty_dataframe(tmp_path):
    features = pd.DataFrame(
        columns=[
            "id",
            "feature-a",
        ]
    )

    output_path = tmp_path / "analysis.jsonl"

    write_output(
        features,
        output_path,
    )

    records = read_jsonl(output_path)

    assert records == [
        {
            "_type": "metadata",
            "schema_version": "1.1",
            "dimensions": [
                "feature-a",
            ],
        }
    ]

def test_write_extraction_result_as_structured_jsonl(
    tmp_path,
):
    result = ExtractionResult(
        ids=pd.Series(
            ["doc-1", "doc-2"],
        ),
        dimensions={
            "feature-a": BatchDimensionResult(
                key="feature-a",
                values=pd.Series([1, 2]),
                kind="atomic",
                metadata={
                    "class_name": "ExampleDimension",
                },
            ),
            "feature-total": BatchDimensionResult(
                key="feature-total",
                values=pd.Series([1, 2]),
                kind="composite",
                metadata={
                    "class_name": "CompositeDimension",
                    "strategy": "CompositeStrategySum",
                    "children": ["feature-a"],
                },
            ),
        },
    )

    output_path = tmp_path / "analysis.jsonl"

    write_output(
        result,
        output_path,
        output_format="structured-jsonl",
    )

    records = read_jsonl(output_path)

    assert len(records) == 3

    metadata = records[0]

    assert metadata["_type"] == "metadata"
    assert metadata["schema_version"] == "1.1"
    assert metadata["dimensions"] == [
        "feature-a",
        "feature-total",
    ]

    assert metadata["dimension_metadata"]["feature-a"] == {
        "kind": "atomic",
        "class_name": "ExampleDimension",
    }

    assert metadata["dimension_metadata"]["feature-total"] == {
        "kind": "composite",
        "class_name": "CompositeDimension",
        "strategy": "CompositeStrategySum",
        "children": ["feature-a"],
    }

    assert records[1] == {
        "_type": "document",
        "id": "doc-1",
        "dimensions": {
            "feature-a": {
                "value": 1,
            },
            "feature-total": {
                "value": 1,
            },
        },
    }

    assert records[2]["id"] == "doc-2"
    assert (
        records[2]["dimensions"]["feature-a"]["value"]
        == 2
    )


def test_write_extraction_result_without_ids(
    tmp_path,
):
    result = ExtractionResult(
        ids=None,
        dimensions={
            "feature-a": BatchDimensionResult(
                key="feature-a",
                values=pd.Series([10, 20]),
            ),
        },
    )

    output_path = tmp_path / "analysis.jsonl"

    write_output(
        result,
        output_path,
        output_format="structured-jsonl",
    )

    records = read_jsonl(output_path)

    assert records[1]["row"] == 1
    assert records[2]["row"] == 2
    assert "id" not in records[1]


def test_write_extraction_result_rejects_inconsistent_lengths(
    tmp_path,
):
    result = ExtractionResult(
        ids=pd.Series(
            ["doc-1", "doc-2"],
        ),
        dimensions={
            "feature-a": BatchDimensionResult(
                key="feature-a",
                values=pd.Series([1]),
            ),
        },
    )

    output_path = tmp_path / "analysis.jsonl"

    with pytest.raises(
        ValueError,
        match="unexpected length",
    ):
        write_output(
            result,
            output_path,
            output_format="structured-jsonl",
        )


def test_write_extraction_result_with_segments_and_derived(
    tmp_path,
):
    result = ExtractionResult(
        ids=pd.Series(["doc-1", "doc-2"]),
        dimensions={
            "feature-a": BatchDimensionResult(
                key="feature-a",
                values=pd.Series([10.0, 20.0]),
                segments={
                    "start": pd.Series([2.0, 4.0]),
                    "middle": pd.Series([3.0, 6.0]),
                    "end": pd.Series([5.0, 10.0]),
                },
                derived={
                    "delta": pd.Series([3.0, 6.0]),
                    "variability": pd.Series([1.25, 2.5]),
                },
            ),
        },
    )

    output_path = tmp_path / "analysis.jsonl"

    write_output(
        result,
        output_path,
        output_format="structured-jsonl",
    )

    records = read_jsonl(output_path)

    assert records[1] == {
        "_type": "document",
        "id": "doc-1",
        "dimensions": {
            "feature-a": {
                "value": 10.0,
                "segments": {
                    "start": {
                        "value": 2.0,
                    },
                    "middle": {
                        "value": 3.0,
                    },
                    "end": {
                        "value": 5.0,
                    },
                },
                "derived": {
                    "delta": 3.0,
                    "variability": 1.25,
                },
            },
        },
    }


def test_extraction_result_to_dataframe_with_segments():
    result = ExtractionResult(
        ids=pd.Series(["doc-1"]),
        dimensions={
            "feature-a": BatchDimensionResult(
                key="feature-a",
                values=pd.Series([10.0]),
                segments={
                    "start": pd.Series([2.0]),
                    "middle": pd.Series([3.0]),
                    "end": pd.Series([5.0]),
                },
            ),
        },
    )

    dataframe = result.to_dataframe(
        include_segments=True,
    )

    assert dataframe.to_dict(orient="records") == [
        {
            "id": "doc-1",
            "feature-a": 10.0,
            "feature-a__start": 2.0,
            "feature-a__middle": 3.0,
            "feature-a__end": 5.0,
        }
    ]


def test_extraction_result_to_dataframe_with_derived():
    result = ExtractionResult(
        ids=pd.Series(["doc-1"]),
        dimensions={
            "feature-a": BatchDimensionResult(
                key="feature-a",
                values=pd.Series([10.0]),
                derived={
                    "delta": pd.Series([3.0]),
                },
            ),
        },
    )

    dataframe = result.to_dataframe(
        include_derived=True,
    )

    assert dataframe.to_dict(orient="records") == [
        {
            "id": "doc-1",
            "feature-a": 10.0,
            "feature-a__delta": 3.0,
        }
    ]


def test_write_extraction_result_rejects_invalid_segment_length(
    tmp_path,
):
    result = ExtractionResult(
        ids=pd.Series(["doc-1", "doc-2"]),
        dimensions={
            "feature-a": BatchDimensionResult(
                key="feature-a",
                values=pd.Series([1.0, 2.0]),
                segments={
                    "start": pd.Series([1.0]),
                },
            ),
        },
    )

    output_path = tmp_path / "analysis.jsonl"

    with pytest.raises(
        ValueError,
        match="Segment 'start'.*unexpected length",
    ):
        write_output(
            result,
            output_path,
            output_format="structured-jsonl",
        )

def test_structured_jsonl_writes_evidence(
    tmp_path,
):
    result = ExtractionResult(
        ids=pd.Series(["doc-1"]),
        dimensions={
            "hashtags": BatchDimensionResult(
                key="hashtags",
                values=pd.Series([50.0]),
                numerators=pd.Series([2]),
                denominators=pd.Series([4]),
                evidence=pd.Series(
                    [
                        [
                            {
                                "text": "#a",
                                "start": 0,
                                "end": 2,
                            },
                            {
                                "text": "#b",
                                "start": 2,
                                "end": 4,
                            },
                        ]
                    ]
                ),
            ),
        },
    )

    output_path = tmp_path / "analysis.jsonl"

    StructuredJSONLOutputResolver().write(
        data=result,
        path=output_path,
    )

    records = [
        json.loads(line)
        for line in output_path.read_text(
            encoding="utf-8"
        ).splitlines()
    ]

    dimension = records[1]["dimensions"]["hashtags"]

    assert dimension == {
        "value": 50.0,
        "numerator": 2,
        "denominator": 4,
        "evidence": [
            {
                "text": "#a",
                "start": 0,
                "end": 2,
            },
            {
                "text": "#b",
                "start": 2,
                "end": 4,
            },
        ],
    }

def test_structured_jsonl_writes_dictionary_evidence(
    tmp_path,
):
    result = ExtractionResult(
        ids=pd.Series(["doc-1"]),
        dimensions={
            "negative": BatchDimensionResult(
                key="negative",
                values=pd.Series([5.0]),
                numerators=pd.Series([2]),
                denominators=pd.Series([40]),
                evidence=pd.Series(
                    [
                        [
                            {
                                "text": "asesinato",
                                "start": 3,
                                "end": 12,
                            },
                            {
                                "text": "caos",
                                "start": 84,
                                "end": 88,
                            },
                        ]
                    ]
                ),
                metadata={
                    "measure": "rate",
                    "normalization_unit": "words",
                    "scale": 100.0,
                },
            ),
        },
    )

    output_path = tmp_path / "analysis.jsonl"

    StructuredJSONLOutputResolver().write(
        data=result,
        path=output_path,
    )

    records = [
        json.loads(line)
        for line in output_path.read_text(
            encoding="utf-8"
        ).splitlines()
    ]

    assert records[1]["dimensions"]["negative"] == {
        "value": 5.0,
        "numerator": 2,
        "denominator": 40,
        "evidence": [
            {
                "text": "asesinato",
                "start": 3,
                "end": 12,
            },
            {
                "text": "caos",
                "start": 84,
                "end": 88,
            },
        ],
    }