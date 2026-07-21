# tests/test_vectorization_jsonl.py

from __future__ import annotations

import io

import pandas as pd
import pytest

from umutextstats.vectorization import (
    read_structured_jsonl,
    read_structured_jsonl_stream,
)


def test_read_structured_jsonl_stream():
    stream = io.StringIO(
        "\n".join(
            [
                (
                    '{"_type":"metadata","schema_version":"1.0",'
                    '"dimensions":["feature-a","feature-b"]}'
                ),
                (
                    '{"_type":"document","id":"doc-1",'
                    '"dimensions":{'
                    '"feature-a":{"value":1},'
                    '"feature-b":{"value":0.5}}}'
                ),
                (
                    '{"_type":"document","id":"doc-2",'
                    '"dimensions":{'
                    '"feature-a":{"value":2},'
                    '"feature-b":{"value":null}}}'
                ),
            ]
        )
    )

    result = read_structured_jsonl_stream(stream)

    expected = pd.DataFrame(
        {
            "id": ["doc-1", "doc-2"],
            "feature-a": [1, 2],
            "feature-b": [0.5, None],
        }
    )

    pd.testing.assert_frame_equal(
        result,
        expected,
        check_dtype=False,
    )


def test_read_structured_jsonl_preserves_dimension_order():
    stream = io.StringIO(
        "\n".join(
            [
                (
                    '{"_type":"metadata","schema_version":"1.0",'
                    '"dimensions":["feature-b","feature-a"]}'
                ),
                (
                    '{"_type":"document","id":"doc-1",'
                    '"dimensions":{'
                    '"feature-a":{"value":1},'
                    '"feature-b":{"value":2}}}'
                ),
            ]
        )
    )

    result = read_structured_jsonl_stream(stream)

    assert list(result.columns) == [
        "id",
        "feature-b",
        "feature-a",
    ]


def test_read_structured_jsonl_without_id():
    stream = io.StringIO(
        "\n".join(
            [
                (
                    '{"_type":"metadata","schema_version":"1.0",'
                    '"dimensions":["feature-a"]}'
                ),
                (
                    '{"_type":"document","row":1,'
                    '"dimensions":{"feature-a":{"value":10}}}'
                ),
            ]
        )
    )

    result = read_structured_jsonl_stream(stream)

    expected = pd.DataFrame(
        {
            "feature-a": [10],
        }
    )

    pd.testing.assert_frame_equal(
        result,
        expected,
        check_dtype=False,
    )


def test_read_structured_jsonl_missing_dimension_becomes_none():
    stream = io.StringIO(
        "\n".join(
            [
                (
                    '{"_type":"metadata","schema_version":"1.0",'
                    '"dimensions":["feature-a","feature-b"]}'
                ),
                (
                    '{"_type":"document","id":"doc-1",'
                    '"dimensions":{"feature-a":{"value":1}}}'
                ),
            ]
        )
    )

    result = read_structured_jsonl_stream(stream)

    assert result.loc[0, "feature-a"] == 1
    assert pd.isna(result.loc[0, "feature-b"])


def test_read_structured_jsonl_from_file(tmp_path):
    input_path = tmp_path / "analysis.jsonl"

    input_path.write_text(
        "\n".join(
            [
                (
                    '{"_type":"metadata","schema_version":"1.0",'
                    '"dimensions":["feature-a"]}'
                ),
                (
                    '{"_type":"document","id":"doc-1",'
                    '"dimensions":{"feature-a":{"value":3}}}'
                ),
            ]
        ),
        encoding="utf-8",
    )

    result = read_structured_jsonl(input_path)

    assert result.to_dict(orient="records") == [
        {
            "id": "doc-1",
            "feature-a": 3,
        }
    ]


def test_read_structured_jsonl_rejects_invalid_json():
    stream = io.StringIO(
        "\n".join(
            [
                (
                    '{"_type":"metadata","schema_version":"1.0",'
                    '"dimensions":[]}'
                ),
                "{invalid-json}",
            ]
        )
    )

    with pytest.raises(
        ValueError,
        match="Invalid JSON on line 2",
    ):
        read_structured_jsonl_stream(stream)


def test_read_structured_jsonl_requires_metadata():
    stream = io.StringIO(
        (
            '{"_type":"document","id":"doc-1",'
            '"dimensions":{"feature-a":{"value":1}}}'
        )
    )

    with pytest.raises(
        ValueError,
        match="Document record found before metadata",
    ):
        read_structured_jsonl_stream(stream)


def test_read_structured_jsonl_rejects_duplicate_metadata():
    stream = io.StringIO(
        "\n".join(
            [
                (
                    '{"_type":"metadata","schema_version":"1.0",'
                    '"dimensions":[]}'
                ),
                (
                    '{"_type":"metadata","schema_version":"1.0",'
                    '"dimensions":[]}'
                ),
            ]
        )
    )

    with pytest.raises(
        ValueError,
        match="more than one metadata record",
    ):
        read_structured_jsonl_stream(stream)


def test_read_structured_jsonl_rejects_unsupported_schema():
    stream = io.StringIO(
        (
            '{"_type":"metadata","schema_version":"2.0",'
            '"dimensions":[]}'
        )
    )

    with pytest.raises(
        ValueError,
        match="Unsupported schema version",
    ):
        read_structured_jsonl_stream(stream)


def test_read_structured_jsonl_rejects_unknown_dimension():
    stream = io.StringIO(
        "\n".join(
            [
                (
                    '{"_type":"metadata","schema_version":"1.0",'
                    '"dimensions":["feature-a"]}'
                ),
                (
                    '{"_type":"document","id":"doc-1",'
                    '"dimensions":{'
                    '"feature-a":{"value":1},'
                    '"feature-b":{"value":2}}}'
                ),
            ]
        )
    )

    with pytest.raises(
        ValueError,
        match="not declared in metadata",
    ):
        read_structured_jsonl_stream(stream)


def test_read_structured_jsonl_rejects_non_object_dimension():
    stream = io.StringIO(
        "\n".join(
            [
                (
                    '{"_type":"metadata","schema_version":"1.0",'
                    '"dimensions":["feature-a"]}'
                ),
                (
                    '{"_type":"document","id":"doc-1",'
                    '"dimensions":{"feature-a":1}}'
                ),
            ]
        )
    )

    with pytest.raises(
        ValueError,
        match="must be an object",
    ):
        read_structured_jsonl_stream(stream)