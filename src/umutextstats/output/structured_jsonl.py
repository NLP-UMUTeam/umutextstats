# src/umutextstats/output/structured_jsonl.py

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, TextIO

import numpy as np
import pandas as pd

from umutextstats.extraction.models import ExtractionResult


SCHEMA_VERSION = "1.0"


class StructuredJSONLOutputResolver:
    """
    Write structured UMUTextStats extraction results as JSONL.

    A pandas DataFrame is also accepted temporarily for backwards
    compatibility.
    """

    def supports(
        self,
        path: str | Path,
        output_format: str | None = None,
    ) -> bool:
        path = Path(path)

        if output_format is not None:
            return output_format.lower() in {
                "jsonl",
                "structured-jsonl",
            }

        return path.suffix.lower() == ".jsonl"

    def write(
        self,
        data: ExtractionResult | pd.DataFrame,
        path: str | Path,
        **kwargs,
    ) -> None:
        path = Path(path)
        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with path.open(
            "w",
            encoding="utf-8",
        ) as stream:
            write_structured_jsonl_stream(
                data=data,
                stream=stream,
                **kwargs,
            )


def write_structured_jsonl_stream(
    data: ExtractionResult | pd.DataFrame,
    stream: TextIO,
    schema_version: str = SCHEMA_VERSION,
) -> None:
    """
    Write an ExtractionResult or a feature DataFrame to a text stream.
    """
    if isinstance(data, ExtractionResult):
        _write_extraction_result(
            result=data,
            stream=stream,
            schema_version=schema_version,
        )
        return

    if isinstance(data, pd.DataFrame):
        _write_dataframe(
            df=data,
            stream=stream,
            schema_version=schema_version,
        )
        return

    raise TypeError(
        "Structured JSONL output expects an ExtractionResult "
        f"or pandas DataFrame, got {type(data).__name__}"
    )


def _write_extraction_result(
    result: ExtractionResult,
    stream: TextIO,
    schema_version: str,
) -> None:
    """
    Write the canonical structured extraction result.
    """
    dimension_keys = list(result.dimensions)

    _validate_result_lengths(result)

    dimension_metadata = {
        key: {
            "kind": dimension.kind,
            **_to_json_mapping(dimension.metadata),
        }
        for key, dimension in result.dimensions.items()
    }

    metadata_record = {
        "_type": "metadata",
        "schema_version": schema_version,
        "dimensions": dimension_keys,
        "dimension_metadata": dimension_metadata,
    }

    if result.metadata:
        metadata_record["extraction_metadata"] = _to_json_mapping(
            result.metadata
        )

    _write_json_line(
        stream,
        metadata_record,
    )

    n_rows = _result_row_count(result)

    for position in range(n_rows):
        document = {
            "_type": "document",
            "dimensions": {
                key: _dimension_document_record(
                    dimension=dimension,
                    position=position,
                )
                for key, dimension in result.dimensions.items()
            },
        }

        if result.ids is not None:
            document["id"] = _to_json_value(
                result.ids.iloc[position]
            )
        else:
            document["row"] = position + 1

        _write_json_line(
            stream,
            document,
        )


def _write_dataframe(
    df: pd.DataFrame,
    stream: TextIO,
    schema_version: str,
) -> None:
    """
    Write a DataFrame using the initial structured JSONL representation.

    This path is retained for backwards compatibility and tests.
    """
    dimension_columns = [
        column
        for column in df.columns
        if column != "id"
    ]

    _write_json_line(
        stream,
        {
            "_type": "metadata",
            "schema_version": schema_version,
            "dimensions": dimension_columns,
        },
    )

    for row_number, (_, row) in enumerate(
        df.iterrows(),
        start=1,
    ):
        document = {
            "_type": "document",
            "dimensions": {
                key: {
                    "value": _to_json_value(row[key]),
                }
                for key in dimension_columns
            },
        }

        if "id" in df.columns:
            document["id"] = _to_json_value(row["id"])
        else:
            document["row"] = row_number

        _write_json_line(
            stream,
            document,
        )


def _validate_result_lengths(
    result: ExtractionResult,
) -> None:
    """
    Ensure all batch results contain the same number of rows.
    """
    expected = _result_row_count(result)

    if result.ids is not None and len(result.ids) != expected:
        raise ValueError(
            "ExtractionResult IDs have an unexpected length: "
            f"expected {expected}, got {len(result.ids)}"
        )

    for key, dimension in result.dimensions.items():
        if len(dimension.values) != expected:
            raise ValueError(
                f"Dimension {key!r} has an unexpected length: "
                f"expected {expected}, got {len(dimension.values)}"
            )

        if (
            dimension.numerators is not None
            and len(dimension.numerators) != expected
        ):
            raise ValueError(
                f"Numerators of dimension {key!r} have an "
                "unexpected length: "
                f"expected {expected}, "
                f"got {len(dimension.numerators)}"
            )

        if (
            dimension.denominators is not None
            and len(dimension.denominators) != expected
        ):
            raise ValueError(
                f"Denominators of dimension {key!r} have an "
                "unexpected length: "
                f"expected {expected}, "
                f"got {len(dimension.denominators)}"
            )

        if (
            dimension.evidence is not None
            and len(dimension.evidence) != expected
        ):
            raise ValueError(
                f"Evidence of dimension {key!r} has an "
                "unexpected length: "
                f"expected {expected}, "
                f"got {len(dimension.evidence)}"
            )

        for segment, values in dimension.segments.items():
            if len(values) != expected:
                raise ValueError(
                    f"Segment {segment!r} of dimension {key!r} "
                    "has an unexpected length: "
                    f"expected {expected}, got {len(values)}"
                )

        for statistic, values in dimension.derived.items():
            if len(values) != expected:
                raise ValueError(
                    f"Derived value {statistic!r} of dimension {key!r} "
                    "has an unexpected length: "
                    f"expected {expected}, got {len(values)}"
                )
        
def _result_row_count(
    result: ExtractionResult,
) -> int:
    """
    Infer the number of documents in an extraction result.
    """
    if result.ids is not None:
        return len(result.ids)

    for dimension in result.dimensions.values():
        return len(dimension.values)

    return 0


def _write_json_line(
    stream: TextIO,
    value: dict[str, Any],
) -> None:
    stream.write(
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        )
    )
    stream.write("\n")


def _to_json_mapping(
    value: dict[str, Any],
) -> dict[str, Any]:
    """
    Recursively convert a mapping to JSON-compatible values.
    """
    return {
        str(key): _to_json_value(item)
        for key, item in value.items()
    }


def _to_json_value(
    value: Any,
) -> Any:
    """
    Convert pandas, NumPy and nested Python values into valid JSON.
    """
    if value is None:
        return None

    if isinstance(value, np.generic):
        value = value.item()

    if isinstance(value, dict):
        return {
            str(key): _to_json_value(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple, set)):
        return [
            _to_json_value(item)
            for item in value
        ]

    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass

    if isinstance(value, float) and not math.isfinite(value):
        return None

    if isinstance(
        value,
        (str, int, float, bool),
    ):
        return value

    return str(value)


def _dimension_document_record(
    dimension,
    position: int,
) -> dict[str, Any]:
    """
    Build the structured record for one dimension and one document.
    """
    record = {
        "value": _to_json_value(
            dimension.values.iloc[position]
        ),
    }

    if dimension.numerators is not None:
        record["numerator"] = _to_json_value(
            dimension.numerators.iloc[position]
        )

    if dimension.denominators is not None:
        record["denominator"] = _to_json_value(
            dimension.denominators.iloc[position]
        )

    if dimension.evidence is not None:
        record["evidence"] = _to_json_value(
            dimension.evidence.iloc[position]
        )

    if dimension.segments:
        record["segments"] = {
            segment: {
                "value": _to_json_value(
                    values.iloc[position]
                ),
            }
            for segment, values in dimension.segments.items()
        }

    if dimension.derived:
        record["derived"] = {
            statistic: _to_json_value(
                values.iloc[position]
            )
            for statistic, values in dimension.derived.items()
        }

    return record