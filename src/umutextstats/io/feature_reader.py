from __future__ import annotations

from io import StringIO
from pathlib import Path
from typing import Any, TextIO

import pandas as pd

from umutextstats.io.structured_jsonl import (
    StructuredExtractionData,
    read_structured_jsonl,
)


class FeatureReader:
    """
    Read already computed feature data.

    Supported inputs:

    - CSV feature matrices.
    - Structured extraction JSONL.
    - Text streams containing either format.
    """

    def read(
        self,
        source: str | Path | TextIO,
    ) -> pd.DataFrame:
        """
        Read a feature matrix from a path or text stream.
        """
        if hasattr(source, "read"):
            return self._read_stream(
                source
            )

        path = Path(source)

        if path.suffix.lower() in {
            ".jsonl",
            ".ndjson",
        }:
            extraction = read_structured_jsonl(
                path
            )

            return self.to_dataframe(
                extraction
            )

        return pd.read_csv(
            path
        )

    def to_dataframe(
        self,
        extraction: StructuredExtractionData,
    ) -> pd.DataFrame:
        """
        Convert structured extraction records to a flat feature matrix.

        Only each dimension's `value` field is included.
        """
        dimension_keys = [
            str(key)
            for key in extraction.metadata.get(
                "dimensions",
                [],
            )
        ]

        rows: list[dict[str, Any]] = []

        for document in extraction.documents:
            row: dict[str, Any] = {}

            if "id" in document:
                row["id"] = document["id"]

            dimensions = document.get(
                "dimensions",
                {},
            )

            if not isinstance(
                dimensions,
                dict,
            ):
                raise ValueError(
                    "Structured document field "
                    "'dimensions' must be a mapping."
                )

            for key, result in dimensions.items():
                if isinstance(
                    result,
                    dict,
                ):
                    row[key] = result.get(
                        "value"
                    )
                else:
                    row[key] = result

            rows.append(row)

        columns: list[str] = []

        if any(
            "id" in row
            for row in rows
        ):
            columns.append(
                "id"
            )

        columns.extend(
            dimension_keys
        )

        if not rows:
            return pd.DataFrame(
                columns=columns
            )

        frame = pd.DataFrame(
            rows
        ).reindex(
            columns=columns
        )

        for column in dimension_keys:
            frame[column] = pd.to_numeric(
                frame[column],
                errors="coerce",
            )

        return frame

    
    def _read_stream(
        self,
        stream: TextIO,
    ) -> pd.DataFrame:
        """
        Detect and read CSV or structured JSONL from a text stream.
        """
        content = stream.read()

        if not content.strip():
            return pd.DataFrame()

        first_nonempty_line = next(
            (
                line.strip()
                for line in content.splitlines()
                if line.strip()
            ),
            "",
        )

        buffer = StringIO(
            content
        )

        if first_nonempty_line.startswith(
            "{"
        ):
            extraction = read_structured_jsonl(
                buffer
            )

            return self.to_dataframe(
                extraction
            )

        return pd.read_csv(
            buffer
        )