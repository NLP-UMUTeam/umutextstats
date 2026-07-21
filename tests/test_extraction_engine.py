# tests/test_extraction_engine.py

from __future__ import annotations

import pandas as pd

from umutextstats.config.models import (
    DimensionConfig,
    UMUTextStatsConfig,
)
from umutextstats.dimensions import DimensionEngine
from umutextstats.extraction import ExtractionEngine


def test_extraction_engine_matches_dimension_engine():
    config = UMUTextStatsConfig(
        dimensions=[
            DimensionConfig(
                key="hashtags",
                class_name="PatternDimension",
                pattern=r"(?<!\S)#[\p{L}\p{N}_]+",
                input_column="text_raw",
                percentage=False,
            ),
        ]
    )

    df = pd.DataFrame(
        {
            "id": ["doc-1", "doc-2"],
            "text_raw": [
                "Hola #Python y #NLP",
                "Sin etiquetas",
            ],
            "text_norm": [
                "Hola #Python y #NLP",
                "Sin etiquetas",
            ],
        }
    )

    expected = DimensionEngine(
        config=config,
        input_column="text_norm",
        show_progress=False,
    ).compute(df)

    result = ExtractionEngine(
        config=config,
        input_column="text_norm",
        show_progress=False,
    ).compute(df)

    actual = result.to_dataframe()

    pd.testing.assert_frame_equal(
        actual,
        expected,
        check_dtype=False,
    )


def test_extraction_result_preserves_dimension_metadata():
    config = UMUTextStatsConfig(
        dimensions=[
            DimensionConfig(
                key="hashtags",
                class_name="PatternDimension",
                pattern=r"(?<!\S)#[\p{L}\p{N}_]+",
                input_column="text_raw",
                percentage=False,
            ),
        ]
    )

    df = pd.DataFrame(
        {
            "text_raw": ["Hola #Python"],
            "text_norm": ["Hola #Python"],
        }
    )

    result = ExtractionEngine(
        config=config,
        show_progress=False,
    ).compute(df)

    dimension = result.dimensions["hashtags"]

    assert dimension.key == "hashtags"
    assert dimension.kind == "atomic"
    assert dimension.metadata["class_name"] == "PatternDimension"
    assert dimension.values.tolist() == [1]


def test_extraction_result_preserves_composite_metadata():
    config = UMUTextStatsConfig(
        dimensions=[
            DimensionConfig(
                key="hashtags-total",
                strategy="CompositeStrategySum",
                children=[
                    DimensionConfig(
                        key="hashtags",
                        class_name="PatternDimension",
                        pattern=r"(?<!\S)#[\p{L}\p{N}_]+",
                        input_column="text_raw",
                        percentage=False,
                    ),
                ],
            ),
        ]
    )

    df = pd.DataFrame(
        {
            "text_raw": ["Hola #Python"],
            "text_norm": ["Hola #Python"],
        }
    )

    result = ExtractionEngine(
        config=config,
        show_progress=False,
    ).compute(df)

    composite = result.dimensions["hashtags-total"]

    assert composite.kind == "composite"
    assert composite.metadata["strategy"] == "CompositeStrategySum"
    assert composite.metadata["children"] == ["hashtags"]
    assert composite.metadata["used_children"] == ["hashtags"]
    assert composite.metadata["missing_children"] == []
    assert composite.values.tolist() == [1]