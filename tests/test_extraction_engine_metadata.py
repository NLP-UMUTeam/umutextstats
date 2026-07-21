from __future__ import annotations

import pandas as pd

from umutextstats.config.models import (
    DimensionConfig,
    UMUTextStatsConfig,
)
from umutextstats.extraction import ExtractionEngine


def test_atomic_dimension_metadata():
    config = UMUTextStatsConfig(
        dimensions=[
            DimensionConfig(
                key="hashtags",
                class_name="PatternDimension",
                description="Number of hashtags",
                pattern=r"(?<!\S)#[\p{L}\p{N}_]+",
                input_column="text_raw",
                percentage=False,
                params={
                    "separator": "whole",
                },
            )
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

    metadata = result.dimensions["hashtags"].metadata

    assert metadata["class_name"] == "PatternDimension"
    assert metadata["description"] == "Number of hashtags"
    assert metadata["pattern"] == r"(?<!\S)#[\p{L}\p{N}_]+"
    assert metadata["input_column"] == "text_raw"
    assert metadata["percentage"] is False
    assert metadata["params"] == {
        "separator": "whole",
    }


def test_composite_dimension_metadata():
    config = UMUTextStatsConfig(
        dimensions=[
            DimensionConfig(
                key="social-addressivity",
                strategy="CompositeStrategySum",
                description="Addressivity features",
                children=[
                    DimensionConfig(
                        key="hashtags",
                        class_name="PatternDimension",
                        pattern=r"(?<!\S)#[\p{L}\p{N}_]+",
                        input_column="text_raw",
                        percentage=False,
                    ),
                    DimensionConfig(
                        key="mentions",
                        class_name="PatternDimension",
                        pattern=r"(?<!\S)@[\p{L}\p{N}_]+",
                        input_column="text_raw",
                        percentage=False,
                    ),
                ],
            )
        ]
    )

    df = pd.DataFrame(
        {
            "text_raw": ["Hola @pepe #Python"],
            "text_norm": ["Hola @pepe #Python"],
        }
    )

    result = ExtractionEngine(
        config=config,
        show_progress=False,
    ).compute(df)

    composite = result.dimensions["social-addressivity"]

    assert composite.kind == "composite"
    assert composite.values.tolist() == [2]

    assert composite.metadata["class_name"] == "CompositeDimension"
    assert composite.metadata["description"] == "Addressivity features"
    assert (
        composite.metadata["strategy"]
        == "CompositeStrategySum"
    )
    assert composite.metadata["children"] == [
        "hashtags",
        "mentions",
    ]
    assert composite.metadata["used_children"] == [
        "hashtags",
        "mentions",
    ]
    assert composite.metadata["missing_children"] == []