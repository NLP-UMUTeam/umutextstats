import pandas as pd

from umutextstats.config.models import DimensionConfig, UMUTextStatsConfig
from umutextstats.dimensions.engine import DimensionEngine


def make_config(strategy):
    return UMUTextStatsConfig(
        dimensions=[
            DimensionConfig(
                key="parent",
                strategy=strategy,
                children=[
                    DimensionConfig(
                        key="length",
                        class_name="LengthDimension",
                    ),
                    DimensionConfig(
                        key="words",
                        class_name="WordCountDimension",
                    ),
                ],
            )
        ]
    )


def compute(strategy):
    df = pd.DataFrame({
        "text_norm": [
            "hola mundo",
            "hola",
        ]
    })

    engine = DimensionEngine(make_config(strategy))
    return engine.compute(df)


def test_composite_sum():
    result = compute("SUM")

    assert list(result["length"]) == [10, 4]
    assert list(result["words"]) == [2, 1]
    assert list(result["parent"]) == [12, 5]


def test_composite_avg():
    result = compute("AVG")

    assert list(result["parent"]) == [6.0, 2.5]


def test_composite_max():
    result = compute("MAX")

    assert list(result["parent"]) == [10, 4]


def test_composite_min():
    result = compute("MIN")

    assert list(result["parent"]) == [2, 1]