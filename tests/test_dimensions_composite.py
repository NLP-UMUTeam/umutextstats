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
            "hola mundo",  # length=10, words=2
            "hola",        # length=4, words=1
        ]
    })

    engine = DimensionEngine(make_config(strategy))
    return engine.compute(df)


# --- SUM ---
def test_composite_sum():
    result = compute("CompositeStrategySum")
    assert list(result["parent"]) == [12, 5]

def test_composite_avg():
    result = compute("CompositeStrategyAvg")
    assert list(result["parent"]) == [6.0, 2.5]

def test_composite_max():
    result = compute("CompositeStrategyMax")
    assert list(result["parent"]) == [10, 4]

def test_composite_min():
    result = compute("CompositeStrategyMin")
    assert list(result["parent"]) == [2, 1]


# --- fallback ---
def test_composite_unknown_defaults_to_sum():
    result = compute("whatever")
    assert list(result["parent"]) == [12, 5]
    
def test_composite_keeps_child_columns():
    result = compute("CompositeStrategySum")

    assert list(result["length"]) == [10, 4]
    assert list(result["words"]) == [2, 1]
    assert list(result["parent"]) == [12, 5]
    
    
def test_composite_with_empty_text():
    df = pd.DataFrame({
        "text_norm": [
            "",
            "hola mundo",
        ]
    })

    engine = DimensionEngine(make_config("CompositeStrategySum"))
    result = engine.compute(df)

    assert list(result["parent"]) == [0, 12]