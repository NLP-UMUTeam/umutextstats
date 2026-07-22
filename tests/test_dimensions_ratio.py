# tests/test_dimensions_ratio.py

import pytest
import pandas as pd

from umutextstats.config.models import DimensionConfig, UMUTextStatsConfig
from umutextstats.dimensions.engine import DimensionEngine
from umutextstats.dimensions.ratio import RatioDimension


def make_config(
    numerator="subjects|objects",
    denominator="subjects|objects|obliques",
    scale=None,
    zero_division=None,
):
    params = {
        "numerator": numerator,
        "denominator": denominator,
    }

    if scale is not None:
        params["scale"] = scale

    if zero_division is not None:
        params["zero_division"] = zero_division

    return UMUTextStatsConfig(
        dimensions=[
            DimensionConfig(
                key="subjects",
                class_name="POSTaggingTag",
                params={"tag": "NOUN"},
            ),
            DimensionConfig(
                key="objects",
                class_name="POSTaggingTag",
                params={"tag": "VERB"},
            ),
            DimensionConfig(
                key="obliques",
                class_name="POSTaggingTag",
                params={"tag": "ADJ"},
            ),
            DimensionConfig(
                key="ratio",
                class_name="RatioDimension",
                params=params,
            ),
        ]
    )


def compute(config):
    df = pd.DataFrame({
        "tagged_pos": [
            "casa__(NOUN), come__(VERB), bonita__(ADJ), perro__(NOUN)",
            "casa__(NOUN), perro__(NOUN)",
            "",
        ]
    })

    engine = DimensionEngine(config)
    return engine.compute(df)


def test_ratio_dimension_basic():
    result = compute(make_config())

    assert list(result["subjects"]) == [50.0, 100.0, 0.0]
    assert list(result["objects"]) == [25.0, 0.0, 0.0]
    assert list(result["obliques"]) == [25.0, 0.0, 0.0]

    assert list(result["ratio"]) == [0.75, 1.0, 0.0]


def test_ratio_dimension_with_scale():
    result = compute(make_config(scale="100"))

    assert list(result["ratio"]) == [75.0, 100.0, 0.0]


def test_ratio_dimension_with_zero_division():
    result = compute(make_config(zero_division="999"))

    assert list(result["ratio"]) == [0.75, 1.0, 999.0]


def test_ratio_dimension_missing_column_raises():
    config = make_config(
        numerator="subjects|missing_column",
        denominator="subjects",
    )

    with pytest.raises(ValueError, match="Missing columns for RatioDimension"):
        compute(config)


def test_ratio_dimension_keeps_source_columns():
    result = compute(make_config())

    assert "subjects" in result.columns
    assert "objects" in result.columns
    assert "obliques" in result.columns
    assert "ratio" in result.columns
    
    
def test_ratio_dimension_with_dependency_tags():
    config = UMUTextStatsConfig(
        dimensions=[
            DimensionConfig(
                key="syntax-dependencies-subjects",
                class_name="DependencyTag",
                params={"deprel": "nsubj"},
            ),
            DimensionConfig(
                key="syntax-dependencies-objects",
                class_name="DependencyTag",
                params={"deprel": "obj"},
            ),
            DimensionConfig(
                key="syntax-dependencies-obliques",
                class_name="DependencyTag",
                params={"deprel": "obl"},
            ),
            DimensionConfig(
                key="syntax-ratios-core-arguments",
                class_name="RatioDimension",
                params={
                    "numerator": "syntax-dependencies-subjects|syntax-dependencies-objects",
                    "denominator": "syntax-dependencies-subjects|syntax-dependencies-objects|syntax-dependencies-obliques",
                },
            ),
        ]
    )

    df = pd.DataFrame({
        "tagged_dep": [
            "María__(nsubj)(2), compra__(root)(0), libros__(obj)(2), casa__(obl)(2)"
        ]
    })

    engine = DimensionEngine(config)
    result = engine.compute(df)

    assert round(result["syntax-ratios-core-arguments"][0], 2) == 0.67

def test_ratio_compute_result_from_data():
    dimension = RatioDimension(
        key="ratio",
        numerator=["a", "b"],
        denominator=["total"],
        scale=100,
    )

    data = {
        "a": pd.Series([2.0, 1.0]),
        "b": pd.Series([3.0, 1.0]),
        "total": pd.Series([20.0, 0.0]),
    }

    result = dimension.compute_result_from_data(
        data=data,
        n_rows=2,
    )

    assert result.values.tolist() == [25.0, 0.0]
    assert result.numerators.tolist() == [5.0, 2.0]
    assert result.denominators.tolist() == [20.0, 0.0]

    assert result.metadata["measure"] == "ratio"
    assert result.metadata["scale"] == 100.0