import pandas as pd

from umutextstats.config.models import DimensionConfig, UMUTextStatsConfig
from umutextstats.dimensions import DimensionEngine


def test_dimension_engine_builds_character_count_dimension_from_character_param():
    config = UMUTextStatsConfig(
        dimensions=[
            DimensionConfig(
                key="commas",
                class_name="CharacterCountDimension",
                params={"character": ","},
            )
        ]
    )

    df = pd.DataFrame(
        {
            "id": [1, 2],
            "text_norm": ["hola, mundo", "sin coma"],
        }
    )

    engine = DimensionEngine(config=config, show_progress=False)

    result = engine.compute(df)

    assert "commas" in result.columns
    assert result.loc[0, "commas"] > 0
    assert result.loc[1, "commas"] == 0
    
    
def test_dimension_engine_builds_pos_tagging_tag_from_tag_and_universal_params():
    config = UMUTextStatsConfig(
        dimensions=[
            DimensionConfig(
                key="present_verbs",
                class_name="POSTaggingTag",
                params={
                    "tag": "VERB",
                    "universal": "Tense=Pres",
                },
            )
        ]
    )

    df = pd.DataFrame(
        {
            "id": [1, 2],
            "text_norm": ["", ""],
            "tagged_pos": [
                "como__(VERB)(Mood=Ind|Tense=Pres|VerbForm=Fin)",
                "comí__(VERB)(Mood=Ind|Tense=Past|VerbForm=Fin)",
            ],
        }
    )

    engine = DimensionEngine(config=config, show_progress=False)

    result = engine.compute(df)

    assert "present_verbs" in result.columns
    assert result.loc[0, "present_verbs"] == 100.0
    assert result.loc[1, "present_verbs"] == 0.0
    
    
    
def test_dimension_engine_composite_sum_dimension():
    config = UMUTextStatsConfig(
        dimensions=[
            DimensionConfig(
                key="punctuation",
                class_name=None,
                strategy="CompositeStrategySum",
                children=[
                    DimensionConfig(
                        key="commas",
                        class_name="CharacterCountDimension",
                        params={"character": ","},
                    ),
                    DimensionConfig(
                        key="dots",
                        class_name="CharacterCountDimension",
                        params={"character": "."},
                    ),
                ],
            )
        ]
    )

    df = pd.DataFrame(
        {
            "id": [1],
            "text_norm": ["hola, mundo."],
        }
    )

    engine = DimensionEngine(config=config, show_progress=False)

    result = engine.compute(df)

    assert "commas" in result.columns
    assert "dots" in result.columns
    assert "punctuation" in result.columns

    assert result.loc[0, "punctuation"] == (
        result.loc[0, "commas"] + result.loc[0, "dots"]
    )
    
    
    