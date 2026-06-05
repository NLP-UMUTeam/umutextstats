from umutextstats.config.models import DimensionConfig
from umutextstats.dimensions.factory import build_dimension_instance
from umutextstats.dimensions.pattern import PatternDimension
from umutextstats.dimensions.word_per_dictionary import WordPerDictionary
from umutextstats.dimensions.pos_tagging_tag import POSTaggingTag


def test_build_pattern_dimension():
    config = DimensionConfig(
        key="x",
        class_name="PatternDimension",
        pattern="abc",
    )

    dimension = build_dimension_instance(
        dimension=config,
        dimension_cls=PatternDimension,
    )

    assert isinstance(dimension, PatternDimension)
    assert dimension.key == "x"
    assert dimension.pattern == "abc"


def test_build_word_per_dictionary_dimension():
    config = DimensionConfig(
        key="x",
        class_name="WordPerDictionary",
        dictionary="animals",
    )

    dimension = build_dimension_instance(
        dimension=config,
        dimension_cls=WordPerDictionary,
    )

    assert isinstance(dimension, WordPerDictionary)
    assert dimension.key == "x"
    assert dimension.dictionary_name == "animals"


def test_build_pos_tagging_tag_dimension():
    config = DimensionConfig(
        key="x",
        class_name="POSTaggingTag",
        params={"tag": "PRON", "universal": "PronType=Prs"},
    )

    dimension = build_dimension_instance(
        dimension=config,
        dimension_cls=POSTaggingTag,
    )

    assert isinstance(dimension, POSTaggingTag)
    assert dimension.key == "x"
    assert dimension.input_column == "tagged_pos"
    assert dimension.postagger_tag == "PRON"
    assert dimension.postagger_universal == "PronType=Prs"