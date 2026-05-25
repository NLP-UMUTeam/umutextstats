from __future__ import annotations

from typing import Callable

from umutextstats.dimensions.registry import normalize_class_name
from umutextstats.config.models import DimensionConfig
from umutextstats.config.params import (
    dictionary_param,
    disabled_regexp_param,
    param,
    percentage_param,
)


def build_dimension(
    dimension: DimensionConfig,
    dimension_cls,
    default_input_column: str = "text_norm",
):
    class_name = normalize_class_name(dimension.class_name)

    input_column = resolve_input_column(
        dimension,
        default_input_column=default_input_column,
    )

    builder = DIMENSION_BUILDERS.get(class_name)

    if builder:
        return builder(
            dimension=dimension,
            dimension_cls=dimension_cls,
            input_column=input_column,
        )

    return dimension_cls(
        key=dimension.key,
        input_column=input_column,
    )


def resolve_input_column(
    dimension: DimensionConfig,
    default_input_column: str = "text_norm",
) -> str:
    return (
        "text_raw"
        if dimension.use_original_input
        else default_input_column
    )


def build_word_per_dictionary(
    dimension: DimensionConfig,
    dimension_cls,
    input_column: str,
):
    return dimension_cls(
        key=dimension.key,
        dictionary_name=dictionary_param(dimension),
        input_column=input_column,
        percentage=percentage_param(dimension),
        use_regex=not disabled_regexp_param(dimension),
        pos_tag=param(dimension, "pos_tag", None),
        pos_input_column=param(dimension, "pos_input_column", "tagged_pos"),
    )


def build_pattern_dimension(
    dimension: DimensionConfig,
    dimension_cls,
    input_column: str,
):
    return dimension_cls(
        key=dimension.key,
        pattern=param(dimension, "pattern", ""),
        input_column=input_column,
        percentage=percentage_param(dimension),
    )


def build_pos_tagging_tag(
    dimension: DimensionConfig,
    dimension_cls,
    input_column: str,
):
    return dimension_cls(
        key=dimension.key,
        input_column="tagged_pos",
        postagger_tag=param(dimension, "tag"),
        postagger_universal=param(dimension, "universal"),
    )


def build_dependency_depth_dimension(
    dimension: DimensionConfig,
    dimension_cls,
    input_column: str,
):
    return dimension_cls(
        key=dimension.key,
        input_column="tagged_dep",
        mode=param(dimension, "mode", "max"),
    )


def build_dependency_distance_dimension(
    dimension: DimensionConfig,
    dimension_cls,
    input_column: str,
):
    return dimension_cls(
        key=dimension.key,
        input_column="tagged_dep",
        mode=param(dimension, "mode", "mean"),
    )


def build_root_pos_tag_dimension(
    dimension: DimensionConfig,
    dimension_cls,
    input_column: str,
):
    return dimension_cls(
        key=dimension.key,
        input_column="tagged_pos",
        tagged_dep_column="tagged_dep",
        tag=param(dimension, "tag"),
    )


def build_passive_voice_dependency_dimension(
    dimension: DimensionConfig,
    dimension_cls,
    input_column: str,
):
    return dimension_cls(
        key=dimension.key,
        input_column="tagged_dep",
    )


def build_dependency_tag(
    dimension: DimensionConfig,
    dimension_cls,
    input_column: str,
):
    return dimension_cls(
        key=dimension.key,
        input_column="tagged_dep",
        deprel=param(dimension, "deprel"),
    )


def build_pos_tagging_expression(
    dimension: DimensionConfig,
    dimension_cls,
    input_column: str,
):
    return dimension_cls(
        key=dimension.key,
        pattern=param(dimension, "pattern", ""),
        input_column="tagged_pos",
        percentage=percentage_param(dimension),
    )


def build_ner_tagging_tag(
    dimension: DimensionConfig,
    dimension_cls,
    input_column: str,
):
    return dimension_cls(
        key=dimension.key,
        tag=param(dimension, "tag"),
        input_column="tagged_ner",
    )


def build_enclitics_personal_pronouns_dictionary(
    dimension: DimensionConfig,
    dimension_cls,
    input_column: str,
):
    return dimension_cls(
        key=dimension.key,
        dictionary_name=dictionary_param(dimension),
        input_column=input_column,
    )


def build_periphrasis_dimension(
    dimension: DimensionConfig,
    dimension_cls,
    input_column: str,
):
    return dimension_cls(
        key=dimension.key,
        auxiliar_verbs=param(dimension, "auxiliar_verbs", ""),
        input_column=input_column,
        tagged_pos_column="tagged_pos",
    )


def build_character_count_dimension(
    dimension: DimensionConfig,
    dimension_cls,
    input_column: str,
):
    chars = param(dimension, "character", "")

    if chars == "SPACE":
        chars = " "

    return dimension_cls(
        key=dimension.key,
        chars=chars,
        input_column=input_column,
    )


def build_text_raw_dimension(
    dimension: DimensionConfig,
    dimension_cls,
    input_column: str,
):
    return dimension_cls(
        key=dimension.key,
        input_column="text_raw",
    )


def build_grammatical_gender_dimension(
    dimension: DimensionConfig,
    dimension_cls,
    input_column: str,
):
    return dimension_cls(
        key=dimension.key,
        dictionary_name=dictionary_param(dimension),
        input_column=input_column,
        tagged_pos_column="tagged_pos",
        percentage=percentage_param(dimension),
        use_regex=not disabled_regexp_param(dimension),
    )


def build_language_dimension(
    dimension: DimensionConfig,
    dimension_cls,
    input_column: str,
):
    return dimension_cls(
        key=dimension.key,
        language=param(dimension, "language", ""),
        input_column=input_column,
    )


def build_rtie_dimension(
    dimension: DimensionConfig,
    dimension_cls,
    input_column: str,
):
    return dimension_cls(
        key=dimension.key,
        input_column=input_column,
        separator=param(dimension, "separator", "by-chunks"),
    )


def build_sentence_per_dictionary(
    dimension: DimensionConfig,
    dimension_cls,
    input_column: str,
):
    return dimension_cls(
        key=dimension.key,
        dictionary_name=dictionary_param(dimension),
        input_column=input_column,
        percentage=percentage_param(dimension),
    )


def build_twitter_reply_to_dimension(
    dimension: DimensionConfig,
    dimension_cls,
    input_column: str,
):
    return dimension_cls(
        key=dimension.key,
        dictionary_name=dictionary_param(dimension),
        input_column="text_raw",
    )


def build_verb_per_dictionary(
    dimension: DimensionConfig,
    dimension_cls,
    input_column: str,
):
    return dimension_cls(
        key=dimension.key,
        dictionary_name=dictionary_param(dimension),
        input_column=input_column,
        percentage=percentage_param(dimension),
    )


def build_word_unique_dimension(
    dimension: DimensionConfig,
    dimension_cls,
    input_column: str,
):
    return dimension_cls(
        key=dimension.key,
        input_column=input_column,
        percentage=percentage_param(dimension),
    )


def build_word_case(
    dimension: DimensionConfig,
    dimension_cls,
    input_column: str,
):
    return dimension_cls(
        key=dimension.key,
        comparator=(
            param(dimension, "word_comparator")
            or param(dimension, "comparator")
            or "upper"
        ),
        input_column="text_raw",
    )


def build_word_length_dimension(
    dimension: DimensionConfig,
    dimension_cls,
    input_column: str,
):
    return dimension_cls(
        key=dimension.key,
        length=param(dimension, "length"),
        comparator=param(dimension, "comparator", "="),
        input_column=input_column,
        percentage=percentage_param(dimension),
    )


DimensionBuilder = Callable[[DimensionConfig, object, str], object]


DIMENSION_BUILDERS: dict[str | None, DimensionBuilder] = {
    "WordPerDictionary": build_word_per_dictionary,
    "PatternDimension": build_pattern_dimension,
    "POSTaggingTag": build_pos_tagging_tag,
    "DependencyDepthDimension": build_dependency_depth_dimension,
    "DependencyDistanceDimension": build_dependency_distance_dimension,
    "RootPOSTagDimension": build_root_pos_tag_dimension,
    "PassiveVoiceDependencyDimension": build_passive_voice_dependency_dimension,
    "DependencyTag": build_dependency_tag,
    "POSTaggingExpression": build_pos_tagging_expression,
    "NERTaggingTag": build_ner_tagging_tag,
    "EncliticsPersonalPronounsDictionary": build_enclitics_personal_pronouns_dictionary,
    "PeriphrasisDimension": build_periphrasis_dimension,
    "CharacterCountDimension": build_character_count_dimension,
    "ErrorCapitalizationStartingWithLowerCaseDimension": build_text_raw_dimension,
    "ErrorStyleSentencesStartingWithNumbers": build_text_raw_dimension,
    "ErrorStyleSentencesStartingWithTheSameWord": build_text_raw_dimension,
    "GrammaticalGenderDimension": build_grammatical_gender_dimension,
    "LanguageDimension": build_language_dimension,
    "RTIEDimension": build_rtie_dimension,
    "RTIEDeviationDimension": build_rtie_dimension,
    "SentencePerDictionary": build_sentence_per_dictionary,
    "TwitterReplyToDimension": build_twitter_reply_to_dimension,
    "VerbPerDictionary": build_verb_per_dictionary,
    "WordUniqueDimension": build_word_unique_dimension,
    "WordCase": build_word_case,
    "WordLengthDimension": build_word_length_dimension,
}