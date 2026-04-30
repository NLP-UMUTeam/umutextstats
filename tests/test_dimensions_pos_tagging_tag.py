# tests/test_dimensions_pos_tagging_tag.py

import pandas as pd

from umutextstats.dimensions.pos_tagging_tag import POSTaggingTag


def compute(tagged_pos, universal=None, tag=None):
    df = pd.DataFrame({"tagged_pos": tagged_pos})

    dim = POSTaggingTag(
        key="pos",
        postagger_tag=tag,
        postagger_universal=universal,
    )

    return list(dim.compute(df))


def test_gender_feminine_words():
    tagged = [
        "casa__(NOUN)(Gender=Fem|Number=Sing), bonito__(ADJ)(Gender=Masc|Number=Sing)"
    ]

    result = compute(tagged, universal="Gender=Fem")

    assert result == [50.0]


def test_gender_masculine_words():
    tagged = [
        "casa__(NOUN)(Gender=Fem|Number=Sing), bonito__(ADJ)(Gender=Masc|Number=Sing)"
    ]

    result = compute(tagged, universal="Gender=Masc")

    assert result == [50.0]


def test_pos_tag_only():
    tagged = [
        "yo__(PRON)(Number=Sing), como__(VERB)(Mood=Ind), casa__(NOUN)(Gender=Fem)"
    ]

    result = compute(tagged, tag="VERB")

    assert round(result[0], 2) == 33.33


def test_tag_and_universal():
    tagged = [
        "casa__(NOUN)(Gender=Fem), bonita__(ADJ)(Gender=Fem), perro__(NOUN)(Gender=Masc)"
    ]

    result = compute(tagged, tag="NOUN", universal="Gender=Fem")

    assert round(result[0], 2) == 33.33


def test_empty_tagged_pos():
    assert compute([""], universal="Gender=Fem") == [0.0]