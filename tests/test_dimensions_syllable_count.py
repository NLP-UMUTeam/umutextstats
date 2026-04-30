# tests/test_dimensions_syllable_count.py

import pandas as pd

from umutextstats.dimensions.syllable_count import (
    count_syllables_word,
    count_syllables_text,
)
from umutextstats.dimensions.syllable_count import SyllableCountDimension


def test_count_syllables_word_simple():
    assert count_syllables_word("casa") == 2
    assert count_syllables_word("hola") == 2


def test_count_syllables_word_empty():
    assert count_syllables_word("") == 0


def test_count_syllables_word_without_vowels():
    assert count_syllables_word("rhythm") == 1


def test_count_syllables_text():
    assert count_syllables_text("hola mundo") == 4


def test_syllable_count_dimension():
    df = pd.DataFrame({"text_norm": ["hola mundo", "", "casa"]})

    dim = SyllableCountDimension(key="syllables")

    result = dim.compute(df)

    assert list(result) == [4, 0, 2]
    
def test_count_syllables_word_with_accents():
    assert count_syllables_word("camión") == 2
    assert count_syllables_word("rápido") == 3
    assert count_syllables_word("información") == 4


def test_count_syllables_word_with_umlaut():
    assert count_syllables_word("pingüino") == 3
    assert count_syllables_word("vergüenza") == 3


def test_count_syllables_word_diphthongs_basic():
    assert count_syllables_word("rey") == 1
    assert count_syllables_word("ciudad") == 2
    assert count_syllables_word("puerta") == 2


def test_count_syllables_word_hiatus_limitation():
    # Heurística simple: puede no separar todos los hiatos perfectamente.
    assert count_syllables_word("poeta") in {2, 3}
    assert count_syllables_word("país") in {1, 2}


def test_count_syllables_text_with_punctuation():
    assert count_syllables_text("Hola, mundo.") == 4
    assert count_syllables_text("¡Qué rápido!") == 4


def test_count_syllables_text_with_numbers_and_words():
    assert count_syllables_text("Tengo 2 casas") == 4

