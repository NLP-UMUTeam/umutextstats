import pandas as pd

from umutextstats.dimensions.average_word_length import AverageWordLengthDimension


def compute(texts):
    df = pd.DataFrame({"text_norm": texts})
    dim = AverageWordLengthDimension(key="avg_word_length") 
    return list(dim.compute(df))


def test_average_word_length_simple():
    assert compute(["hola mundo"]) == [4.5]


def test_average_word_length_different_lengths():
    assert compute(["yo casa"]) == [3.0]


def test_average_word_length_counts_numbers():
    assert compute(["yo tengo 2 casas"]) == [3.25]


def test_average_word_length_empty():
    assert compute([""]) == [0.0]


def test_average_word_length_punctuation():
    assert compute(["hola, mundo!"]) == [4.5]
    
    
def test_average_word_length_decimal_comma_counts_as_one_token():
    # Vale(4), 3,14(4), euros(5)
    assert compute(["Vale 3,14 euros"]) == [13 / 3]


def test_average_word_length_decimal_dot_counts_as_one_token():
    # Vale(4), 3.14(4), euros(5)
    assert compute(["Vale 3.14 euros"]) == [13 / 3]


def test_average_word_length_hyphenated_word_counts_as_one_token():
    # texto(5), teórico-práctico(16)
    assert compute(["texto teórico-práctico"]) == [10.5]


def test_average_word_length_apostrophe_word_counts_as_one_token():
    # l'amour(7), existe(6)
    assert compute(["l'amour existe"]) == [6.5]


def test_average_word_length_unicode_words():
    # Müller(6), naïve(5), niño(4)
    assert compute(["Müller naïve niño"]) == [5.0]


def test_average_word_length_emojis_are_ignored():
    assert compute(["hola 😊 mundo"]) == [4.5]


def test_average_word_length_only_emojis():
    assert compute(["😊 😂 😍"]) == [0.0]


def test_average_word_length_whitespace_only():
    assert compute(["   \t\n  "]) == [0.0]


def test_average_word_length_multiple_rows():
    assert compute(["hola mundo", "", "yo tengo 2 casas"]) == [4.5, 0.0, 3.25]