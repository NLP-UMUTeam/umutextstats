# tests/test_dimensions_character_count.py

import pandas as pd

from umutextstats.dimensions.character_count import CharacterCountDimension


def compute(texts, chars):
    df = pd.DataFrame({"text_norm": texts})
    dim = CharacterCountDimension(key="chars", chars=chars)
    return list(dim.compute(df))


def test_single_character_count_percentage():
    assert compute(["hola"], "a") == [25.0]


def test_multiple_character_count_percentage():
    assert compute(["banana"], "an") == [83.33333333333333]


def test_no_matches():
    assert compute(["hola"], "z") == [0.0]


def test_empty_text():
    assert compute([""], "a") == [0.0]


def test_empty_chars():
    assert compute(["hola"], "") == [0.0]


def test_multiple_rows():
    result = compute(["hola", "banana", ""], "a")

    assert result == [25.0, 50.0, 0.0]


def test_accents():
    assert compute(["camión"], "ó") == [16.666666666666668]
    
def test_uppercase_is_case_sensitive():
    assert compute(["Hola"], "h") == [0.0]


def test_uppercase_exact_match():
    assert compute(["Hola"], "H") == [25.0]


def test_space_character():
    assert compute(["hola mundo"], " ") == [10.0]


def test_emoji_character():
    assert compute(["hola 😊"], "😊") == [100 / 6]


def test_multiple_emojis():
    assert compute(["😊😊😊"], "😊") == [100.0]


def test_unicode_enye():
    assert compute(["niño"], "ñ") == [25.0]


def test_unicode_umlaut():
    assert compute(["pingüino"], "ü") == [12.5]


def test_repeated_chars_parameter_is_deduplicated():
    assert compute(["aaaa"], "aa") == [100.0]
    

def test_newline_character():
    assert compute(["hola\nmundo"], "\n") == [10.0]


def test_tab_character():
    assert compute(["hola\tmundo"], "\t") == [10.0]


def test_only_spaces():
    assert compute(["   "], " ") == [100.0]


def test_multiple_rows_with_unicode():
    result = compute(["niño", "camión", ""], "ó")

    assert result == [0.0, 16.666666666666668, 0.0]


def test_whitespace_regex_counts_spaces():
    assert compute(["hola mundo"], r"\s") == [10.0]


def test_whitespace_regex_counts_newlines():
    assert compute(["hola\nmundo"], r"\s") == [10.0]


def test_whitespace_regex_counts_tabs():
    assert compute(["hola\tmundo"], r"\s") == [10.0]


def test_whitespace_regex_counts_mixed_whitespace():
    assert compute(["a b\tc\nd"], r"\s") == [50.0]


def test_whitespace_regex_counts_mixed_whitespace():
    result = compute(["a b\tc\nd"], r"\s")

    assert result[0] == 100 * 3 / 7