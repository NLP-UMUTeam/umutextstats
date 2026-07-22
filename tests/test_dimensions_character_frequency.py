# tests/test_dimensions_character_count.py

import pandas as pd

from umutextstats.dimensions.count import CharacterFrequencyDimension


def compute(texts, chars):
    df = pd.DataFrame({"text_norm": texts})
    dim = CharacterFrequencyDimension(key="chars", chars=chars)
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
    assert compute(["a b\tc\nd"], r"\s") == [
        100 * 3 / 7
    ]

def compute_result(texts, chars):
    df = pd.DataFrame(
        {
            "text_norm": texts,
        }
    )

    dimension = CharacterFrequencyDimension(
        key="chars",
        chars=chars,
    )

    return dimension.compute_result(df)

def test_character_result_contains_counts():
    result = compute_result(
        ["banana"],
        "an",
    )

    assert result.values.tolist() == [
        100 * 5 / 6
    ]

    assert result.numerators.tolist() == [5]
    assert result.denominators.tolist() == [6]

def test_character_result_contains_evidence():
    text = "banana"

    result = compute_result(
        [text],
        "an",
    )

    evidence = result.evidence.iloc[0]

    assert len(evidence) == 5

    assert [
        item["text"]
        for item in evidence
    ] == [
        "a",
        "n",
        "a",
        "n",
        "a",
    ]

    for item in evidence:
        assert (
            text[
                item["start"]:
                item["end"]
            ]
            == item["text"]
        )

def test_character_result_deduplicates_configured_chars():
    result = compute_result(
        ["aaaa"],
        "aa",
    )

    assert result.values.tolist() == [100.0]
    assert result.numerators.tolist() == [4]
    assert result.denominators.tolist() == [4]
    assert len(result.evidence.iloc[0]) == 4

def test_whitespace_result_contains_evidence():
    text = "a b\tc\nd"

    result = compute_result(
        [text],
        r"\s",
    )

    assert result.numerators.tolist() == [3]
    assert result.denominators.tolist() == [7]
    assert result.values.tolist() == [
        100 * 3 / 7
    ]

    assert [
        item["text"]
        for item in result.evidence.iloc[0]
    ] == [
        " ",
        "\t",
        "\n",
    ]

def test_empty_text_result():
    result = compute_result(
        [""],
        "a",
    )

    assert result.values.tolist() == [0.0]
    assert result.numerators.tolist() == [0]
    assert result.denominators.tolist() == [0]
    assert result.evidence.iloc[0] == []

def test_character_compute_matches_compute_result():
    texts = [
        "hola",
        "banana",
        "",
        "niño",
        "a b\tc\nd",
    ]

    df = pd.DataFrame(
        {
            "text_norm": texts,
        }
    )

    dimension = CharacterFrequencyDimension(
        key="chars",
        chars="a",
    )

    expected = dimension.compute(df)
    actual = dimension.compute_result(df).values

    pd.testing.assert_series_equal(
        actual,
        expected,
        check_dtype=False,
        check_names=False,
    )

def test_whitespace_compute_matches_compute_result():
    df = pd.DataFrame(
        {
            "text_norm": [
                "hola mundo",
                "hola\nmundo",
                "hola\tmundo",
                "",
            ],
        }
    )

    dimension = CharacterFrequencyDimension(
        key="whitespace",
        chars=r"\s",
    )

    expected = dimension.compute(df)
    actual = dimension.compute_result(df).values

    pd.testing.assert_series_equal(
        actual,
        expected,
        check_dtype=False,
        check_names=False,
    )