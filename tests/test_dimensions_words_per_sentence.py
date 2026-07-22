import pandas as pd

from umutextstats.dimensions.words_per_sentence import WordPerSentenceDimension


def compute(texts):
    df = pd.DataFrame({"text_norm": texts})
    dim = WordPerSentenceDimension(key="wps")
    return list(dim.compute(df))


def test_accents_and_enye():
    assert compute(["El niño comió piña."]) == [4.0]


def test_inverted_question_marks():
    assert compute(["¿Hola mundo?"]) == [2.0]


def test_inverted_exclamation_marks():
    assert compute(["¡Qué día bonito!"]) == [3.0]


def test_multiple_punctuation_counts_as_one_sentence_boundary():
    assert compute(["Hola mundo!!!"]) == [2.0]


def test_ellipsis_counts_as_one_sentence_boundary():
    assert compute(["Hola mundo..."]) == [2.0]


def test_emojis_do_not_count_as_words():
    assert compute(["Hola mundo 😊."]) == [2.0]


def test_only_emojis():
    assert compute(["😊 😂 😍"]) == [0.0]


def test_only_punctuation():
    assert compute(["!!! ... ¿?"]) == [0.0]


def test_quotes_around_sentence():
    assert compute(['"Hola mundo."']) == [2.0]


def test_spanish_quotes_around_sentence():
    assert compute(["«Hola mundo.»"]) == [2.0]


def test_multiple_spaces_tabs_and_newlines():
    assert compute(["hola   mundo\tcruel\nhoy."]) == [4.0]


def test_two_sentences_with_question_and_exclamation():
    assert compute(["Hola mundo. ¿Casa bonita?"]) == [2.0]


def test_numbers_count_as_tokens_if_liwc_like():
    assert compute(["Tengo 2 perros."]) == [3.0]


def test_hashtag_counts_as_word_or_not_document_decision():
    assert compute(["Me gusta #python."]) == [3.0]


def test_mention_counts_as_word_or_not_document_decision():
    assert compute(["Hola @usuario."]) == [2.0]


def test_abbreviation_should_not_split_sentence():
    assert compute(["El Sr. García vino."]) == [4.0]


def test_decimal_comma_number_should_not_split_sentence():
    assert compute(["Vale 3,14 euros."]) == [3.0]


def test_empty_whitespace():
    assert compute(["   \t\n  "]) == [0.0]
    
    
def test_numbers_are_counted():
    assert compute(["Tengo 2 perros."]) == [3.0]


def test_unicode_letters():
    assert compute(["Müller naïve niño."]) == [3.0]


def test_hyphenated_word_counts_as_one():
    assert compute(["texto teórico-práctico."]) == [2.0]


def test_apostrophe_word_counts_as_one():
    assert compute(["l'amour existe."]) == [2.0]


def test_emojis_are_not_words():
    assert compute(["hola 😊 mundo."]) == [2.0]

def test_words_per_sentence_result():
    df = pd.DataFrame(
        {
            "text_norm": [
                "Uno dos tres. Cuatro cinco."
            ]
        }
    )

    dimension = WordPerSentenceDimension(
        key="words_per_sentence"
    )

    result = dimension.compute_result(df)

    assert result.numerators.tolist() == [5]
    assert result.denominators.tolist() == [2]
    assert result.values.tolist() == [2.5]
    assert result.evidence is None

def test_words_per_sentence_compute_matches_result():
    df = pd.DataFrame(
        {
            "text_norm": [
                "Uno dos tres. Cuatro cinco.",
                "",
                "Texto sin puntuación",
            ]
        }
    )

    dimension = WordPerSentenceDimension(
        key="words_per_sentence"
    )

    expected = dimension.compute(df)
    actual = dimension.compute_result(df).values

    pd.testing.assert_series_equal(
        actual,
        expected,
        check_dtype=False,
        check_names=False,
    )

def test_words_per_sentence_empty_text():
    df = pd.DataFrame(
        {
            "text_norm": [""]
        }
    )

    result = WordPerSentenceDimension(
        key="words_per_sentence"
    ).compute_result(df)

    assert result.values.tolist() == [0.0]
    assert result.numerators.tolist() == [0]
    assert result.denominators.tolist() == [0]