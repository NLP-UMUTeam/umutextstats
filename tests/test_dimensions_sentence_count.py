import pandas as pd

from umutextstats.dimensions.sentence_count import SentenceCountDimension


def compute(texts):
    df = pd.DataFrame({"text_norm": texts})
    dim = SentenceCountDimension(key="sentences")
    return list(dim.compute(df))


# =========================
# Casos básicos
# =========================

def test_single_sentence_no_punctuation():
    assert compute(["hola mundo"]) == [1]


def test_single_sentence_with_period():
    assert compute(["hola mundo."]) == [1]


def test_two_sentences():
    assert compute(["hola mundo. adiós mundo."]) == [2]


# =========================
# Puntuación variada
# =========================

def test_exclamation_and_question():
    assert compute(["hola! qué tal?"]) == [2]


def test_multiple_punctuation_marks():
    assert compute(["hola!!! qué tal???"]) == [2]


# =========================
# Edge cases
# =========================

def test_empty_string():
    assert compute([""]) == [0]


def test_only_spaces():
    assert compute(["   "]) == [0]


def test_no_text_but_symbols():
    assert compute(["!!!"]) == [1]


# =========================
# Casos reales
# =========================

def test_spanish_sentence():
    assert compute(["¡Hola! ¿Cómo estás? Todo bien."]) == [3]


def test_sentence_without_final_punctuation():
    assert compute(["Esto es una frase sin punto"]) == [1]


def test_mixed_case():
    assert compute(["Hola mundo. Esto es una prueba sin final"]) == [1]


# =========================
# Múltiples filas
# =========================

def test_multiple_rows():
    texts = [
        "hola mundo",
        "hola mundo.",
        "hola. mundo.",
        "",
    ]

    assert compute(texts) == [1, 1, 2, 0]