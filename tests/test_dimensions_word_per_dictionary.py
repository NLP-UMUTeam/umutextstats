import pandas as pd

from umutextstats.dimensions.word_per_dictionary import WordPerDictionary
from umutextstats.dictionaries import DictionaryEntries

class DummyDictionaryLoader:
    def __init__(self, entries, exceptions=None):
        self.entries = entries
        self.exceptions = exceptions or []

    def load(self, name):
        return DictionaryEntries(
            words=self.entries,
            exceptions=self.exceptions,
        )


def compute(
    texts,
    entries,
    exceptions=None,
    percentage=True,
    use_regex=True,
):
    df = pd.DataFrame({"text_norm": texts})

    dim = WordPerDictionary(
        key="dict",
        dictionary_name="dummy",
        input_column="text_norm",
        percentage=percentage,
        use_regex=use_regex,
        dictionary_loader=DummyDictionaryLoader(entries, exceptions),
    )

    return list(dim.compute(df))


# =========================
# Conteo básico
# =========================

def test_basic_count():
    result = compute(["hola mundo"], ["hola"])
    assert result == [50.0]  # 1/2 palabras


def test_multiple_matches():
    result = compute(["hola hola mundo"], ["hola"])
    assert round(result[0], 2) == 66.67


def test_no_matches():
    result = compute(["hola mundo"], ["perro"])
    assert result == [0.0]


# =========================
# Sin porcentaje
# =========================

def test_count_without_percentage():
    result = compute(["hola hola mundo"], ["hola"], percentage=False)
    assert result == [2]


# =========================
# Edge cases
# =========================

def test_empty_text():
    result = compute([""], ["hola"])
    assert result == [0]


def test_only_punctuation():
    result = compute(["!!!"], ["hola"])
    assert result == [0]


def test_multiple_rows():
    texts = [
        "hola mundo",
        "hola hola",
        "perro gato",
    ]

    result = compute(texts, ["hola"])

    assert result[0] == 50.0
    assert result[1] == 100.0
    assert result[2] == 0.0


# =========================
# Regex vs no regex
# =========================

def test_regex_mode():
    result = compute(["animalito animal"], ["animal\\p{L}*"])
    assert result == [100.0]
    
def test_regex_mode_matches_variants():
    result = compute(["logro logramos mundo"], ["logr\\p{L}+"])
    assert round(result[0], 2) == 66.67


def test_plain_mode():
    result = compute(["animalito animal"], ["animal"], use_regex=False)
    assert result == [50.0]
    
    
# =========================
# Exceptions
# =========================

def test_regex_exceptions_are_subtracted():
    result = compute(
        ["logro éxito fracaso mundo"],
        ["logro", "éxito"],
        exceptions=["fracaso"],
        percentage=False,
        use_regex=True,
    )

    assert result == [1]


def test_plain_exceptions_are_subtracted():
    result = compute(
        ["soy estoy tengo"],
        ["soy", "estoy", "tengo"],
        exceptions=["estoy"],
        percentage=False,
        use_regex=False,
    )

    assert result == [2]


def test_exceptions_do_not_go_below_zero():
    result = compute(
        ["fracaso"],
        ["logro"],
        exceptions=["fracaso"],
        percentage=False,
        use_regex=False,
    )

    assert result == [0]