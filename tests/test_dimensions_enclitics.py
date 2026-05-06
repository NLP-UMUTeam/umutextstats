import pandas as pd

from umutextstats.dictionaries import DictionaryEntries
from umutextstats.dimensions.enclitics_personal_pronouns import (
    EncliticsPersonalPronounsDictionary,
    remove_accents,
)


class DummyDictionaryLoader:
    def load(self, name):
        return DictionaryEntries(
            words=["da", "hacer", "explicar", "come"],
            exceptions=[],
        )


def compute(texts):
    df = pd.DataFrame({"text_norm": texts})

    dim = EncliticsPersonalPronounsDictionary(
        key="enclitics",
        dictionary_name="dummy",
        dictionary_loader=DummyDictionaryLoader(),
    )

    return list(dim.compute(df))


def test_remove_accents():
    assert remove_accents("dámelo") == "damelo"


def test_enclitic_basic():
    result = compute(["quiero hacerlo ahora"])
    assert round(result[0], 2) == 33.33


def test_enclitic_with_accent():
    result = compute(["dámelo ahora"])
    assert round(result[0], 2) == 50.0


def test_enclitic_multiple():
    result = compute(["dámelo y explicarnos"])
    assert round(result[0], 2) == 66.67


def test_no_match():
    result = compute(["quiero hacer esto"])
    assert result == [0.0]


def test_empty():
    result = compute([""])
    assert result == [0.0]

