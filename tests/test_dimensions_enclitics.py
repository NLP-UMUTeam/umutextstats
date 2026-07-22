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


def test_enclitics_result_contains_evidence():
    df = pd.DataFrame(
        {
            "text_norm": [
                "Quiero dámelo mañana",
            ],
        }
    )

    dimension = EncliticsPersonalPronounsDictionary(
        key="enclitics",
        dictionary_name="verbs",
        dictionary_loader=DummyDictionaryLoader(),
    )

    result = dimension.compute_result(df)

    assert result.numerators.tolist() == [1]
    assert result.denominators.tolist() == [3]
    assert result.values.tolist() == [
        100.0 / 3
    ]

    evidence = result.evidence.iloc[0]

    assert len(evidence) == 1
    assert evidence[0]["text"] == "damelo"

    normalized = remove_accents(
        df.iloc[0]["text_norm"].lower()
    )

    assert (
        normalized[
            evidence[0]["start"]:
            evidence[0]["end"]
        ]
        == evidence[0]["text"]
    )

def test_enclitics_compute_single_matches_compute_result():
    df = pd.DataFrame(
        {
            "text_norm": [
                "Quiero dámelo mañana",
                "No hay coincidencias",
                "",
            ],
        }
    )

    dimension = EncliticsPersonalPronounsDictionary(
        key="enclitics",
        dictionary_name="verbs",
        dictionary_loader=DummyDictionaryLoader(),
    )

    expected = df.apply(
        dimension.compute_single,
        axis=1,
    )

    actual = dimension.compute_result(df).values

    pd.testing.assert_series_equal(
        actual,
        expected,
        check_dtype=False,
        check_names=False,
    )

def test_enclitics_result_contains_multiple_matches():
    df = pd.DataFrame(
        {
            "text_norm": [
                "Dámelo y cómetelo",
            ],
        }
    )

    dimension = EncliticsPersonalPronounsDictionary(
        key="enclitics",
        dictionary_name="verbs",
        dictionary_loader=DummyDictionaryLoader(),
    )

    result = dimension.compute_result(df)

    assert result.numerators.tolist() == [2]

    assert [
        item["text"]
        for item in result.evidence.iloc[0]
    ] == [
        "damelo",
        "cometelo",
    ]