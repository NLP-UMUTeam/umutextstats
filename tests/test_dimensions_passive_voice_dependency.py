import pandas as pd

from umutextstats.dimensions.passive_voice_dependency import PassiveVoiceDependencyDimension

from umutextstats.text.patterns import DEPENDENCY_ITEM_REGEX

def compute(tagged_dep):
    df = pd.DataFrame({
        "tagged_dep": tagged_dep,
    })

    dim = PassiveVoiceDependencyDimension(
        key="passive",
    )

    return list(dim.compute(df))


def test_detects_aux_pass():
    tagged = [
        (
            "La__(det)(2), carta__(nsubj:pass)(4), "
            "fue__(aux:pass)(4), escrita__(root)(0)"
        )
    ]

    result = compute(tagged)

    assert result == [100.0]


def test_detects_nsubj_pass():
    tagged = [
        (
            "Los__(det)(2), libros__(nsubj:pass)(4), "
            "fueron__(aux)(4), vendidos__(root)(0)"
        )
    ]

    result = compute(tagged)

    assert result == [100.0]


def test_detects_expl_pass():
    tagged = [
        (
            "Se__(expl:pass)(2), vende__(root)(0), "
            "casa__(obj)(2)"
        )
    ]

    result = compute(tagged)

    assert result == [100.0]


def test_active_voice_returns_zero():
    tagged = [
        (
            "Juan__(nsubj)(2), come__(root)(0), "
            "pan__(obj)(2)"
        )
    ]

    result = compute(tagged)

    assert result == [0.0]


def test_mixed_sentences():
    tagged = [
        (
            "Juan__(nsubj)(2), come__(root)(0) || "
            "La__(det)(2), carta__(nsubj:pass)(4), "
            "fue__(aux:pass)(4), escrita__(root)(0)"
        )
    ]

    result = compute(tagged)

    assert result == [50.0]


def test_empty_input():
    result = compute([""])

    assert result == [0.0]

def test_passive_voice_result_contains_evidence():
    tagged = (
        "La__(det)(2), "
        "casa__(nsubj:pass)(4), "
        "fue__(aux:pass)(4), "
        "construida__(root)(0)"
        " || "
        "Pedro__(nsubj)(2), "
        "corre__(root)(0)"
    )

    df = pd.DataFrame(
        {
            "tagged_dep": [tagged],
        }
    )

    dimension = PassiveVoiceDependencyDimension(
        key="passive_voice",
    )

    result = dimension.compute_result(df)

    assert result.values.tolist() == [50.0]
    assert result.numerators.tolist() == [1]
    assert result.denominators.tolist() == [2]

    evidence = result.evidence.iloc[0]

    assert len(evidence) == 1
    assert result.numerators.iloc[0] == len(evidence)

    passive = evidence[0]

    assert [
        trigger["deprel"]
        for trigger in passive["triggers"]
    ] == [
        "nsubj:pass",
        "aux:pass",
    ]

    assert (
        tagged[
            passive["start"]:
            passive["end"]
        ]
        == passive["text"]
    )

    for trigger in passive["triggers"]:
        assert (
            tagged[
                trigger["start"]:
                trigger["end"]
            ]
            == trigger["text"]
        )

def test_passive_voice_compute_matches_compute_result():
    tagged = (
        "La__(det)(2), "
        "casa__(nsubj:pass)(4), "
        "fue__(aux:pass)(4), "
        "construida__(root)(0)"
        " || "
        "Pedro__(nsubj)(2), "
        "corre__(root)(0)"
    )

    df = pd.DataFrame(
        {
            "tagged_dep": [tagged],
        }
    )

    dimension = PassiveVoiceDependencyDimension(
        key="passive_voice",
    )

    expected = dimension.compute(df)
    actual = dimension.compute_result(df).values

    pd.testing.assert_series_equal(
        actual,
        expected,
        check_dtype=False,
        check_names=False,
    )