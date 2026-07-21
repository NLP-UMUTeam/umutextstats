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


def compute_result(
    texts,
    entries,
    exceptions=None,
    percentage=True,
    use_regex=True,
    tagged_pos=None,
    pos_tag=None,
):
    loader = DummyDictionaryLoader(
        entries=entries,
        exceptions=exceptions or [],
    )

    dimension = WordPerDictionary(
        key="test",
        dictionary_name="test",
        input_column="text_norm",
        percentage=percentage,
        use_regex=use_regex,
        pos_tag=pos_tag,
        pos_input_column="tagged_pos",
        dictionary_loader=loader,
    )

    data = {
        "text_norm": texts,
    }

    if tagged_pos is not None:
        data["tagged_pos"] = tagged_pos

    df = pd.DataFrame(data)

    return dimension.compute_result(df)


def compute(
    texts,
    entries,
    exceptions=None,
    percentage=True,
    use_regex=True,
    tagged_pos=None,
    pos_tag=None,
):
    df = pd.DataFrame({"text_norm": texts})

    if tagged_pos is not None:
        df["tagged_pos"] = tagged_pos

    dim = WordPerDictionary(
        key="dict",
        dictionary_name="dummy",
        input_column="text_norm",
        percentage=percentage,
        use_regex=use_regex,
        dictionary_loader=DummyDictionaryLoader(entries, exceptions),
        pos_tag=pos_tag, 
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

def test_unrelated_regex_exception_does_not_subtract():
    result = compute(
        ["logro éxito fracaso mundo"],
        ["logro", "éxito"],
        exceptions=["fracaso"],
        percentage=False,
        use_regex=True,
    )

    assert result == [2]


def test_regex_exception_removes_covered_match():
    result = compute(
        ["logro éxito fracaso mundo"],
        ["logro", "éxito", "fracaso"],
        exceptions=["fracaso"],
        percentage=False,
        use_regex=True,
    )

    assert result == [2]


def test_regex_wildcard_exception_removes_covered_match():
    result = compute(
        ["logro éxito fracasado mundo"],
        ["logro", "éxito", r"fracas\p{L}*"],
        exceptions=[r"fracas\p{L}*"],
        percentage=False,
        use_regex=True,
    )

    assert result == [2]

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
    
    
# =========================
# POS filter
# =========================

def test_pos_filter_counts_only_matching_pos_regex():
    result = compute(
        texts=["casa amable terrible"],
        entries=[r"\p{L}{3,}[ai]?bles?"],
        percentage=False,
        use_regex=True,
        tagged_pos=[
            "casa__(NOUN)(Gender=Fem), "
            "amable__(ADJ)(Number=Sing), "
            "terrible__(ADJ)(Number=Sing)"
        ],
        pos_tag="ADJ",
    )

    assert result == [2]


def test_pos_filter_ignores_matching_regex_with_wrong_pos():
    result = compute(
        texts=["amable terrible"],
        entries=[r"\p{L}{3,}[ai]?bles?"],
        percentage=False,
        use_regex=True,
        tagged_pos=[
            "amable__(NOUN)(Number=Sing), "
            "terrible__(NOUN)(Number=Sing)"
        ],
        pos_tag="ADJ",
    )

    assert result == [0]


def test_pos_filter_mixed_pos_counts_only_allowed_words():
    result = compute(
        texts=["amable terrible posible"],
        entries=[r"\p{L}{3,}[ai]?bles?"],
        percentage=False,
        use_regex=True,
        tagged_pos=[
            "amable__(ADJ)(Number=Sing), "
            "terrible__(NOUN)(Number=Sing), "
            "posible__(ADJ)(Number=Sing)"
        ],
        pos_tag="ADJ",
    )

    assert result == [2]


def test_pos_filter_percentage_uses_total_words_denominator():
    result = compute(
        texts=["casa amable terrible corre"],
        entries=[r"\p{L}{3,}[ai]?bles?"],
        percentage=True,
        use_regex=True,
        tagged_pos=[
            "casa__(NOUN)(Gender=Fem), "
            "amable__(ADJ)(Number=Sing), "
            "terrible__(ADJ)(Number=Sing), "
            "corre__(VERB)(Number=Sing)"
        ],
        pos_tag="ADJ",
    )

    assert result == [50.0]


def test_pos_filter_plain_mode():
    result = compute(
        texts=["hola mundo hola"],
        entries=["hola"],
        percentage=False,
        use_regex=False,
        tagged_pos=[
            "hola__(INTJ), mundo__(NOUN), hola__(NOUN)"
        ],
        pos_tag="INTJ",
    )

    assert result == [1]


def test_pos_filter_plain_mode_ignores_wrong_pos():
    result = compute(
        texts=["hola hola"],
        entries=["hola"],
        percentage=False,
        use_regex=False,
        tagged_pos=[
            "hola__(NOUN), hola__(NOUN)"
        ],
        pos_tag="INTJ",
    )

    assert result == [0]


def test_pos_filter_with_exceptions_regex():
    result = compute(
        texts=["amable terrible posible"],
        entries=[r"\p{L}{3,}[ai]?bles?"],
        exceptions=["terrible"],
        percentage=False,
        use_regex=True,
        tagged_pos=[
            "amable__(ADJ), terrible__(ADJ), posible__(ADJ)"
        ],
        pos_tag="ADJ",
    )

    assert result == [2]


def test_pos_filter_missing_tagged_pos_returns_zero():
    result = compute(
        texts=["amable terrible"],
        entries=[r"\p{L}{3,}[ai]?bles?"],
        percentage=False,
        use_regex=True,
        tagged_pos=[""],
        pos_tag="ADJ",
    )

    assert result == [0]


def test_pos_filter_disabled_keeps_original_behavior():
    result = compute(
        texts=["amable terrible"],
        entries=[r"\p{L}{3,}[ai]?bles?"],
        percentage=False,
        use_regex=True,
        tagged_pos=[
            "amable__(NOUN), terrible__(NOUN)"
        ],
        pos_tag=None,
    )

    assert result == [2]
    
    
def test_pos_filter_percentage_uses_word_count_column_if_available():
    df = pd.DataFrame(
        {
            "text_norm": ["amable terrible"],
            "tagged_pos": ["amable__(ADJ), terrible__(ADJ)"],
            "word_count": [4],
        }
    )

    dim = WordPerDictionary(
        key="dict",
        dictionary_name="dummy",
        input_column="text_norm",
        percentage=True,
        use_regex=True,
        dictionary_loader=DummyDictionaryLoader(
            [r"\p{L}{3,}[ai]?bles?"]
        ),
        pos_tag="ADJ",
    )

    result = list(dim.compute(df))

    assert result == [50.0]
    
    
def test_pos_filter_accepts_multiple_pos_tags_regex():
    result = compute(
        texts=["casita pequeñita rápidamente"],
        entries=[r"\p{L}{2,}itas?"],
        percentage=False,
        use_regex=True,
        tagged_pos=[
            "casita__(NOUN)(Gender=Fem|Number=Sing), "
            "pequeñita__(ADJ)(Gender=Fem|Number=Sing), "
            "rápidamente__(ADV)"
        ],
        pos_tag=["NOUN", "ADJ"],
    )

    assert result == [2]


def test_pos_filter_multiple_pos_tags_ignores_non_allowed_pos():
    result = compute(
        texts=["casita pequeñita rápidamente"],
        entries=[r"\p{L}{2,}itas?"],
        percentage=False,
        use_regex=True,
        tagged_pos=[
            "casita__(NOUN)(Gender=Fem|Number=Sing), "
            "pequeñita__(ADJ)(Gender=Fem|Number=Sing), "
            "rápidamente__(ADV)"
        ],
        pos_tag=["NOUN"],
    )

    assert result == [1]


def test_pos_filter_accepts_multiple_pos_tags_plain_mode():
    result = compute(
        texts=["casita pequeñita rápidamente"],
        entries=["casita", "pequeñita", "rápidamente"],
        percentage=False,
        use_regex=False,
        tagged_pos=[
            "casita__(NOUN)(Gender=Fem|Number=Sing), "
            "pequeñita__(ADJ)(Gender=Fem|Number=Sing), "
            "rápidamente__(ADV)"
        ],
        pos_tag=["NOUN", "ADJ"],
    )

    assert result == [2]

def test_pos_filter_evidence_matches_result_plain_mode():
    text = "casita pequeñita rápidamente"

    result = compute_result(
        texts=[text],
        entries=[
            "casita",
            "pequeñita",
            "rápidamente",
        ],
        percentage=False,
        use_regex=False,
        tagged_pos=[
            "casita__(NOUN)(Gender=Fem|Number=Sing), "
            "pequeñita__(ADJ)(Gender=Fem|Number=Sing), "
            "rápidamente__(ADV)"
        ],
        pos_tag=["NOUN", "ADJ"],
    )

    assert result.values.tolist() == [2]
    assert result.numerators.tolist() == [2]

    evidence = result.evidence.iloc[0]

    assert evidence == [
        {
            "text": "casita",
            "start": 0,
            "end": 6,
        },
        {
            "text": "pequeñita",
            "start": 7,
            "end": 16,
        },
    ]

    for item in evidence:
        assert (
            text[item["start"]:item["end"]]
            == item["text"]
        )