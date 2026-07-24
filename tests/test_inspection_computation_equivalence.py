from __future__ import annotations

import pytest

from dataclasses import dataclass
from typing import Any

import pandas as pd

from umutextstats.dimensions.periphrasis import (
    PeriphrasisDimension,
)
from umutextstats.dimensions.word_per_dictionary import (
    WordPerDictionary,
)

from umutextstats.dimensions.grammatical_gender import (
    GrammaticalGenderDimension,
)

NormalizedMatch = tuple[
    str | None,
    int | None,
    int | None,
]


def normalize_evidence(
    evidence: list[dict[str, Any]],
) -> list[NormalizedMatch]:
    return [
        (
            item.get("text"),
            item.get("start"),
            item.get("end"),
        )
        for item in evidence
    ]


def normalize_inspection_matches(
    matches,
) -> list[NormalizedMatch]:
    return [
        (
            match.match,
            match.start,
            match.end,
        )
        for match in matches
    ]


@dataclass(frozen=True)
class FakeDictionaryEntries:
    words: list[str]
    exceptions: list[str]


class FakeDictionaryLoader:
    def __init__(
        self,
        *,
        words: list[str],
        exceptions: list[str] | None = None,
    ) -> None:
        self.entries = FakeDictionaryEntries(
            words=words,
            exceptions=exceptions or [],
        )

    def load(
        self,
        name: str,
    ) -> FakeDictionaryEntries:
        return self.entries


def assert_inspection_matches_evidence(
    *,
    computation,
    inspection,
    row_index: int = 0,
) -> None:
    evidence = computation.evidence.iloc[
        row_index
    ]

    assert normalize_evidence(
        evidence
    ) == normalize_inspection_matches(
        inspection.matches
    )


def test_periphrasis_inspect_matches_computation_evidence():
    dimension = PeriphrasisDimension(
        key="test-periphrasis",
        auxiliar_verbs="estoy+gerund",
    )

    text = "Hoy estoy trabajando"

    df = pd.DataFrame(
        {
            "text_norm": [
                text,
            ],
            "tagged_pos": [
                (
                    "Hoy__(ADV), "
                    "estoy__(AUX)"
                    "(Mood=Ind|VerbForm=Fin), "
                    "trabajando__(VERB)"
                    "(VerbForm=Ger)"
                )
            ],
        }
    )

    computation = dimension.compute_result(
        df
    )

    inspection = dimension.inspect(
        df.iloc[0]
    )

    assert computation.values.iloc[0] == 1.0
    assert computation.numerators.iloc[0] == 1

    assert_inspection_matches_evidence(
        computation=computation,
        inspection=inspection,
    )

    expected_text = "estoy trabajando"
    expected_start = text.index(
        expected_text
    )
    expected_end = (
        expected_start
        + len(expected_text)
    )

    assert normalize_evidence(
        computation.evidence.iloc[0]
    ) == [
        (
            expected_text,
            expected_start,
            expected_end,
        )
    ]


def test_word_per_dictionary_non_regex_matches_are_equivalent():
    loader = FakeDictionaryLoader(
        words=[
            "problema",
            "triste",
        ],
    )

    dimension = WordPerDictionary(
        key="test-dictionary",
        dictionary_name="test",
        percentage=False,
        use_regex=False,
        dictionary_loader=loader,
    )

    text = (
        "El problema parece grave, "
        "pero no estoy triste."
    )

    df = pd.DataFrame(
        {
            "text_norm": [
                text,
            ],
        }
    )

    computation = dimension.compute_result(
        df
    )

    inspection = dimension.inspect(
        df.iloc[0]
    )

    assert computation.values.iloc[0] == 2
    assert computation.numerators.iloc[0] == 2

    assert_inspection_matches_evidence(
        computation=computation,
        inspection=inspection,
    )

    expected = [
        (
            "problema",
            text.index("problema"),
            (
                text.index("problema")
                + len("problema")
            ),
        ),
        (
            "triste",
            text.index("triste"),
            (
                text.index("triste")
                + len("triste")
            ),
        ),
    ]

    assert normalize_evidence(
        computation.evidence.iloc[0]
    ) == expected


def test_word_per_dictionary_non_regex_exception_path_is_equivalent():
    """
    Check equivalence without imposing support for multiword
    exceptions in non-regex mode.

    In the current implementation, "menos mal" is stored as one
    non-regex exception entry and does not match an individual token.
    Both occurrences of "mal" are therefore accepted.
    """
    loader = FakeDictionaryLoader(
        words=[
            "mal",
        ],
        exceptions=[
            "menos mal",
        ],
    )

    dimension = WordPerDictionary(
        key="test-dictionary",
        dictionary_name="test",
        percentage=False,
        use_regex=False,
        dictionary_loader=loader,
    )

    text = (
        "Esto está mal, pero menos mal "
        "que terminó."
    )

    df = pd.DataFrame(
        {
            "text_norm": [
                text,
            ],
        }
    )

    computation = dimension.compute_result(
        df
    )

    inspection = dimension.inspect(
        df.iloc[0]
    )

    assert_inspection_matches_evidence(
        computation=computation,
        inspection=inspection,
    )

    first_start = text.index(
        "mal"
    )

    second_start = text.index(
        "mal",
        first_start + 1,
    )

    expected = [
        (
            "mal",
            first_start,
            first_start + len("mal"),
        ),
        (
            "mal",
            second_start,
            second_start + len("mal"),
        ),
    ]

    assert normalize_evidence(
        computation.evidence.iloc[0]
    ) == expected


def test_word_per_dictionary_regex_exception_path_is_equivalent():
    """
    Check that inspection and structured extraction use the same
    accepted matches when regex exceptions are enabled.

    This test deliberately checks equivalence rather than asserting
    a separate interpretation of the dictionary syntax.
    """
    loader = FakeDictionaryLoader(
        words=[
            "mal",
        ],
        exceptions=[
            r"menos\s+mal",
        ],
    )

    dimension = WordPerDictionary(
        key="test-dictionary",
        dictionary_name="test",
        percentage=False,
        use_regex=True,
        dictionary_loader=loader,
    )

    text = (
        "Esto está mal, pero menos mal "
        "que terminó."
    )

    df = pd.DataFrame(
        {
            "text_norm": [
                text,
            ],
        }
    )

    computation = dimension.compute_result(
        df
    )

    inspection = dimension.inspect(
        df.iloc[0]
    )

    assert_inspection_matches_evidence(
        computation=computation,
        inspection=inspection,
    )


def test_word_per_dictionary_empty_result_is_equivalent():
    loader = FakeDictionaryLoader(
        words=[
            "triste",
        ],
    )

    dimension = WordPerDictionary(
        key="test-dictionary",
        dictionary_name="test",
        percentage=False,
        use_regex=False,
        dictionary_loader=loader,
    )

    df = pd.DataFrame(
        {
            "text_norm": [
                "El texto no contiene la entrada.",
            ],
        }
    )

    computation = dimension.compute_result(
        df
    )

    inspection = dimension.inspect(
        df.iloc[0]
    )

    assert computation.values.iloc[0] == 0
    assert computation.numerators.iloc[0] == 0
    assert computation.evidence.iloc[0] == []
    assert inspection.matches == []

    assert_inspection_matches_evidence(
        computation=computation,
        inspection=inspection,
    )

def test_grammatical_gender_inspect_matches_computation_evidence():
    loader = FakeDictionaryLoader(
        words=[
            "amiga",
            "amable",
        ],
    )

    dimension = GrammaticalGenderDimension(
        key="test-grammatical-gender",
        dictionary_name="test",
        percentage=False,
        use_regex=False,
        dictionary_loader=loader,
    )

    text = "La amiga amable llegó"

    df = pd.DataFrame(
        {
            "text_norm": [
                text,
            ],
            "tagged_pos": [
                (
                    "La__(DET)"
                    "(Definite=Def|Gender=Fem|Number=Sing), "
                    "amiga__(NOUN)"
                    "(Gender=Fem|Number=Sing), "
                    "amable__(ADJ)"
                    "(Gender=Fem|Number=Sing), "
                    "llegó__(VERB)"
                    "(Mood=Ind|Tense=Past|VerbForm=Fin)"
                )
            ],
        }
    )

    computation = dimension.compute_result(
        df
    )

    inspection = dimension.inspect(
        df.iloc[0]
    )

    assert computation.values.iloc[0] == 2
    assert computation.numerators.iloc[0] == 2

    assert_inspection_matches_evidence(
        computation=computation,
        inspection=inspection,
    )

    expected = [
        (
            "amiga",
            text.index("amiga"),
            text.index("amiga") + len("amiga"),
        ),
        (
            "amable",
            text.index("amable"),
            text.index("amable") + len("amable"),
        ),
    ]

    assert normalize_evidence(
        computation.evidence.iloc[0]
    ) == expected

def test_grammatical_gender_percentage_is_consistent():
    loader = FakeDictionaryLoader(
        words=[
            "amiga",
        ],
    )

    dimension = GrammaticalGenderDimension(
        key="test-grammatical-gender",
        dictionary_name="test",
        percentage=True,
        use_regex=False,
        dictionary_loader=loader,
    )

    text = "La amiga amable llegó"

    df = pd.DataFrame(
        {
            "text_norm": [
                text,
            ],
            "tagged_pos": [
                (
                    "La__(DET)"
                    "(Definite=Def|Gender=Fem|Number=Sing), "
                    "amiga__(NOUN)"
                    "(Gender=Fem|Number=Sing), "
                    "amable__(ADJ)"
                    "(Gender=Fem|Number=Sing), "
                    "llegó__(VERB)"
                    "(Mood=Ind|Tense=Past|VerbForm=Fin)"
                )
            ],
        }
    )

    computation = dimension.compute_result(
        df
    )

    inspection = dimension.inspect(
        df.iloc[0]
    )

    assert computation.numerators.iloc[0] == 1

    denominator = computation.denominators.iloc[
        0
    ]

    assert denominator > 0

    assert computation.values.iloc[0] == pytest.approx(
        100.0 / denominator
    )

    assert_inspection_matches_evidence(
        computation=computation,
        inspection=inspection,
    )


def test_grammatical_gender_pos_fallback_is_equivalent():
    loader = FakeDictionaryLoader(
        words=[
            "amiga",
        ],
    )

    dimension = GrammaticalGenderDimension(
        key="test-grammatical-gender",
        dictionary_name="test",
        percentage=False,
        use_regex=False,
        dictionary_loader=loader,
    )

    df = pd.DataFrame(
        {
            "text_norm": [
                "",
            ],
            "tagged_pos": [
                (
                    "amiga__(NOUN)"
                    "(Gender=Fem|Number=Sing), "
                    "amable__(ADJ)"
                    "(Gender=Fem|Number=Sing)"
                )
            ],
        }
    )

    computation = dimension.compute_result(
        df
    )

    inspection = dimension.inspect(
        df.iloc[0]
    )

    assert_inspection_matches_evidence(
        computation=computation,
        inspection=inspection,
    )