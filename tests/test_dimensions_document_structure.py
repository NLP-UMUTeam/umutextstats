import pandas as pd

import pytest

from statistics import pstdev

from umutextstats.dimensions.document_structure_paragraphs import (
    ParagraphCountDimension,
    AverageParagraphLengthDimension,
    ParagraphLengthDeviationDimension,
    DialogueParagraphPercentageDimension,
    paragraph_lengths,
    analyze_paragraphs
)


def compute_count(texts):
    df = pd.DataFrame({"text_raw": texts})
    dim = ParagraphCountDimension(
        key="paragraph_count",
        input_column="text_raw",
    )
    return list(dim.compute(df))


def compute_avg(texts):
    df = pd.DataFrame({"text_raw": texts})
    dim = AverageParagraphLengthDimension(
        key="paragraph_length_avg",
        input_column="text_raw",
    )
    return list(dim.compute(df))


def compute_std(texts):
    df = pd.DataFrame({"text_raw": texts})
    dim = ParagraphLengthDeviationDimension(
        key="paragraph_length_std",
        input_column="text_raw",
    )
    return list(dim.compute(df))


def test_paragraph_count_empty_text():
    assert compute_count([""]) == [0]


def test_paragraph_count_single_paragraph():
    assert compute_count(["Esto es un párrafo."]) == [1]


def test_paragraph_count_single_paragraph_with_line_break():
    assert compute_count(["Esto es un párrafo.\nSigue siendo el mismo párrafo."]) == [1]


def test_paragraph_count_two_paragraphs_separated_by_blank_line():
    assert compute_count(["Primer párrafo.\n\nSegundo párrafo."]) == [2]


def test_paragraph_count_multiple_blank_lines_count_as_single_separator():
    assert compute_count(["Primer párrafo.\n\n\nSegundo párrafo."]) == [2]


def test_paragraph_count_blank_line_with_spaces_counts_as_separator():
    assert compute_count(["Primer párrafo.\n   \nSegundo párrafo."]) == [2]


def test_paragraph_count_leading_and_trailing_blank_lines_are_ignored():
    assert compute_count(["\n\nPrimer párrafo.\n\nSegundo párrafo.\n\n"]) == [2]


def test_paragraph_count_whitespace_only_text():
    assert compute_count(["   \n \n   "]) == [0]


def test_paragraph_count_multiple_rows():
    assert compute_count(["Un párrafo.", "Uno.\n\nDos.", ""]) == [1, 2, 0]


def test_average_paragraph_length_empty_text():
    assert compute_avg([""]) == [0.0]


def test_average_paragraph_length_single_paragraph():
    assert compute_avg(["Uno dos tres."]) == [3.0]


def test_average_paragraph_length_two_equal_paragraphs():
    assert compute_avg(["Uno dos.\n\nTres cuatro."]) == [2.0]


def test_average_paragraph_length_two_different_paragraphs():
    assert compute_avg(["Uno dos.\n\nTres cuatro cinco seis."]) == [3.0]


def test_average_paragraph_length_single_line_break_is_same_paragraph():
    assert compute_avg(["Uno dos.\nTres cuatro."]) == [4.0]


def test_average_paragraph_length_ignores_empty_paragraphs():
    assert compute_avg(["Uno dos.\n\n\nTres cuatro cinco."]) == [2.5]


def test_average_paragraph_length_multiple_rows():
    assert compute_avg(["Uno dos.", "Uno.\n\nDos tres.", ""]) == [2.0, 1.5, 0.0]


def test_paragraph_length_std_empty_text():
    assert compute_std([""]) == [0.0]


def test_paragraph_length_std_single_paragraph():
    assert compute_std(["Uno dos tres."]) == [0.0]


def test_paragraph_length_std_equal_paragraphs():
    assert compute_std(["Uno dos.\n\nTres cuatro."]) == [0.0]


def test_paragraph_length_std_different_paragraphs():
    assert compute_std(["Uno dos.\n\nTres cuatro cinco seis."]) == [1.0]


def test_paragraph_length_std_multiple_paragraphs():
    assert compute_std(["Uno.\n\nDos tres.\n\nCuatro cinco seis."]) == [
        0.816496580927726
    ]


def test_paragraph_length_std_multiple_rows():
    assert compute_std(["Uno dos.", "Uno.\n\nDos tres.", ""]) == [0.0, 0.5, 0.0]


def test_paragraph_count_uses_text_raw():
    df = pd.DataFrame(
        {
            "text_raw": ["Uno.\n\nDos."],
            "text_norm": ["Uno. Dos."],
        }
    )

    dim = ParagraphCountDimension(
        key="paragraph_count",
        input_column="text_raw",
    )

    assert list(dim.compute(df)) == [2]

def test_paragraph_count_result_contains_evidence():
    text = (
        "Primer párrafo."
        "\n\n"
        "Segundo párrafo con más palabras."
    )

    df = pd.DataFrame(
        {
            "text_norm": [text],
        }
    )

    dimension = ParagraphCountDimension(
        key="paragraph_count",
    )

    result = dimension.compute_result(df)

    assert result.values.tolist() == [2]
    assert result.numerators.tolist() == [2]
    assert result.denominators is None
    assert len(result.evidence.iloc[0]) == 2

    for evidence in result.evidence.iloc[0]:
        assert (
            text[
                evidence["start"]:
                evidence["end"]
            ]
            == evidence["text"]
        )

def test_average_paragraph_length_result():
    text = "Uno dos.\n\nTres cuatro cinco."

    df = pd.DataFrame(
        {
            "text_norm": [text],
        }
    )

    dimension = AverageParagraphLengthDimension(
        key="paragraph_length_avg",
    )

    result = dimension.compute_result(df)

    assert result.numerators.tolist() == [5]
    assert result.denominators.tolist() == [2]
    assert result.values.tolist() == [2.5]

    assert [
        item["word_count"]
        for item in result.evidence.iloc[0]
    ] == [2, 3]

def test_paragraph_length_deviation_result():
    text = "Uno dos.\n\nTres cuatro cinco cuatro."

    df = pd.DataFrame(
        {
            "text_norm": [text],
        }
    )

    dimension = ParagraphLengthDeviationDimension(
        key="paragraph_length_std",
    )

    result = dimension.compute_result(df)

    expected = pstdev([2, 4])

    assert result.values.tolist() == [expected]
    assert result.numerators is None
    assert result.denominators is None

    assert [
        item["word_count"]
        for item in result.evidence.iloc[0]
    ] == [2, 4]

def test_dialogue_paragraph_result_contains_evidence():
    text = (
        "Párrafo narrativo."
        "\n\n"
        "—Hola, ¿qué tal?"
    )

    df = pd.DataFrame(
        {
            "text_norm": [text],
        }
    )

    dimension = DialogueParagraphPercentageDimension(
        key="dialogue_paragraphs",
    )

    result = dimension.compute_result(df)

    assert result.values.tolist() == [50.0]
    assert result.numerators.tolist() == [1]
    assert result.denominators.tolist() == [2]

    evidence = result.evidence.iloc[0]

    assert len(evidence) == 1
    assert evidence[0]["text"] == "—Hola, ¿qué tal?"

@pytest.mark.parametrize(
    "dimension",
    [
        ParagraphCountDimension(key="paragraph_count"),
        AverageParagraphLengthDimension(
            key="paragraph_length_avg"
        ),
        ParagraphLengthDeviationDimension(
            key="paragraph_length_std"
        ),
        DialogueParagraphPercentageDimension(
            key="dialogue_paragraphs"
        ),
    ],
)
def test_paragraph_compute_matches_compute_result(
    dimension,
):
    df = pd.DataFrame(
        {
            "text_norm": [
                "Uno dos.\n\n—Tres cuatro cinco.",
                "",
            ]
        }
    )

    expected = dimension.compute(df)
    actual = dimension.compute_result(df).values

    pd.testing.assert_series_equal(
        actual,
        expected,
        check_dtype=False,
        check_names=False,
    )

def test_paragraph_spans_and_lengths_are_aligned():
    text = (
        "Primer párrafo.\n\n"
        "Segundo párrafo con más palabras.\n\n\n"
        "Tercero."
    )

    evidence = analyze_paragraphs(text)

    assert len(evidence) == 3

    for paragraph in evidence:
        assert (
            text[paragraph.start:paragraph.end]
            == paragraph.text
        )

    assert [
        paragraph.word_count
        for paragraph in evidence
    ] == paragraph_lengths(text)

def test_paragraph_count_uses_precomputed_value():
    df = pd.DataFrame(
        {
            "text_norm": ["Uno.\n\nDos."],
            "paragraph_count": [7],
        }
    )

    dimension = ParagraphCountDimension(
        key="paragraph_count"
    )

    result = dimension.compute_result(df)

    assert result.values.tolist() == [7]
    assert result.numerators.tolist() == [2]
    assert len(result.evidence.iloc[0]) == 2

@pytest.mark.parametrize(
    "text",
    [
        "",
        "   ",
        "\n\n\n",
    ],
)
def test_empty_paragraph_inputs(text):
    df = pd.DataFrame(
        {
            "text_norm": [text],
        }
    )

    count = ParagraphCountDimension(
        key="paragraph_count"
    ).compute_result(df)

    average = AverageParagraphLengthDimension(
        key="paragraph_length_avg"
    ).compute_result(df)

    deviation = ParagraphLengthDeviationDimension(
        key="paragraph_length_std"
    ).compute_result(df)

    dialogue = DialogueParagraphPercentageDimension(
        key="dialogue_paragraphs"
    ).compute_result(df)

    assert count.values.tolist() == [0]
    assert count.evidence.iloc[0] == []

    assert average.values.tolist() == [0.0]
    assert average.numerators.tolist() == [0]
    assert average.denominators.tolist() == [0]

    assert deviation.values.tolist() == [0.0]
    assert deviation.evidence.iloc[0] == []

    assert dialogue.values.tolist() == [0.0]
    assert dialogue.numerators.tolist() == [0]
    assert dialogue.denominators.tolist() == [0]