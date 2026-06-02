import pandas as pd

from umutextstats.dimensions.document_structure_paragraphs import (
    ParagraphCountDimension,
    AverageParagraphLengthDimension,
    ParagraphLengthDeviationDimension
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