import pandas as pd

from umutextstats.io import read_input


def test_read_csv_creates_text_columns():
    df = read_input("tests/fixtures/sample.csv", text_column="tweet")

    assert "text_raw" in df.columns
    assert "text" in df.columns

    assert df.loc[0, "text_raw"] == "Hola mundo"
    assert df.loc[0, "text"] == "Hola mundo"


def test_read_csv_preserves_original_columns():
    df = read_input("tests/fixtures/sample.csv", text_column="tweet")

    assert "tweet" in df.columns
    assert "label" in df.columns


def test_read_csv_converts_empty_text_to_empty_string(): 
    df = read_input("tests/fixtures/sample.csv", text_column="tweet")

    assert df.loc[2, "text_raw"] == ""
    assert df.loc[2, "text"] == ""


def test_read_csv_uses_existing_id_column():
    df = read_input("tests/fixtures/sample.csv", text_column="tweet")

    assert list(df["id"]) == [1, 2, 3]


def test_read_csv_raises_error_if_text_column_missing():
    try:
        read_input("tests/fixtures/sample.csv", text_column="missing")
        assert False
    except ValueError as exc:
        assert "Text column 'missing' not found" in str(exc)