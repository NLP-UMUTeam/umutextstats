# tests/test_output_writer.py

import json

import pandas as pd
import pytest

from umutextstats.output import write_output


def test_write_csv_output(tmp_path):
    df = pd.DataFrame({"id": [1], "length": [10]})
    path = tmp_path / "features.csv"

    write_output(df, path)

    assert path.exists()
    assert path.read_text().startswith("id,length")


def test_write_json_output(tmp_path):
    df = pd.DataFrame({"id": [1], "length": [10]})
    path = tmp_path / "features.json"

    write_output(df, path)

    data = json.loads(path.read_text())

    assert data == [{"id": 1, "length": 10}]


def test_write_forced_json_output(tmp_path):
    df = pd.DataFrame({"id": [1], "length": [10]})
    path = tmp_path / "features.out"

    write_output(df, path, output_format="json")

    data = json.loads(path.read_text())

    assert data == [{"id": 1, "length": 10}]


def test_unknown_output_format_raises_error(tmp_path):
    df = pd.DataFrame({"id": [1]})
    path = tmp_path / "features.unknown"

    with pytest.raises(ValueError):
        write_output(df, path)