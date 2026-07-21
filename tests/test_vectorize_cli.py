# tests/test_vectorize_cli.py

from __future__ import annotations

import io

from umutextstats.cli.main import build_parser


def test_vectorize_command_writes_csv_to_stdout(
    monkeypatch,
    capsys,
):
    input_jsonl = "\n".join(
        [
            (
                '{"_type":"metadata","schema_version":"1.0",'
                '"dimensions":["feature-a","feature-b"]}'
            ),
            (
                '{"_type":"document","id":"doc-1",'
                '"dimensions":{'
                '"feature-a":{"value":1},'
                '"feature-b":{"value":0.5}}}'
            ),
        ]
    )

    monkeypatch.setattr(
        "sys.stdin",
        io.StringIO(input_jsonl),
    )

    parser = build_parser()

    args = parser.parse_args(
        [
            "vectorize",
            "-",
        ]
    )

    args.func(args)

    captured = capsys.readouterr()

    assert captured.err == ""
    assert captured.out == (
        "id,feature-a,feature-b\n"
        "doc-1,1,0.5\n"
    )


def test_vectorize_command_writes_csv_file(
    tmp_path,
):
    input_path = tmp_path / "analysis.jsonl"
    output_path = tmp_path / "features.csv"

    input_path.write_text(
        "\n".join(
            [
                (
                    '{"_type":"metadata","schema_version":"1.0",'
                    '"dimensions":["feature-a"]}'
                ),
                (
                    '{"_type":"document","id":"doc-1",'
                    '"dimensions":{"feature-a":{"value":2}}}'
                ),
            ]
        ),
        encoding="utf-8",
    )

    parser = build_parser()

    args = parser.parse_args(
        [
            "vectorize",
            str(input_path),
            "-o",
            str(output_path),
        ]
    )

    args.func(args)

    assert output_path.read_text(
        encoding="utf-8",
    ) == (
        "id,feature-a\n"
        "doc-1,2\n"
    )