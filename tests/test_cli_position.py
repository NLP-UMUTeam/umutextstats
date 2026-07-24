from __future__ import annotations

import json
from dataclasses import dataclass

import pytest

from umutextstats.cli.main import (
    build_parser,
)
from umutextstats.cli.position import (
    run_position,
)


@dataclass(frozen=True)
class FakeDescriptor:
    kind: str
    source: str
    unit: str


@dataclass(frozen=True)
class FakeSegment:
    index: int
    start: float
    end: float
    count: int


@dataclass(frozen=True)
class FakeDistribution:
    segments: list[FakeSegment]
    total_count: int


@dataclass(frozen=True)
class FakeAggregationResult:
    dimension_key: str
    descriptor: FakeDescriptor
    segments: int
    document_count: int
    distribution: FakeDistribution


def make_fake_result() -> FakeAggregationResult:
    return FakeAggregationResult(
        dimension_key="emotion-positive",
        descriptor=FakeDescriptor(
            kind="text_span",
            source="text_norm",
            unit="characters",
        ),
        segments=3,
        document_count=2,
        distribution=FakeDistribution(
            segments=[
                FakeSegment(
                    index=0,
                    start=0.0,
                    end=1.0 / 3.0,
                    count=2,
                ),
                FakeSegment(
                    index=1,
                    start=1.0 / 3.0,
                    end=2.0 / 3.0,
                    count=1,
                ),
                FakeSegment(
                    index=2,
                    start=2.0 / 3.0,
                    end=1.0,
                    count=3,
                ),
            ],
            total_count=6,
        ),
    )


def test_position_command_is_registered():
    parser = build_parser()

    args = parser.parse_args(
        [
            "position",
            "results.jsonl",
            "emotion-positive",
        ]
    )

    assert args.command == "position"
    assert args.input == "results.jsonl"
    assert args.dimension == "emotion-positive"
    assert args.segments == 3
    assert args.output is None
    assert args.func is run_position


def test_position_command_accepts_segments_and_output():
    parser = build_parser()

    args = parser.parse_args(
        [
            "position",
            "results.jsonl",
            "emotion-positive",
            "--segments",
            "10",
            "--output",
            "position.json",
        ]
    )

    assert args.input == "results.jsonl"
    assert args.dimension == "emotion-positive"
    assert args.segments == 10
    assert args.output == "position.json"


def test_position_prints_aggregated_result(
    monkeypatch,
    capsys,
):
    calls = []

    def fake_aggregate(
        path,
        dimension_key,
        *,
        segments,
    ):
        calls.append(
            {
                "path": path,
                "dimension_key": dimension_key,
                "segments": segments,
            }
        )

        return make_fake_result()

    monkeypatch.setattr(
        "umutextstats.cli.position."
        "aggregate_dimension_from_jsonl",
        fake_aggregate,
    )

    parser = build_parser()

    args = parser.parse_args(
        [
            "position",
            "results.jsonl",
            "emotion-positive",
            "--segments",
            "3",
        ]
    )

    args.func(args)

    captured = capsys.readouterr()

    payload = json.loads(
        captured.out
    )

    assert calls == [
        {
            "path": "results.jsonl",
            "dimension_key": "emotion-positive",
            "segments": 3,
        }
    ]

    assert payload["dimension_key"] == (
        "emotion-positive"
    )

    assert payload["descriptor"] == {
        "kind": "text_span",
        "source": "text_norm",
        "unit": "characters",
    }

    assert payload["segments"] == 3
    assert payload["document_count"] == 2

    assert payload["distribution"]["total_count"] == 6

    assert [
        segment["count"]
        for segment in payload["distribution"]["segments"]
    ] == [
        2,
        1,
        3,
    ]


def test_position_writes_json_file(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(
        "umutextstats.cli.position."
        "aggregate_dimension_from_jsonl",
        lambda path, dimension_key, *, segments: (
            make_fake_result()
        ),
    )

    output_path = (
        tmp_path
        / "position.json"
    )

    parser = build_parser()

    args = parser.parse_args(
        [
            "position",
            "results.jsonl",
            "emotion-positive",
            "--segments",
            "3",
            "--output",
            str(output_path),
        ]
    )

    args.func(args)

    assert output_path.exists()

    payload = json.loads(
        output_path.read_text(
            encoding="utf-8",
        )
    )

    assert payload["dimension_key"] == (
        "emotion-positive"
    )

    assert payload["document_count"] == 2


@pytest.mark.parametrize(
    "segments",
    [
        "1",
        "3",
        "10",
    ],
)
def test_position_passes_segment_count_to_service(
    monkeypatch,
    segments,
):
    received_segments = []

    def fake_aggregate(
        path,
        dimension_key,
        *,
        segments,
    ):
        received_segments.append(
            segments
        )

        return make_fake_result()

    monkeypatch.setattr(
        "umutextstats.cli.position."
        "aggregate_dimension_from_jsonl",
        fake_aggregate,
    )

    parser = build_parser()

    args = parser.parse_args(
        [
            "position",
            "results.jsonl",
            "emotion-positive",
            "--segments",
            segments,
        ]
    )

    args.func(args)

    assert received_segments == [
        int(segments),
    ]