from __future__ import annotations

import argparse
import sys

from umutextstats.cli.command import CommandSpec
from umutextstats.output import write_output
from umutextstats.vectorization import (
    read_structured_jsonl,
    read_structured_jsonl_stream,
)


def add_vectorize_arguments(
    parser: argparse.ArgumentParser,
) -> None:
    parser.add_argument(
        "input",
        help="Structured JSONL file or '-' for stdin",
    )

    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help=(
            "Output feature matrix. Supported: .csv, .json. "
            "If omitted, writes CSV to stdout."
        ),
    )

    parser.add_argument(
        "--output-format",
        default=None,
        choices=[
            "csv",
            "json",
        ],
        help=(
            "Explicit output format. Normally inferred from "
            "the output filename."
        ),
    )

    parser.add_argument(
        "--position-dimension",
        action="append",
        default=[],
        help=(
            "Dimension whose evidence will be converted into "
            "document-level positional features. "
            "May be specified multiple times."
        ),
    )

    parser.add_argument(
        "--position-segments",
        type=int,
        default=5,
        help=(
            "Number of relative segments used for positional "
            "features. Default: 5"
        ),
    )


def run_vectorize(
    args: argparse.Namespace,
) -> None:
    reader_kwargs = {
        "position_dimensions": (
            args.position_dimension
        ),
        "position_segments": (
            args.position_segments
        ),
    }

    if args.input == "-":
        features = read_structured_jsonl_stream(
            sys.stdin,
            **reader_kwargs,
        )
    else:
        features = read_structured_jsonl(
            args.input,
            **reader_kwargs,
        )

    if args.output:
        write_output(
            features,
            path=args.output,
            output_format=args.output_format,
        )
        return

    sys.stdout.write(
        features.to_csv(
            index=False,
        )
    )


COMMAND = CommandSpec(
    name="vectorize",
    help="Convert structured JSONL into a feature matrix",
    add_arguments=add_vectorize_arguments,
    run=run_vectorize,
)