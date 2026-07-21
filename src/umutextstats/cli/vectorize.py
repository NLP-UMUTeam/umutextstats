# src/umutextstats/cli/vectorize.py

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


def run_vectorize(
    args: argparse.Namespace,
) -> None:
    if args.input == "-":
        features = read_structured_jsonl_stream(
            sys.stdin,
        )
    else:
        features = read_structured_jsonl(
            args.input,
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