from __future__ import annotations

import argparse
from pathlib import Path

from rich.console import Console
from umutextstats.config import load_config
from umutextstats.config.inspect import (
    inspect_dimension_text,
    render_inspection,
)


def add_inspect_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "key",
        help="Dimension key to inspect.",
    )

    parser.add_argument(
        "input",
        help="Input text file or '-' for stdin.",
    )

    parser.add_argument(
        "-c",
        "--config",
        default=None,
        help="Configuration file. If omitted, package default config is used.",
    )


def run_inspect(args: argparse.Namespace) -> None:
    config = load_config(args.config)

    if args.input == "-":
        import sys

        text = sys.stdin.read()
    else:
        text = Path(args.input).read_text(encoding="utf-8")

    inspection = inspect_dimension_text(
        config=config,
        key=args.key,
        text=text,
    )

    console = Console()
    console.print(render_inspection(inspection, text))
