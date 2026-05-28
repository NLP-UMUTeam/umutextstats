# src/umutextstats/cli/explain.py

from __future__ import annotations

import argparse
import json

from umutextstats.cli.command import CommandSpec
from umutextstats.config import load_config
from umutextstats.config.explain import (
    explanation_to_dict,
    find_dimension,
    render_dimension_explanation,
)


def add_explain_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "key",
        help="Dimension key to explain.",
    )

    parser.add_argument(
        "-c",
        "--config",
        default=None,
        help="Configuration file. If omitted, package default config is used.",
    )

    parser.add_argument(
        "--json",
        action="store_true",
        help="Print explanation as JSON.",
    )

def run_explain(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    explanation = find_dimension(config, args.key)

    if explanation is None:
        raise SystemExit(f"Dimension not found: {args.key}")

    if args.json:
        print(
            json.dumps(
                explanation_to_dict(explanation),
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    print(render_dimension_explanation(explanation))


COMMAND = CommandSpec(
    name="explain",
    help="Explain a dimension",
    add_arguments=add_explain_arguments,
    run=run_explain,
)