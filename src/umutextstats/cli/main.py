# cli/main.py

from __future__ import annotations

import argparse

from umutextstats.cli.analyze import add_analyze_arguments, run_analyze
from umutextstats.cli.summarize import add_summarize_arguments, run_summarize
from umutextstats.cli.aggregate import add_aggregate_arguments, run_aggregate
from umutextstats.cli.cache import add_cache_arguments, run_cache
from umutextstats.cli.config import add_config_arguments, run_config
from umutextstats.cli.explain import add_explain_arguments, run_explain
from umutextstats.cli.inspect import add_inspect_arguments, run_inspect


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="umutextstats",
        description="UMUTextStats linguistic feature extraction tool",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    analyze_parser = subparsers.add_parser(
        "analyze",
        help="Extract linguistic features",
    )
    add_analyze_arguments(analyze_parser)
    analyze_parser.set_defaults(func=run_analyze)

    summarize_parser = subparsers.add_parser(
        "summarize",
        help="Compute summary statistics",
    )
    add_summarize_arguments(summarize_parser)
    summarize_parser.set_defaults(func=run_summarize)
    
    aggregate_parser = subparsers.add_parser(
        "aggregate",
        help="Compute grouped statistics",
    )

    add_aggregate_arguments(aggregate_parser)
    aggregate_parser.set_defaults(func=run_aggregate)


    cache_parser = subparsers.add_parser(
        "cache",
        help="Cache management commands",
    )
    add_cache_arguments(cache_parser)
    cache_parser.set_defaults(func=run_cache)


    config_parser = subparsers.add_parser(
        "config",
        help="Configuration utilities",
    )
    add_config_arguments(config_parser)
    config_parser.set_defaults(func=run_config)


    explain_parser = subparsers.add_parser(
        "explain",
        help="Explain a linguistic dimension",
    )
    add_explain_arguments(explain_parser)
    explain_parser.set_defaults(func=run_explain)

    inspect_parser = subparsers.add_parser(
        "inspect",
        help="Inspect matches for a single dimension",
    )
    add_inspect_arguments(inspect_parser)
    inspect_parser.set_defaults(func=run_inspect)

    return parser



def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()