# cli/main.py

from __future__ import annotations

import argparse

from umutextstats.cli.aggregate import COMMAND as AGGREGATE_COMMAND
from umutextstats.cli.analyze import COMMAND as ANALYZE_COMMAND
from umutextstats.cli.cache import COMMAND as CACHE_COMMAND
from umutextstats.cli.config import COMMAND as CONFIG_COMMAND
from umutextstats.cli.explain import COMMAND as EXPLAIN_COMMAND
from umutextstats.cli.inspect import COMMAND as INSPECT_COMMAND
from umutextstats.cli.summarize import COMMAND as SUMMARIZE_COMMAND


COMMANDS = [
    ANALYZE_COMMAND,
    SUMMARIZE_COMMAND,
    AGGREGATE_COMMAND,
    CACHE_COMMAND,
    CONFIG_COMMAND,
    EXPLAIN_COMMAND,
    INSPECT_COMMAND,
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="umutextstats",
        description="UMUTextStats linguistic feature extraction tool",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    for command in COMMANDS:
        command_parser = subparsers.add_parser(
            command.name,
            help=command.help,
        )
        command.add_arguments(command_parser)
        command_parser.set_defaults(func=command.run)

    return parser



def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()