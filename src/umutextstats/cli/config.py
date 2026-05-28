from __future__ import annotations

import argparse

from umutextstats.cli.command import CommandSpec
from umutextstats.config.convert import convert_config
from umutextstats.config.loader import load_config
from umutextstats.config.validator import validate_config
from umutextstats.config.tree import render_config_tree

def add_config_arguments(parser: argparse.ArgumentParser) -> None:
    subparsers = parser.add_subparsers(dest="config_command", required=True)

    convert_parser = subparsers.add_parser(
        "convert",
        help="Convert configuration files between supported formats",
    )
    convert_parser.add_argument("input_path")
    convert_parser.add_argument("output_path")
    convert_parser.set_defaults(config_func=run_config_convert)
    
    validate_parser = subparsers.add_parser(
        "validate",
        help="Validate a configuration file",
    )
    validate_parser.add_argument("config_path")
    validate_parser.set_defaults(config_func=run_config_validate)
    
    
    tree_parser = subparsers.add_parser(
        "tree",
        help="Print the configuration dimension tree",
    )
    tree_parser.add_argument(
        "config_path",
        nargs="?",
        default=None,
        help="Configuration file. If omitted, package default config is used.",
    )
    tree_parser.add_argument(
        "--max-depth",
        type=int,
        default=None,
        help="Maximum tree depth to print.",
    )
    tree_parser.set_defaults(config_func=run_config_tree)


def run_config(args: argparse.Namespace) -> None:
    args.config_func(args)


def run_config_convert(args: argparse.Namespace) -> None:
    convert_config(args.input_path, args.output_path)
    print(f"Config converted: {args.input_path} -> {args.output_path}")
    
    
def run_config_validate(args: argparse.Namespace) -> None:
    config = load_config(args.config_path)
    issues = validate_config(config)

    if not issues:
        print(f"Config is valid: {args.config_path}")
        return

    for issue in issues:
        location = f" {issue.key}:" if issue.key else ""
        print(f"[{issue.level}]{location} {issue.message}")

    if any(issue.level == "error" for issue in issues):
        raise SystemExit(1)


def run_config_tree(args: argparse.Namespace) -> None:
    config = load_config(args.config_path)
    print(render_config_tree(config, max_depth=args.max_depth))


COMMAND = CommandSpec(
    name="config",
    help="Configuration utilities",
    add_arguments=add_config_arguments,
    run=run_config,
)