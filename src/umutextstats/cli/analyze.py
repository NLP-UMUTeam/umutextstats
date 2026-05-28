# src/umutextstats/cli/analyze.py

from __future__ import annotations

import argparse

from umutextstats.cli.command import CommandSpec
from umutextstats.cache import CacheManager
from umutextstats.config import load_config
from umutextstats.dimensions import DimensionEngine
from umutextstats.io import read_input
from umutextstats.nlp import annotate_dataframe_with_stanza
from umutextstats.output import write_output
from umutextstats.preprocessing.pipeline import preprocess_dataframe_cached
from umutextstats.common import add_common_features_cached
from umutextstats.utils.profiler import Profiler


def add_analyze_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "input",
        help="Input CSV file or '-' for stdin",
    )

    parser.add_argument(
        "-t",
        "--text-column",
        required=True,
        help="Name of the column containing the input text",
    )

    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output file path. Supported: .csv, .json. If omitted, prints to stdout.",
    )
    
    parser.add_argument(
        "--head",
        type=int,
        default=None,
        help="Limit output rows",
    )
    
    parser.add_argument(
        "-c",
        "--config",
        default=None,
        help="XML configuration file. If omitted, package default.xml is used.",
    )
    
    parser.add_argument(
        "--only",
        default=None,
        help=(
            "Evaluate only selected dimension subtree(s). "
            "Use exact keys separated by '|', e.g. 'phonetics|morphosyntax'."
        ),
    )

    parser.add_argument(
        "--cache-dir",
        default=".cache",
        help="Cache directory",
    )

    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable stage cache",
    )

    parser.add_argument(
        "--no-stanza",
        action="store_true",
        help="Skip Stanza POS/NER annotation",
    )

    parser.add_argument(
        "--stats",
        default=None,
        help="Optional path to save profiling stats. Supported: .csv, .json",
    )

    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress bars",
    )


def parse_only_keys(value: str | None) -> set[str]:
    if not value:
        return set()

    return {
        item.strip()
        for item in value.split("|")
        if item.strip()
    }


def filter_dimension_tree(dimensions, selected_keys: set[str]):
    if not selected_keys:
        return list(dimensions)

    selected = []

    for dimension in dimensions:
        if dimension.key in selected_keys:
            selected.append(dimension)
            continue

        children = filter_dimension_tree(dimension.children, selected_keys)

        if children:
            dimension.children = children
            selected.append(dimension)

    return selected


def filter_config(config, only: str | None):
    selected_keys = parse_only_keys(only)

    if not selected_keys:
        return config

    config.dimensions = filter_dimension_tree(config.dimensions, selected_keys)

    if not config.dimensions:
        valid = ", ".join(sorted(selected_keys))
        raise ValueError(f"No dimensions found for --only: {valid}")

    return config


def resolve_cache_policy(
    use_cache: bool,
    is_stdin: bool,
    head: int | None,
    only: str | None,
):
    use_cache = use_cache and not is_stdin
    read_cache = use_cache
    write_cache = use_cache and head is None and only is None

    return read_cache, write_cache


def run_analyze(args: argparse.Namespace) -> None:
    is_stdin = args.input == "-"
    
    use_cache = (not args.no_cache) and (not is_stdin)
    read_cache, write_cache = resolve_cache_policy (
        use_cache=not args.no_cache,
        is_stdin=args.input == "-",
        head=args.head,
        only=args.only,
    )
    
    show_progress = not args.no_progress and args.output is not None

    cache = CacheManager(args.cache_dir)
    profiler = Profiler(enabled=args.stats is not None)

    with profiler.track("config", "load_config"):
        config = load_config(args.config)
        config = filter_config(config, args.only)

    with profiler.track("io", "read_input"):
        df = read_input(args.input, text_column=args.text_column)

    if args.head is not None:
        df = df.head(args.head)

    with profiler.track("preprocessing", "normalize"):
        df = preprocess_dataframe_cached(
            df,
            input_path=args.input,
            cache=cache,
            use_cache=read_cache,
            write_cache=write_cache,
            show_progress=show_progress,
            head=args.head,
        )


    if not args.no_stanza:
        with profiler.track("nlp", "stanza"):
            df = annotate_dataframe_with_stanza(
                df,
                input_path=args.input,
                cache=cache,
                use_cache=read_cache,
                write_cache=write_cache,
                head=args.head
            )

    with profiler.track("common", "features"):
        df = add_common_features_cached(
            df,
            input_path=args.input,
            cache=cache,
            use_cache=read_cache,
            write_cache=write_cache,
            head=args.head,
        )


    engine = DimensionEngine(
        config=config,
        input_column="text_norm",
        include_unimplemented=True,
        profiler=profiler,
        show_progress=show_progress,
    )

    with profiler.track("dimensions", "compute"):
        features = engine.compute(df)
        

    with profiler.track("output", "write_features"):
        if args.output:
            write_output(features, args.output)
        else:
            print(features.to_csv(index=False))
    
    if args.stats:
        write_output(profiler.dataframe(), args.stats)


COMMAND = CommandSpec(
    name="analyze",
    help="Extract linguistic features",
    add_arguments=add_analyze_arguments,
    run=run_analyze,
)