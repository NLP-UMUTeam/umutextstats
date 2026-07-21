# src/umutextstats/cli/extract.py

from __future__ import annotations

import argparse
import sys

from umutextstats.cache import CacheManager
from umutextstats.cli.analyze import (
    filter_config,
    resolve_cache_policy,
)
from umutextstats.cli.command import CommandSpec
from umutextstats.common import add_common_features_cached
from umutextstats.config import load_config
from umutextstats.extraction import ExtractionEngine
from umutextstats.io import read_input
from umutextstats.nlp import annotate_dataframe_with_stanza
from umutextstats.output import (
    write_output,
    write_structured_jsonl_stream,
)
from umutextstats.preprocessing.pipeline import (
    preprocess_dataframe_cached,
)
from umutextstats.utils.profiler import Profiler


def add_extract_arguments(
    parser: argparse.ArgumentParser,
) -> None:
    parser.add_argument(
        "input",
        help="Input CSV file or '-' for stdin",
    )

    parser.add_argument(
        "-t",
        "--text-column",
        default="text",
        help=(
            "Name of the column containing the input text "
            "(default: text)"
        ),
    )

    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help=(
            "Output JSONL file. "
            "If omitted, writes structured JSONL to stdout."
        ),
    )

    parser.add_argument(
        "--head",
        type=int,
        default=None,
        help="Limit input rows",
    )

    parser.add_argument(
        "-c",
        "--config",
        default=None,
        help=(
            "YAML or legacy XML configuration file. "
            "If omitted, the package default is used."
        ),
    )

    parser.add_argument(
        "--only",
        default=None,
        help=(
            "Evaluate only selected dimension subtree(s). "
            "Use exact keys separated by '|', "
            "for example 'phonetics|morphosyntax'."
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
        help=(
            "Optional path to save profiling stats. "
            "Supported: .csv, .json"
        ),
    )

    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress bars",
    )


def run_extract(
    args: argparse.Namespace,
) -> None:
    is_stdin = args.input == "-"

    read_cache, write_cache = resolve_cache_policy(
        use_cache=not args.no_cache,
        is_stdin=is_stdin,
        head=args.head,
        only=args.only,
    )

    # Avoid mixing progress output with JSONL written to stdout.
    show_progress = (
        not args.no_progress
        and args.output is not None
    )

    cache = CacheManager(args.cache_dir)
    profiler = Profiler(
        enabled=args.stats is not None
    )

    with profiler.track("config", "load_config"):
        config = load_config(args.config)
        config = filter_config(
            config,
            args.only,
        )

    with profiler.track("io", "read_input"):
        df = read_input(
            args.input,
            text_column=args.text_column,
        )

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
                head=args.head,
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

    engine = ExtractionEngine(
        config=config,
        input_column="text_norm",
        include_unimplemented=True,
        profiler=profiler,
        show_progress=show_progress,
    )

    with profiler.track("dimensions", "compute"):
        extraction_result = engine.compute(df)

    with profiler.track(
        "output",
        "write_structured_jsonl",
    ):
        if args.output:
            write_output(
                extraction_result,
                args.output,
                output_format="structured-jsonl",
            )
        else:
            write_structured_jsonl_stream(
                extraction_result,
                sys.stdout,
            )

    if args.stats:
        write_output(
            profiler.dataframe(),
            args.stats,
        )


COMMAND = CommandSpec(
    name="extract",
    help="Extract structured linguistic features as JSONL",
    add_arguments=add_extract_arguments,
    run=run_extract,
)