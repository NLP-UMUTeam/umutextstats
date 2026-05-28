# src/umutextstats/cli/cache.py

from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path

from umutextstats.cli.command import CommandSpec
from umutextstats.cache import CacheManager


def add_cache_arguments(parser: argparse.ArgumentParser) -> None:
    subparsers = parser.add_subparsers(
        dest="cache_command",
        required=True,
    )

    clear_parser = subparsers.add_parser("clear", help="Delete all cache files")
    clear_parser.add_argument("--cache-dir", default=None)
    clear_parser.add_argument("-y", "--yes", action="store_true")

    info_parser = subparsers.add_parser("info", help="Show cache summary")
    info_parser.add_argument("--cache-dir", default=None)

    list_parser = subparsers.add_parser("list", help="List cache files")
    list_parser.add_argument("--cache-dir", default=None)
    list_parser.add_argument("--limit", type=int, default=50)

    prune_parser = subparsers.add_parser("prune", help="Delete old cache files")
    prune_parser.add_argument("--cache-dir", default=None)
    prune_parser.add_argument(
        "--older-than-days",
        type=int,
        required=True,
        help="Delete cache files older than N days",
    )
    prune_parser.add_argument("-y", "--yes", action="store_true")


def run_cache(args: argparse.Namespace) -> None:
    if args.cache_command == "clear":
        run_cache_clear(args)
    elif args.cache_command == "info":
        run_cache_info(args)
    elif args.cache_command == "list":
        run_cache_list(args)
    elif args.cache_command == "prune":
        run_cache_prune(args)
    else:
        raise ValueError(f"Unknown cache command: {args.cache_command}")


def get_cache_dir(args: argparse.Namespace) -> Path:
    cache = CacheManager()
    return Path(args.cache_dir) if args.cache_dir else Path(cache.cache_dir)


def format_size(size: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]

    value = float(size)
    for unit in units:
        if value < 1024:
            return f"{value:.1f} {unit}"
        value /= 1024

    return f"{value:.1f} PB"


def iter_cache_files(cache_dir: Path):
    if not cache_dir.exists():
        return []

    return [path for path in cache_dir.rglob("*") if path.is_file()]


def run_cache_clear(args: argparse.Namespace) -> None:
    cache_dir = get_cache_dir(args)

    if not cache_dir.exists():
        print(f"Cache directory does not exist: {cache_dir}")
        return

    if not args.yes:
        answer = input(f"Delete cache directory '{cache_dir}'? [y/N] ")
        if answer.lower() not in {"y", "yes"}:
            print("Cancelled.")
            return

    shutil.rmtree(cache_dir)
    print(f"Cache deleted: {cache_dir}")


def run_cache_info(args: argparse.Namespace) -> None:
    cache_dir = get_cache_dir(args)
    files = iter_cache_files(cache_dir)

    total_size = sum(path.stat().st_size for path in files)

    print(f"Cache dir: {cache_dir}")
    print(f"Exists: {cache_dir.exists()}")
    print(f"Files: {len(files)}")
    print(f"Size: {format_size(total_size)}")


def run_cache_list(args: argparse.Namespace) -> None:
    cache_dir = get_cache_dir(args)
    files = iter_cache_files(cache_dir)

    if not files:
        print(f"No cache files found in: {cache_dir}")
        return

    files = sorted(
        files,
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )

    for path in files[: args.limit]:
        stat = path.stat()
        rel = path.relative_to(cache_dir)
        modified = time.strftime(
            "%Y-%m-%d %H:%M:%S",
            time.localtime(stat.st_mtime),
        )

        print(f"{modified}  {format_size(stat.st_size):>10}  {rel}")

    if len(files) > args.limit:
        print(f"... {len(files) - args.limit} more files")


def run_cache_prune(args: argparse.Namespace) -> None:
    cache_dir = get_cache_dir(args)
    files = iter_cache_files(cache_dir)

    cutoff = time.time() - args.older_than_days * 24 * 60 * 60

    old_files = [
        path for path in files
        if path.stat().st_mtime < cutoff
    ]

    if not old_files:
        print(
            f"No cache files older than "
            f"{args.older_than_days} days found."
        )
        return

    total_size = sum(path.stat().st_size for path in old_files)

    print(
        f"Found {len(old_files)} files older than "
        f"{args.older_than_days} days "
        f"({format_size(total_size)})."
    )

    if not args.yes:
        answer = input("Delete these files? [y/N] ")
        if answer.lower() not in {"y", "yes"}:
            print("Cancelled.")
            return

    for path in old_files:
        path.unlink()

    remove_empty_dirs(cache_dir)

    print(
        f"Deleted {len(old_files)} files "
        f"({format_size(total_size)})."
    )


def remove_empty_dirs(root: Path) -> None:
    if not root.exists():
        return

    for path in sorted(root.rglob("*"), reverse=True):
        if path.is_dir():
            try:
                path.rmdir()
            except OSError:
                pass

COMMAND = CommandSpec(
    name="cache",
    help="Cache management commands",
    add_arguments=add_cache_arguments,
    run=run_cache,
)