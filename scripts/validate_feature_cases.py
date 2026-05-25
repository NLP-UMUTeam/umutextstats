# scripts/validate_feature_cases.py

from pathlib import Path

import yaml

from rich.console import Console
from rich.table import Table

from umutextstats.config.inspect import inspect_dimension_text
from umutextstats.config.loader import load_config


def iter_dimensions(dimensions):
    for dimension in dimensions:
        validation = getattr(dimension, "validation", None)

        if validation:
            cases_path = validation.get("cases")

            if cases_path:
                yield dimension, Path(cases_path)

        yield from iter_dimensions(dimension.children)


console = Console()


def print_stats_table(stats):
    if not stats:
        return

    table = Table(title="Feature Validation Summary")

    table.add_column("Dimension", style="cyan")
    table.add_column("Cases", justify="right")
    table.add_column("Passed", justify="right", style="green")
    table.add_column("Failed", justify="right", style="red")
    table.add_column("File", style="dim")

    for row in stats:
        table.add_row(
            row["dimension"],
            str(row["cases"]),
            str(row["passed"]),
            str(row["failed"]),
            row["file"],
        )

    console.print()
    console.print(table)


def main():
    config = load_config()
    root = Path(__file__).parent.parent

    total_cases = 0
    failures = []
    stats = []

    for dimension, cases_path in iter_dimensions(config.dimensions):
        full_cases_path = root / cases_path

        passed = 0
        failed = 0

        if not full_cases_path.exists():
            failed += 1
            failures.append(
                f"{dimension.key}: cases file does not exist: {full_cases_path}"
            )
            stats.append(
                {
                    "dimension": dimension.key,
                    "file": str(cases_path),
                    "cases": 0,
                    "passed": passed,
                    "failed": failed,
                }
            )
            continue

        data = yaml.safe_load(
            full_cases_path.read_text(encoding="utf-8")
        ) or {}

        cases = data.get("cases", [])

        if not cases:
            failed += 1
            failures.append(
                f"{dimension.key}: no cases found in {full_cases_path}"
            )
            stats.append(
                {
                    "dimension": dimension.key,
                    "file": str(cases_path),
                    "cases": 0,
                    "passed": passed,
                    "failed": failed,
                }
            )
            continue

        for i, case in enumerate(cases):
            total_cases += 1
            case_failed = False

            annotations = case.get("annotations") or {}

            inspection_text = case["text"]

            if annotations.get("tagged_pos"):
                inspection_text = annotations["tagged_pos"]

            inspection = inspect_dimension_text(
                config=config,
                key=dimension.key,
                text=inspection_text,
            )

            matches = len(inspection.matches)

            if "expected_min" in case and case["expected_min"] <= 0:
                failures.append(
                    f"{dimension.key}[{i}]: expected_min must be > 0. "
                    f"Use expected: 0 for negatives."
                )
                case_failed = True

            if "expected" in case and matches != case["expected"]:
                failures.append(
                    f"{dimension.key}[{i}]: expected {case['expected']} "
                    f"matches, got {matches}. Text: {case['text']!r}"
                )
                case_failed = True

            if "expected_min" in case and matches < case["expected_min"]:
                failures.append(
                    f"{dimension.key}[{i}]: expected at least "
                    f"{case['expected_min']} matches, got {matches}. "
                    f"Text: {case['text']!r}"
                )
                case_failed = True

            if case_failed:
                failed += 1
            else:
                passed += 1

        stats.append(
            {
                "dimension": dimension.key,
                "file": str(cases_path),
                "cases": len(cases),
                "passed": passed,
                "failed": failed,
            }
        )

    print(f"Validated feature cases: {total_cases}")
    print_stats_table(stats)

    if failures:
        print("")
        print("Failures:")
        for failure in failures:
            print(f"- {failure}")

        raise SystemExit(1)

    print("")
    print("All feature cases passed.")


if __name__ == "__main__":
    main()