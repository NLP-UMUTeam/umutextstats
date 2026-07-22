#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit a structured JSONL file produced by "
            "UMUTextStats extract."
        )
    )

    parser.add_argument(
        "input",
        type=Path,
        help="Structured JSONL file produced by extract.",
    )

    parser.add_argument(
        "--show-dimensions",
        action="store_true",
        help="List all dimensions and their main metadata.",
    )

    parser.add_argument(
        "--show-ratios",
        action="store_true",
        help="List RatioDimension definitions and validation results.",
    )

    parser.add_argument(
        "--show-composites",
        action="store_true",
        help="List composite dimensions and their children.",
    )

    parser.add_argument(
        "--show-problems",
        action="store_true",
        help="Show detailed validation problems.",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=50,
        help="Maximum number of detailed rows to print per section.",
    )

    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-9,
        help="Numeric tolerance used in formula validation.",
    )

    return parser.parse_args()


def read_jsonl(
    path: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    metadata: dict[str, Any] | None = None
    documents: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON at line {line_number}: {exc}"
                ) from exc

            record_type = record.get("_type")

            if record_type == "metadata":
                if metadata is not None:
                    raise ValueError(
                        "The JSONL file contains more than one "
                        "metadata record."
                    )

                metadata = record

            elif record_type == "document":
                documents.append(record)

            else:
                raise ValueError(
                    f"Unknown record type at line {line_number}: "
                    f"{record_type!r}"
                )

    if metadata is None:
        raise ValueError(
            "The JSONL file does not contain a metadata record."
        )

    return metadata, documents


def is_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def numbers_close(
    left: float,
    right: float,
    tolerance: float,
) -> bool:
    return math.isclose(
        float(left),
        float(right),
        rel_tol=tolerance,
        abs_tol=tolerance,
    )


def format_number(value: Any) -> str:
    if not is_number(value):
        return repr(value)

    return f"{float(value):.10g}"


def truncate(
    value: str,
    width: int,
) -> str:
    if len(value) <= width:
        return value

    return value[: width - 1] + "…"


def print_table(
    headers: list[str],
    rows: Iterable[Iterable[Any]],
) -> None:
    rendered_rows = [
        [str(value) for value in row]
        for row in rows
    ]

    if not rendered_rows:
        print("(none)")
        return

    widths = [
        len(header)
        for header in headers
    ]

    for row in rendered_rows:
        for index, value in enumerate(row):
            widths[index] = max(
                widths[index],
                len(value),
            )

    header_line = "  ".join(
        header.ljust(widths[index])
        for index, header in enumerate(headers)
    )

    separator = "  ".join(
        "-" * width
        for width in widths
    )

    print(header_line)
    print(separator)

    for row in rendered_rows:
        print(
            "  ".join(
                value.ljust(widths[index])
                for index, value in enumerate(row)
            )
        )


def collect_metadata_problems(
    metadata: dict[str, Any],
) -> list[str]:
    problems: list[str] = []

    dimensions = metadata.get("dimensions", [])
    dimension_metadata = metadata.get(
        "dimension_metadata",
        {},
    )

    if not isinstance(dimensions, list):
        problems.append(
            "metadata.dimensions is not a list."
        )
        return problems

    if not isinstance(dimension_metadata, dict):
        problems.append(
            "metadata.dimension_metadata is not a mapping."
        )
        return problems

    listed = set(dimensions)
    described = set(dimension_metadata)

    for key in sorted(listed - described):
        problems.append(
            f"Dimension '{key}' has no metadata."
        )

    for key in sorted(described - listed):
        problems.append(
            f"Metadata exists for unlisted dimension '{key}'."
        )

    for key, info in dimension_metadata.items():
        kind = info.get("kind")
        class_name = info.get("class_name")

        if kind == "unimplemented":
            problems.append(
                f"Dimension '{key}' is unimplemented."
            )

        if class_name == "RatioDimension":
            numerator_dimensions = info.get(
                "numerator_dimensions",
                [],
            )
            denominator_dimensions = info.get(
                "denominator_dimensions",
                [],
            )

            if not numerator_dimensions:
                problems.append(
                    f"Ratio '{key}' has no numerator dimensions."
                )

            if not denominator_dimensions:
                problems.append(
                    f"Ratio '{key}' has no denominator dimensions."
                )

            for dependency in (
                list(numerator_dimensions)
                + list(denominator_dimensions)
            ):
                if dependency not in listed:
                    problems.append(
                        f"Ratio '{key}' references missing "
                        f"dimension '{dependency}'."
                    )

        if kind == "composite":
            children = info.get("children", [])
            used_children = info.get(
                "used_children",
                [],
            )
            missing_children = info.get(
                "missing_children",
                [],
            )

            if missing_children:
                problems.append(
                    f"Composite '{key}' has missing children: "
                    f"{', '.join(missing_children)}."
                )

            for child in children:
                if child not in listed:
                    problems.append(
                        f"Composite '{key}' references unlisted "
                        f"child '{child}'."
                    )

            unknown_used = (
                set(used_children)
                - set(children)
            )

            if unknown_used:
                problems.append(
                    f"Composite '{key}' reports unexpected used "
                    f"children: {', '.join(sorted(unknown_used))}."
                )

    return problems


def collect_document_problems(
    metadata: dict[str, Any],
    documents: list[dict[str, Any]],
    tolerance: float,
) -> tuple[list[str], Counter]:
    problems: list[str] = []
    counters: Counter = Counter()

    dimensions = metadata.get("dimensions", [])
    dimension_metadata = metadata.get(
        "dimension_metadata",
        {},
    )

    expected_dimensions = set(dimensions)

    for document_index, document in enumerate(documents):
        document_id = document.get(
            "id",
            document_index,
        )

        values = document.get(
            "dimensions",
            {},
        )

        if not isinstance(values, dict):
            problems.append(
                f"Document {document_id!r} has no valid "
                "'dimensions' mapping."
            )
            continue

        actual_dimensions = set(values)

        for key in sorted(
            expected_dimensions - actual_dimensions
        ):
            counters["missing_dimension_values"] += 1
            problems.append(
                f"Document {document_id!r}: missing dimension '{key}'."
            )

        for key in sorted(
            actual_dimensions - expected_dimensions
        ):
            counters["unexpected_dimension_values"] += 1
            problems.append(
                f"Document {document_id!r}: unexpected dimension "
                f"'{key}'."
            )

        for key, result in values.items():
            if not isinstance(result, dict):
                counters["invalid_dimension_results"] += 1
                problems.append(
                    f"Document {document_id!r}, dimension '{key}': "
                    "result is not an object."
                )
                continue

            info = dimension_metadata.get(
                key,
                {},
            )

            value = result.get("value")
            numerator = result.get("numerator")
            denominator = result.get("denominator")
            evidence = result.get("evidence")

            if "value" not in result:
                counters["missing_values"] += 1
                problems.append(
                    f"Document {document_id!r}, dimension '{key}': "
                    "missing value."
                )

            if evidence is not None and not isinstance(
                evidence,
                list,
            ):
                counters["invalid_evidence"] += 1
                problems.append(
                    f"Document {document_id!r}, dimension '{key}': "
                    "evidence is not a list."
                )

            measure = info.get("measure")
            class_name = info.get("class_name")
            aggregation = info.get("aggregation")

            if measure == "count":
                if numerator is None:
                    counters["counts_without_numerator"] += 1
                    problems.append(
                        f"Document {document_id!r}, dimension '{key}': "
                        "count has no numerator."
                    )

                elif (
                    is_number(value)
                    and is_number(numerator)
                    and not numbers_close(
                        value,
                        numerator,
                        tolerance,
                    )
                ):
                    counters["invalid_count_formulas"] += 1
                    problems.append(
                        f"Document {document_id!r}, dimension '{key}': "
                        f"value={format_number(value)} differs from "
                        f"numerator={format_number(numerator)}."
                    )

            if measure == "rate":
                validate_scaled_ratio(
                    problems=problems,
                    counters=counters,
                    document_id=document_id,
                    key=key,
                    value=value,
                    numerator=numerator,
                    denominator=denominator,
                    scale=info.get("scale", 100.0),
                    zero_division=0.0,
                    tolerance=tolerance,
                    problem_name="rate",
                )

            if class_name == "RatioDimension":
                validate_scaled_ratio(
                    problems=problems,
                    counters=counters,
                    document_id=document_id,
                    key=key,
                    value=value,
                    numerator=numerator,
                    denominator=denominator,
                    scale=info.get("scale", 1.0),
                    zero_division=info.get(
                        "zero_division",
                        0.0,
                    ),
                    tolerance=tolerance,
                    problem_name="ratio",
                )

            if aggregation == "mean":
                validate_scaled_ratio(
                    problems=problems,
                    counters=counters,
                    document_id=document_id,
                    key=key,
                    value=value,
                    numerator=numerator,
                    denominator=denominator,
                    scale=1.0,
                    zero_division=0.0,
                    tolerance=tolerance,
                    problem_name="composite mean",
                )

                used_children = info.get(
                    "used_children",
                    [],
                )

                if (
                    is_number(denominator)
                    and int(denominator)
                    != len(used_children)
                ):
                    counters[
                        "invalid_composite_denominators"
                    ] += 1

                    problems.append(
                        f"Document {document_id!r}, dimension '{key}': "
                        f"mean denominator={format_number(denominator)}, "
                        f"but used_children={len(used_children)}."
                    )

    return problems, counters


def validate_scaled_ratio(
    *,
    problems: list[str],
    counters: Counter,
    document_id: Any,
    key: str,
    value: Any,
    numerator: Any,
    denominator: Any,
    scale: Any,
    zero_division: Any,
    tolerance: float,
    problem_name: str,
) -> None:
    if not (
        is_number(value)
        and is_number(numerator)
        and is_number(denominator)
        and is_number(scale)
        and is_number(zero_division)
    ):
        counters[f"{problem_name}_missing_components"] += 1

        problems.append(
            f"Document {document_id!r}, dimension '{key}': "
            f"{problem_name} lacks numeric value/components."
        )
        return

    if float(denominator) == 0.0:
        expected = float(zero_division)
    else:
        expected = (
            float(numerator)
            / float(denominator)
        ) * float(scale)

    if not numbers_close(
        value,
        expected,
        tolerance,
    ):
        counters[f"invalid_{problem_name}_formulas"] += 1

        problems.append(
            f"Document {document_id!r}, dimension '{key}': "
            f"value={format_number(value)}, expected "
            f"{format_number(expected)} from "
            f"{format_number(numerator)} / "
            f"{format_number(denominator)} × "
            f"{format_number(scale)}."
        )


def print_summary(
    metadata: dict[str, Any],
    documents: list[dict[str, Any]],
    metadata_problems: list[str],
    document_problems: list[str],
) -> None:
    dimension_metadata = metadata.get(
        "dimension_metadata",
        {},
    )

    kinds = Counter(
        info.get("kind", "unknown")
        for info in dimension_metadata.values()
    )

    classes = Counter(
        info.get("class_name", "unknown")
        for info in dimension_metadata.values()
    )

    measures = Counter(
        info.get("measure", "unknown")
        for info in dimension_metadata.values()
    )

    ratio_count = sum(
        1
        for info in dimension_metadata.values()
        if info.get("class_name") == "RatioDimension"
    )

    composites = sum(
        1
        for info in dimension_metadata.values()
        if info.get("kind") == "composite"
    )

    composites_missing = sum(
        1
        for info in dimension_metadata.values()
        if info.get("missing_children")
    )

    print("Extraction audit")
    print("================")
    print(
        f"Schema version:       "
        f"{metadata.get('schema_version', 'unknown')}"
    )
    print(
        f"Documents:            {len(documents)}"
    )
    print(
        f"Dimensions:           {len(dimension_metadata)}"
    )
    print(
        f"Atomic:               {kinds.get('atomic', 0)}"
    )
    print(
        f"Derived:              {kinds.get('derived', 0)}"
    )
    print(
        f"Composite:            {composites}"
    )
    print(
        f"Unimplemented:        "
        f"{kinds.get('unimplemented', 0)}"
    )
    print(
        f"Ratio dimensions:     {ratio_count}"
    )
    print(
        f"Composites incomplete:{composites_missing:>5}"
    )
    print(
        f"Metadata problems:    {len(metadata_problems)}"
    )
    print(
        f"Document problems:    {len(document_problems)}"
    )

    print("\nKinds")
    print_table(
        ["kind", "count"],
        sorted(kinds.items()),
    )

    print("\nMeasures")
    print_table(
        ["measure", "count"],
        sorted(measures.items()),
    )

    print("\nClasses")
    print_table(
        ["class", "count"],
        sorted(
            classes.items(),
            key=lambda item: (
                -item[1],
                item[0],
            ),
        ),
    )


def print_dimensions(
    metadata: dict[str, Any],
) -> None:
    rows = []

    for key in metadata.get("dimensions", []):
        info = metadata.get(
            "dimension_metadata",
            {},
        ).get(key, {})

        rows.append(
            [
                truncate(key, 60),
                info.get("kind", ""),
                info.get("class_name", ""),
                info.get("measure", ""),
                info.get("aggregation", ""),
            ]
        )

    print("\nDimensions")
    print_table(
        [
            "key",
            "kind",
            "class",
            "measure",
            "aggregation",
        ],
        rows,
    )


def print_ratios(
    metadata: dict[str, Any],
) -> None:
    rows = []

    for key, info in metadata.get(
        "dimension_metadata",
        {},
    ).items():
        if info.get("class_name") != "RatioDimension":
            continue

        rows.append(
            [
                truncate(key, 55),
                " + ".join(
                    info.get(
                        "numerator_dimensions",
                        [],
                    )
                ) or "—",
                " + ".join(
                    info.get(
                        "denominator_dimensions",
                        [],
                    )
                ) or "—",
                format_number(
                    info.get("scale")
                ),
                format_number(
                    info.get("zero_division")
                ),
            ]
        )

    print("\nRatios")
    print_table(
        [
            "key",
            "numerator dimensions",
            "denominator dimensions",
            "scale",
            "zero division",
        ],
        rows,
    )


def print_composites(
    metadata: dict[str, Any],
) -> None:
    rows = []

    for key, info in metadata.get(
        "dimension_metadata",
        {},
    ).items():
        if info.get("kind") != "composite":
            continue

        rows.append(
            [
                truncate(key, 55),
                info.get("aggregation", ""),
                str(len(info.get("children", []))),
                str(len(info.get("used_children", []))),
                ", ".join(
                    info.get("missing_children", [])
                ) or "—",
            ]
        )

    print("\nComposites")
    print_table(
        [
            "key",
            "aggregation",
            "children",
            "used",
            "missing",
        ],
        rows,
    )


def main() -> int:
    args = parse_args()

    metadata, documents = read_jsonl(
        args.input
    )

    metadata_problems = (
        collect_metadata_problems(
            metadata
        )
    )

    document_problems, _ = (
        collect_document_problems(
            metadata=metadata,
            documents=documents,
            tolerance=args.tolerance,
        )
    )

    print_summary(
        metadata=metadata,
        documents=documents,
        metadata_problems=metadata_problems,
        document_problems=document_problems,
    )

    if args.show_dimensions:
        print_dimensions(metadata)

    if args.show_ratios:
        print_ratios(metadata)

    if args.show_composites:
        print_composites(metadata)

    all_problems = (
        metadata_problems
        + document_problems
    )

    if args.show_problems:
        print("\nProblems")

        if not all_problems:
            print("(none)")
        else:
            for problem in all_problems[: args.limit]:
                print(f"- {problem}")

            remaining = (
                len(all_problems)
                - args.limit
            )

            if remaining > 0:
                print(
                    f"... and {remaining} more problem(s)."
                )

    if all_problems:
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())