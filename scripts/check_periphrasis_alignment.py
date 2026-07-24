from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Check periphrasis evidence stored in a structured "
            "UMUTextStats JSONL file."
        )
    )

    parser.add_argument(
        "input",
        type=Path,
        help="Structured JSONL extraction file",
    )

    parser.add_argument(
        "--examples",
        type=int,
        default=5,
        help="Examples retained for each problem category",
    )

    return parser.parse_args()


def is_non_negative_integer(
    value: Any,
) -> bool:
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        and value >= 0
    )


def main() -> None:
    args = parse_args()

    problem_counts: Counter[str] = Counter()
    examples: dict[str, list[str]] = defaultdict(list)

    descriptor_counts: Counter[str] = Counter()
    dimension_counts: Counter[str] = Counter()

    total_occurrences = 0
    valid_occurrences = 0

    def add_problem(
        category: str,
        message: str,
    ) -> None:
        problem_counts[category] += 1

        if len(examples[category]) < args.examples:
            examples[category].append(message)

    with args.input.open(
        encoding="utf-8",
    ) as stream:
        metadata = json.loads(
            next(stream)
        )

        dimension_metadata = metadata.get(
            "dimension_metadata",
            {},
        )

        periphrasis_keys = {
            key
            for key, value in dimension_metadata.items()
            if value.get("class_name") == "PeriphrasisDimension"
        }

        print("Periphrasis dimensions:", len(periphrasis_keys))

        for key in sorted(periphrasis_keys):
            descriptor = (
                dimension_metadata
                .get(key, {})
                .get("evidence")
            )

            descriptor_counts[
                json.dumps(
                    descriptor,
                    sort_keys=True,
                )
            ] += 1

        for line_number, line in enumerate(
            stream,
            start=2,
        ):
            if not line.strip():
                continue

            document = json.loads(
                line
            )

            dimensions = document.get(
                "dimensions",
                {},
            )

            for key in periphrasis_keys:
                record = dimensions.get(
                    key,
                    {},
                )

                evidence = record.get(
                    "evidence"
                ) or []

                if evidence:
                    dimension_counts[key] += len(
                        evidence
                    )

                for occurrence_index, occurrence in enumerate(
                    evidence
                ):
                    total_occurrences += 1

                    if not isinstance(
                        occurrence,
                        dict,
                    ):
                        add_problem(
                            "invalid_occurrence",
                            (
                                f"line {line_number}, {key}, "
                                f"evidence {occurrence_index}: "
                                "not an object"
                            ),
                        )
                        continue

                    text = occurrence.get(
                        "text"
                    )

                    token_start = occurrence.get(
                        "token_start"
                    )

                    token_end = occurrence.get(
                        "token_end"
                    )

                    if not isinstance(
                        text,
                        str,
                    ):
                        add_problem(
                            "invalid_text",
                            (
                                f"line {line_number}, {key}, "
                                f"evidence {occurrence_index}: "
                                f"text={text!r}"
                            ),
                        )
                        continue

                    if not is_non_negative_integer(
                        token_start
                    ):
                        add_problem(
                            "invalid_token_start",
                            (
                                f"line {line_number}, {key}, "
                                f"evidence {occurrence_index}: "
                                f"token_start={token_start!r}"
                            ),
                        )
                        continue

                    if not is_non_negative_integer(
                        token_end
                    ):
                        add_problem(
                            "invalid_token_end",
                            (
                                f"line {line_number}, {key}, "
                                f"evidence {occurrence_index}: "
                                f"token_end={token_end!r}"
                            ),
                        )
                        continue

                    if token_start >= token_end:
                        add_problem(
                            "invalid_token_range",
                            (
                                f"line {line_number}, {key}, "
                                f"evidence {occurrence_index}: "
                                f"[{token_start}, {token_end})"
                            ),
                        )
                        continue

                    span_length = (
                        token_end
                        - token_start
                    )

                    text_token_count = len(
                        text.split()
                    )

                    if span_length != text_token_count:
                        add_problem(
                            "span_text_length_mismatch",
                            (
                                f"line {line_number}, {key}, "
                                f"evidence {occurrence_index}: "
                                f"span={span_length}, "
                                f"text tokens={text_token_count}, "
                                f"text={text!r}"
                            ),
                        )
                        continue

                    valid_occurrences += 1

    print()
    print("Descriptors")

    for descriptor, count in (
        descriptor_counts.most_common()
    ):
        print(f"{count:>4}  {descriptor}")

    print()
    print("Occurrences")
    print(f"Total: {total_occurrences}")
    print(f"Valid: {valid_occurrences}")

    print()
    print("Occurrences by dimension")

    for key, count in (
        dimension_counts.most_common()
    ):
        print(f"{count:>4}  {key}")

    print()
    print("Problems")

    if not problem_counts:
        print("(none)")
        return

    for category, count in (
        problem_counts.most_common()
    ):
        print()
        print(f"{category}: {count}")

        for example in examples[category]:
            print(f"  - {example}")


if __name__ == "__main__":
    main()