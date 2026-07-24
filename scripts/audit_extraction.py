from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


SUPPORTED_SCHEMA_VERSIONS = {
    "1.0",
    "1.1",
}


@dataclass
class ProblemCollection:
    """
    Group audit problems by category while retaining a few examples.
    """

    counts: Counter[str] = field(
        default_factory=Counter
    )
    examples: dict[str, list[str]] = field(
        default_factory=lambda: defaultdict(list)
    )
    max_examples_per_category: int = 3

    def add(
        self,
        category: str,
        message: str,
    ) -> None:
        self.counts[category] += 1

        category_examples = self.examples[
            category
        ]

        if (
            len(category_examples)
            < self.max_examples_per_category
        ):
            category_examples.append(
                message
            )

    @property
    def total(self) -> int:
        return sum(
            self.counts.values()
        )


@dataclass
class AuditResult:
    schema_version: str | None = None
    document_count: int = 0
    dimension_count: int = 0

    kind_counts: Counter[str] = field(
        default_factory=Counter
    )
    measure_counts: Counter[str] = field(
        default_factory=Counter
    )
    class_counts: Counter[str] = field(
        default_factory=Counter
    )
    evidence_kind_counts: Counter[str] = field(
        default_factory=Counter
    )
    evidence_source_counts: Counter[str] = field(
        default_factory=Counter
    )

    ratio_dimensions: int = 0
    composite_incomplete: int = 0
    unimplemented: int = 0
    dimensions_with_evidence: int = 0

    metadata_problems: ProblemCollection = field(
        default_factory=ProblemCollection
    )
    document_problems: ProblemCollection = field(
        default_factory=ProblemCollection
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit a structured UMUTextStats JSONL extraction."
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
        default=3,
        help=(
            "Maximum examples retained for each problem category "
            "(default: 3)"
        ),
    )

    parser.add_argument(
        "--top",
        type=int,
        default=20,
        help=(
            "Maximum rows displayed in large summary tables "
            "(default: 20)"
        ),
    )

    parser.add_argument(
        "--strict",
        action="store_true",
        help=(
            "Exit with status 1 when audit problems are detected"
        ),
    )

    return parser.parse_args()


def read_json_record(
    line: str,
    *,
    line_number: int,
) -> dict[str, Any]:
    try:
        value = json.loads(
            line
        )
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Invalid JSON at line {line_number}: {exc}"
        ) from exc

    if not isinstance(
        value,
        dict,
    ):
        raise ValueError(
            f"JSONL line {line_number} must contain an object."
        )

    return value


def audit_extraction(
    path: Path,
    *,
    examples_per_category: int = 3,
) -> AuditResult:
    result = AuditResult()

    result.metadata_problems.max_examples_per_category = (
        examples_per_category
    )
    result.document_problems.max_examples_per_category = (
        examples_per_category
    )

    with path.open(
        encoding="utf-8",
    ) as stream:
        first_line = stream.readline()

        if not first_line:
            raise ValueError(
                "The input JSONL file is empty."
            )

        metadata = read_json_record(
            first_line,
            line_number=1,
        )

        audit_metadata(
            metadata=metadata,
            result=result,
        )

        for line_number, line in enumerate(
            stream,
            start=2,
        ):
            if not line.strip():
                continue

            document = read_json_record(
                line,
                line_number=line_number,
            )

            result.document_count += 1

            audit_document(
                metadata=metadata,
                document=document,
                line_number=line_number,
                result=result,
            )

    return result


def audit_metadata(
    metadata: Mapping[str, Any],
    result: AuditResult,
) -> None:
    if metadata.get("_type") != "metadata":
        result.metadata_problems.add(
            "invalid_record_type",
            "The first record must have '_type=metadata'.",
        )

    schema_version = metadata.get(
        "schema_version"
    )

    result.schema_version = (
        None
        if schema_version is None
        else str(schema_version)
    )

    if schema_version is None:
        result.metadata_problems.add(
            "missing_schema_version",
            "Metadata has no schema_version.",
        )
    elif (
        str(schema_version)
        not in SUPPORTED_SCHEMA_VERSIONS
    ):
        result.metadata_problems.add(
            "unsupported_schema_version",
            f"Unsupported schema version: {schema_version!r}.",
        )

    dimensions = metadata.get(
        "dimensions"
    )

    if not isinstance(
        dimensions,
        list,
    ):
        result.metadata_problems.add(
            "invalid_dimensions_list",
            "Metadata field 'dimensions' must be a list.",
        )
        dimensions = []

    result.dimension_count = len(
        dimensions
    )

    duplicates = [
        key
        for key, count in Counter(
            dimensions
        ).items()
        if count > 1
    ]

    for key in duplicates:
        result.metadata_problems.add(
            "duplicate_dimension",
            f"Duplicated dimension key: {key!r}.",
        )

    dimension_metadata = metadata.get(
        "dimension_metadata"
    )

    if not isinstance(
        dimension_metadata,
        Mapping,
    ):
        result.metadata_problems.add(
            "invalid_dimension_metadata",
            "'dimension_metadata' must be a mapping.",
        )
        dimension_metadata = {}

    for dimension_key in dimensions:
        raw_metadata = dimension_metadata.get(
            dimension_key
        )

        if raw_metadata is None:
            result.metadata_problems.add(
                "missing_dimension_metadata",
                f"Dimension {dimension_key!r} has no metadata.",
            )
            continue

        if not isinstance(
            raw_metadata,
            Mapping,
        ):
            result.metadata_problems.add(
                "invalid_dimension_metadata_entry",
                (
                    f"Metadata for dimension {dimension_key!r} "
                    "must be a mapping."
                ),
            )
            continue

        audit_dimension_metadata(
            dimension_key=str(
                dimension_key
            ),
            metadata=raw_metadata,
            result=result,
        )

    unexpected_metadata = (
        set(dimension_metadata)
        - set(dimensions)
    )

    for dimension_key in sorted(
        unexpected_metadata
    ):
        result.metadata_problems.add(
            "unlisted_dimension_metadata",
            (
                "Metadata exists for a dimension not listed in "
                f"'dimensions': {dimension_key!r}."
            ),
        )


def audit_dimension_metadata(
    dimension_key: str,
    metadata: Mapping[str, Any],
    result: AuditResult,
) -> None:
    kind = str(
        metadata.get(
            "kind",
            "unknown",
        )
    )

    class_name = str(
        metadata.get(
            "class_name",
            "unknown",
        )
    )

    measure = metadata.get(
        "measure"
    )

    if measure is None:
        measure = (
            "composite"
            if kind == "composite"
            else "unknown"
        )

    result.kind_counts[
        kind
    ] += 1

    result.class_counts[
        class_name
    ] += 1

    result.measure_counts[
        str(measure)
    ] += 1

    if kind == "unimplemented":
        result.unimplemented += 1

    if (
        measure == "ratio"
        or class_name == "RatioDimension"
    ):
        result.ratio_dimensions += 1

    if kind == "composite":
        missing_children = metadata.get(
            "missing_children"
        )

        if (
            isinstance(
                missing_children,
                list,
            )
            and missing_children
        ):
            result.composite_incomplete += 1

    descriptor = metadata.get(
        "evidence"
    )

    if descriptor is None:
        return

    result.dimensions_with_evidence += 1

    audit_evidence_descriptor(
        dimension_key=dimension_key,
        descriptor=descriptor,
        result=result,
    )


def audit_evidence_descriptor(
    dimension_key: str,
    descriptor: Any,
    result: AuditResult,
) -> None:
    if not isinstance(
        descriptor,
        Mapping,
    ):
        result.metadata_problems.add(
            "invalid_evidence_descriptor",
            (
                f"{dimension_key}: evidence descriptor "
                "must be a mapping."
            ),
        )
        return

    values: dict[str, str] = {}

    for field_name in (
        "kind",
        "source",
        "unit",
    ):
        value = descriptor.get(
            field_name
        )

        if (
            not isinstance(
                value,
                str,
            )
            or not value.strip()
        ):
            result.metadata_problems.add(
                "invalid_evidence_descriptor_field",
                (
                    f"{dimension_key}: invalid descriptor "
                    f"field {field_name!r}."
                ),
            )
            continue

        values[
            field_name
        ] = value

    if "kind" in values:
        result.evidence_kind_counts[
            values["kind"]
        ] += 1

    if "source" in values:
        result.evidence_source_counts[
            values["source"]
        ] += 1


def audit_document(
    metadata: Mapping[str, Any],
    document: Mapping[str, Any],
    line_number: int,
    result: AuditResult,
) -> None:
    if document.get("_type") != "document":
        result.document_problems.add(
            "invalid_document_type",
            f"line {line_number}: record has no '_type=document'.",
        )

    if (
        "id" not in document
        and "row" not in document
    ):
        result.document_problems.add(
            "missing_document_identifier",
            (
                f"line {line_number}: record has neither "
                "'id' nor 'row'."
            ),
        )

    dimensions = document.get(
        "dimensions"
    )

    if not isinstance(
        dimensions,
        Mapping,
    ):
        result.document_problems.add(
            "invalid_document_dimensions",
            (
                f"line {line_number}: 'dimensions' "
                "must be a mapping."
            ),
        )
        return

    expected_dimensions = metadata.get(
        "dimensions",
        [],
    )

    if isinstance(
        expected_dimensions,
        list,
    ):
        missing_dimensions = (
            set(expected_dimensions)
            - set(dimensions)
        )

        unexpected_dimensions = (
            set(dimensions)
            - set(expected_dimensions)
        )

        for key in sorted(
            missing_dimensions
        ):
            result.document_problems.add(
                "missing_dimension",
                (
                    f"line {line_number}: missing "
                    f"dimension {key!r}."
                ),
            )

        for key in sorted(
            unexpected_dimensions
        ):
            result.document_problems.add(
                "unexpected_dimension",
                (
                    f"line {line_number}: unexpected "
                    f"dimension {key!r}."
                ),
            )

    reference_lengths = document.get(
        "reference_lengths",
        {},
    )

    if reference_lengths is None:
        reference_lengths = {}

    if not isinstance(
        reference_lengths,
        Mapping,
    ):
        result.document_problems.add(
            "invalid_reference_lengths",
            (
                f"line {line_number}: 'reference_lengths' "
                "must be a mapping."
            ),
        )
        reference_lengths = {}

    for source, length in (
        reference_lengths.items()
    ):
        if not is_non_negative_integer(
            length
        ):
            result.document_problems.add(
                "invalid_reference_length",
                (
                    f"line {line_number}: invalid reference "
                    f"length for {source!r}: {length!r}."
                ),
            )

    dimension_metadata = metadata.get(
        "dimension_metadata",
        {},
    )

    if not isinstance(
        dimension_metadata,
        Mapping,
    ):
        dimension_metadata = {}

    for dimension_key, dimension_record in (
        dimensions.items()
    ):
        if not isinstance(
            dimension_record,
            Mapping,
        ):
            result.document_problems.add(
                "invalid_dimension_record",
                (
                    f"line {line_number}: {dimension_key}: "
                    "dimension record must be a mapping."
                ),
            )
            continue

        audit_dimension_document_record(
            dimension_key=str(
                dimension_key
            ),
            dimension_record=dimension_record,
            dimension_metadata=(
                dimension_metadata.get(
                    dimension_key,
                    {},
                )
            ),
            reference_lengths=reference_lengths,
            line_number=line_number,
            result=result,
        )


def audit_dimension_document_record(
    dimension_key: str,
    dimension_record: Mapping[str, Any],
    dimension_metadata: Any,
    reference_lengths: Mapping[str, Any],
    line_number: int,
    result: AuditResult,
) -> None:
    if "value" not in dimension_record:
        result.document_problems.add(
            "missing_dimension_value",
            (
                f"line {line_number}: {dimension_key}: "
                "missing value."
            ),
        )

    numerator = dimension_record.get(
        "numerator"
    )

    denominator = dimension_record.get(
        "denominator"
    )

    if (
        numerator is not None
        and not is_number(
            numerator
        )
    ):
        result.document_problems.add(
            "invalid_numerator",
            (
                f"line {line_number}: {dimension_key}: "
                f"invalid numerator {numerator!r}."
            ),
        )

    if (
        denominator is not None
        and not is_number(
            denominator
        )
    ):
        result.document_problems.add(
            "invalid_denominator",
            (
                f"line {line_number}: {dimension_key}: "
                f"invalid denominator {denominator!r}."
            ),
        )

    has_evidence_field = (
        "evidence"
        in dimension_record
    )

    evidence = dimension_record.get(
        "evidence"
    )

    if (
        has_evidence_field
        and not isinstance(
            evidence,
            list,
        )
    ):
        result.document_problems.add(
            "invalid_evidence_collection",
            (
                f"line {line_number}: {dimension_key}: "
                "evidence must be a list."
            ),
        )
        evidence = []

    if evidence is None:
        evidence = []

    descriptor = None

    if isinstance(
        dimension_metadata,
        Mapping,
    ):
        descriptor = dimension_metadata.get(
            "evidence"
        )

    if descriptor is not None:
        if not has_evidence_field:
            result.document_problems.add(
                "missing_evidence_field",
                (
                    f"line {line_number}: {dimension_key}: "
                    "dimension declares evidence but the "
                    "document has no evidence field."
                ),
            )

        audit_dimension_evidence(
            dimension_key=dimension_key,
            evidence=evidence,
            descriptor=descriptor,
            reference_lengths=reference_lengths,
            line_number=line_number,
            result=result,
        )

        audit_evidence_cardinality(
            dimension_key=dimension_key,
            numerator=numerator,
            evidence=evidence,
            descriptor=descriptor,
            line_number=line_number,
            result=result,
        )

    segments = dimension_record.get(
        "segments"
    )

    if (
        segments is not None
        and not isinstance(
            segments,
            Mapping,
        )
    ):
        result.document_problems.add(
            "invalid_segments",
            (
                f"line {line_number}: {dimension_key}: "
                "segments must be a mapping."
            ),
        )

    derived = dimension_record.get(
        "derived"
    )

    if (
        derived is not None
        and not isinstance(
            derived,
            Mapping,
        )
    ):
        result.document_problems.add(
            "invalid_derived_values",
            (
                f"line {line_number}: {dimension_key}: "
                "derived must be a mapping."
            ),
        )


def audit_evidence_cardinality(
    dimension_key: str,
    numerator: Any,
    evidence: list[Any],
    descriptor: Any,
    line_number: int,
    result: AuditResult,
) -> None:
    """
    Compare numerator and evidence only when explicitly requested.

    Future descriptors may declare:

        "cardinality": "numerator"

    Without that declaration, no relationship is assumed.
    """
    if not isinstance(
        descriptor,
        Mapping,
    ):
        return

    if (
        descriptor.get(
            "cardinality"
        )
        != "numerator"
    ):
        return

    if numerator is None:
        result.document_problems.add(
            "missing_evidence_numerator",
            (
                f"line {line_number}: {dimension_key}: "
                "descriptor declares numerator cardinality "
                "but numerator is absent."
            ),
        )
        return

    if not is_integer_like(
        numerator
    ):
        result.document_problems.add(
            "invalid_evidence_numerator",
            (
                f"line {line_number}: {dimension_key}: "
                f"evidence numerator is not integer-like: "
                f"{numerator!r}."
            ),
        )
        return

    if int(numerator) != len(
        evidence
    ):
        result.document_problems.add(
            "evidence_count_mismatch",
            (
                f"line {line_number}: {dimension_key}: "
                f"numerator {numerator!r} differs from "
                f"evidence count {len(evidence)}."
            ),
        )


def audit_dimension_evidence(
    dimension_key: str,
    evidence: list[Any],
    descriptor: Any,
    reference_lengths: Mapping[str, Any],
    line_number: int,
    result: AuditResult,
) -> None:
    if not isinstance(
        descriptor,
        Mapping,
    ):
        return

    kind = descriptor.get(
        "kind"
    )
    source = descriptor.get(
        "source"
    )
    unit = descriptor.get(
        "unit"
    )

    if not isinstance(
        source,
        str,
    ):
        return

    if source not in reference_lengths:
        result.document_problems.add(
            "missing_reference_length",
            (
                f"line {line_number}: {dimension_key}: "
                f"missing reference length for {source!r}."
            ),
        )
        return

    reference_length = (
        reference_lengths[source]
    )

    if not is_non_negative_integer(
        reference_length
    ):
        return

    # Other evidence kinds can be audited later by dedicated validators.
    if (
        kind != "text_span"
        or unit != "characters"
    ):
        return

    for occurrence_index, occurrence in enumerate(
        evidence
    ):
        audit_text_span_occurrence(
            dimension_key=dimension_key,
            occurrence=occurrence,
            occurrence_index=occurrence_index,
            reference_length=int(
                reference_length
            ),
            source=source,
            line_number=line_number,
            result=result,
        )


def audit_text_span_occurrence(
    dimension_key: str,
    occurrence: Any,
    occurrence_index: int,
    reference_length: int,
    source: str,
    line_number: int,
    result: AuditResult,
) -> None:
    if not isinstance(
        occurrence,
        Mapping,
    ):
        result.document_problems.add(
            "invalid_evidence_occurrence",
            (
                f"line {line_number}: {dimension_key}: "
                f"evidence {occurrence_index} must be a mapping."
            ),
        )
        return

    start = occurrence.get(
        "start"
    )
    end = occurrence.get(
        "end"
    )

    if not is_non_negative_integer(
        start
    ):
        result.document_problems.add(
            "invalid_evidence_start",
            (
                f"line {line_number}: {dimension_key}: "
                f"evidence {occurrence_index} has invalid "
                f"start {start!r}."
            ),
        )
        return

    if not is_non_negative_integer(
        end
    ):
        result.document_problems.add(
            "invalid_evidence_end",
            (
                f"line {line_number}: {dimension_key}: "
                f"evidence {occurrence_index} has invalid "
                f"end {end!r}."
            ),
        )
        return

    start_value = int(
        start
    )
    end_value = int(
        end
    )

    if start_value > end_value:
        result.document_problems.add(
            "reversed_evidence_span",
            (
                f"line {line_number}: {dimension_key}: "
                f"evidence {occurrence_index} has start "
                f"{start_value} greater than end {end_value}."
            ),
        )
        return

    if end_value > reference_length:
        result.document_problems.add(
            "evidence_out_of_bounds",
            (
                f"line {line_number}: {dimension_key}: "
                f"evidence {occurrence_index} exceeds "
                f"{source!r} length "
                f"({start_value}, {end_value}, "
                f"length={reference_length})."
            ),
        )


def is_number(
    value: Any,
) -> bool:
    return (
        isinstance(
            value,
            (int, float),
        )
        and not isinstance(
            value,
            bool,
        )
    )


def is_integer_like(
    value: Any,
) -> bool:
    if isinstance(
        value,
        bool,
    ):
        return False

    if isinstance(
        value,
        int,
    ):
        return True

    if isinstance(
        value,
        float,
    ):
        return value.is_integer()

    return False


def is_non_negative_integer(
    value: Any,
) -> bool:
    return (
        is_integer_like(
            value
        )
        and int(value) >= 0
    )


def print_counter_table(
    title: str,
    key_title: str,
    counter: Counter[str],
    *,
    limit: int | None = None,
) -> None:
    print()
    print(title)

    if not counter:
        print("(none)")
        return

    rows = sorted(
        counter.items(),
        key=lambda item: (
            -item[1],
            item[0],
        ),
    )

    hidden = 0

    if (
        limit is not None
        and len(rows) > limit
    ):
        hidden = len(rows) - limit
        rows = rows[:limit]

    key_width = max(
        len(key_title),
        max(
            len(str(key))
            for key, _ in rows
        ),
    )

    count_width = max(
        len("count"),
        max(
            len(str(count))
            for _, count in rows
        ),
    )

    print(
        f"{key_title:<{key_width}}  "
        f"{'count':>{count_width}}"
    )

    print(
        f"{'-' * key_width}  "
        f"{'-' * count_width}"
    )

    for key, count in rows:
        print(
            f"{key:<{key_width}}  "
            f"{count:>{count_width}}"
        )

    if hidden:
        print(
            f"... {hidden} additional rows omitted"
        )


def print_problem_collection(
    title: str,
    problems: ProblemCollection,
) -> None:
    print()
    print(title)

    if problems.total == 0:
        print("(none)")
        return

    print(
        f"Total: {problems.total}"
    )

    for category, count in (
        problems.counts.most_common()
    ):
        print()
        print(
            f"{category}: {count}"
        )

        for example in problems.examples.get(
            category,
            [],
        ):
            print(
                f"  - {example}"
            )


def print_audit(
    result: AuditResult,
    *,
    top: int,
) -> None:
    print("Extraction audit")
    print("================")

    print(
        f"Schema version:          "
        f"{result.schema_version}"
    )
    print(
        f"Documents:               "
        f"{result.document_count}"
    )
    print(
        f"Dimensions:              "
        f"{result.dimension_count}"
    )
    print(
        f"Atomic:                  "
        f"{result.kind_counts.get('atomic', 0)}"
    )
    print(
        f"Derived:                 "
        f"{result.kind_counts.get('derived', 0)}"
    )
    print(
        f"Composite:               "
        f"{result.kind_counts.get('composite', 0)}"
    )
    print(
        f"Unimplemented:           "
        f"{result.unimplemented}"
    )
    print(
        f"Ratio dimensions:        "
        f"{result.ratio_dimensions}"
    )
    print(
        f"Dimensions with evidence:"
        f" {result.dimensions_with_evidence}"
    )
    print(
        f"Composites incomplete:   "
        f"{result.composite_incomplete}"
    )
    print(
        f"Metadata problems:       "
        f"{result.metadata_problems.total}"
    )
    print(
        f"Document problems:       "
        f"{result.document_problems.total}"
    )

    print_counter_table(
        title="Kinds",
        key_title="kind",
        counter=result.kind_counts,
    )

    print_counter_table(
        title="Measures",
        key_title="measure",
        counter=result.measure_counts,
    )

    print_counter_table(
        title="Evidence kinds",
        key_title="kind",
        counter=result.evidence_kind_counts,
    )

    print_counter_table(
        title="Evidence sources",
        key_title="source",
        counter=result.evidence_source_counts,
    )

    print_counter_table(
        title=f"Classes (top {top})",
        key_title="class",
        counter=result.class_counts,
        limit=top,
    )

    print_problem_collection(
        title="Metadata problems",
        problems=result.metadata_problems,
    )

    print_problem_collection(
        title="Document problems",
        problems=result.document_problems,
    )


def main() -> None:
    args = parse_args()

    result = audit_extraction(
        args.input,
        examples_per_category=args.examples,
    )

    print_audit(
        result=result,
        top=args.top,
    )

    if (
        args.strict
        and (
            result.metadata_problems.total
            or result.document_problems.total
        )
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()