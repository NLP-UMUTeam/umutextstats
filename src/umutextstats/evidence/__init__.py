from umutextstats.evidence.distribution import (
    build_positional_distribution,
)
from umutextstats.evidence.jsonl import (
    dimension_distribution_from_document,
    dimension_evidence_descriptor,
    dimension_occurrences_from_document,
    dimension_position_metadata,
    document_reference_length,
    evidence_occurrence_from_record,
    evidence_occurrences_from_dimension_record,
)

from umutextstats.evidence.aggregation import (
    AggregatedPositionalDistribution,
    aggregate_positional_distributions,
)

from umutextstats.evidence.models import (
    EvidenceOccurrence,
    PositionalDistribution,
    PositionalSegment,
)

from umutextstats.evidence.service import (
    PositionalAggregationResult,
    aggregate_dimension_from_jsonl,
    iter_dimension_distributions_from_jsonl,
)

__all__ = [
    "EvidenceOccurrence",
    "PositionalDistribution",
    "PositionalSegment",
    "build_positional_distribution",
    "dimension_distribution_from_document",
    "dimension_occurrences_from_document",
    "dimension_position_metadata",
    "evidence_occurrence_from_record",
    "evidence_occurrences_from_dimension_record",
    "AggregatedPositionalDistribution",
    "aggregate_positional_distributions",
    "dimension_evidence_descriptor",
    "document_reference_length",
    "PositionalAggregationResult",
    "aggregate_dimension_from_jsonl",
    "iter_dimension_distributions_from_jsonl",
]