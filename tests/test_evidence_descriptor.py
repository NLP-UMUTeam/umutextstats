from __future__ import annotations

import pandas as pd
import pytest

from umutextstats.evidence.descriptors import (
    EvidenceDescriptor,
)
from umutextstats.extraction.engine import (
    ExtractionEngine,
)


def test_evidence_descriptor_to_dict():
    descriptor = EvidenceDescriptor(
        kind="text_span",
        source="text_norm",
        unit="characters",
    )

    assert descriptor.to_dict() == {
        "kind": "text_span",
        "source": "text_norm",
        "unit": "characters",
    }


@pytest.mark.parametrize(
    ("field", "values"),
    [
        (
            "kind",
            {
                "kind": "",
                "source": "text_norm",
                "unit": "characters",
            },
        ),
        (
            "source",
            {
                "kind": "text_span",
                "source": "",
                "unit": "characters",
            },
        ),
        (
            "unit",
            {
                "kind": "text_span",
                "source": "text_norm",
                "unit": "",
            },
        ),
    ],
)
def test_evidence_descriptor_rejects_empty_fields(
    field,
    values,
):
    with pytest.raises(
        ValueError,
        match=field,
    ):
        EvidenceDescriptor(
            **values
        )


def test_engine_uses_explicit_descriptor():
    explicit = EvidenceDescriptor(
        kind="token_span",
        source="tokens",
        unit="tokens",
    )

    class FakeDimension:
        def evidence_descriptor(self):
            return EvidenceDescriptor(
                kind="text_span",
                source="text_norm",
                unit="characters",
            )

    resolved = (
        ExtractionEngine._resolve_evidence_descriptor(
            instance=FakeDimension(),
            explicit_descriptor=explicit,
            evidence=None,
        )
    )

    assert resolved is explicit


def test_engine_uses_explicit_descriptor_even_without_evidence():
    explicit = EvidenceDescriptor(
        kind="token_span",
        source="tokens",
        unit="tokens",
    )

    class FakeDimension:
        pass

    resolved = (
        ExtractionEngine._resolve_evidence_descriptor(
            instance=FakeDimension(),
            explicit_descriptor=explicit,
            evidence=None,
        )
    )

    assert resolved is explicit


def test_engine_uses_dimension_default_descriptor_when_evidence_exists():
    expected = EvidenceDescriptor(
        kind="text_span",
        source="text_norm",
        unit="characters",
    )

    evidence = pd.Series(
        [
            [],
            [
                {
                    "start": 0,
                    "end": 4,
                }
            ],
        ],
        dtype=object,
    )

    class FakeDimension:
        def evidence_descriptor(self):
            return expected

    resolved = (
        ExtractionEngine._resolve_evidence_descriptor(
            instance=FakeDimension(),
            explicit_descriptor=None,
            evidence=evidence,
        )
    )

    assert resolved == expected


def test_engine_uses_dimension_default_descriptor_for_empty_evidence_batch():
    expected = EvidenceDescriptor(
        kind="text_span",
        source="text_norm",
        unit="characters",
    )

    evidence = pd.Series(
        [
            [],
            [],
        ],
        dtype=object,
    )

    class FakeDimension:
        def evidence_descriptor(self):
            return expected

    resolved = (
        ExtractionEngine._resolve_evidence_descriptor(
            instance=FakeDimension(),
            explicit_descriptor=None,
            evidence=evidence,
        )
    )

    assert resolved == expected


def test_engine_does_not_use_default_descriptor_without_evidence():
    class FakeDimension:
        def evidence_descriptor(self):
            return EvidenceDescriptor(
                kind="text_span",
                source="text_norm",
                unit="characters",
            )

    resolved = (
        ExtractionEngine._resolve_evidence_descriptor(
            instance=FakeDimension(),
            explicit_descriptor=None,
            evidence=None,
        )
    )

    assert resolved is None


def test_engine_returns_none_without_descriptor():
    evidence = pd.Series(
        [
            [],
        ],
        dtype=object,
    )

    class FakeDimension:
        pass

    resolved = (
        ExtractionEngine._resolve_evidence_descriptor(
            instance=FakeDimension(),
            explicit_descriptor=None,
            evidence=evidence,
        )
    )

    assert resolved is None