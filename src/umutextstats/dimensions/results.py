from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from umutextstats.evidence.descriptors import (
    EvidenceDescriptor,
)


@dataclass
class DimensionComputation:
    values: pd.Series
    numerators: pd.Series | None = None
    denominators: pd.Series | None = None
    evidence: pd.Series | None = None
    evidence_descriptor: EvidenceDescriptor | None = None
    metadata: dict[str, Any] = field(
        default_factory=dict
    )