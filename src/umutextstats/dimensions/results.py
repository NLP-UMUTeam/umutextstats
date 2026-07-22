# src/umutextstats/dimensions/results.py

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

@dataclass
class DimensionComputation:
    values: pd.Series
    numerators: pd.Series | None = None
    denominators: pd.Series | None = None
    evidence: pd.Series | None = None
    metadata: dict[str, Any] = field(default_factory=dict)