# src/umutextstats/dimensions/results.py

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass
class DimensionComputation:
    """
    Raw batch computation produced by a dimension.

    `values` contains the final exported dimension value.

    `numerators` and `denominators` describe how normalized values were
    calculated when that information is available.
    """

    values: pd.Series
    numerators: pd.Series | None = None
    denominators: pd.Series | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class DimensionComputation:
    """
    Structured batch computation produced by a dimension.
    """

    values: pd.Series
    numerators: pd.Series | None = None
    denominators: pd.Series | None = None
    evidence: pd.Series | None = None
    metadata: dict[str, Any] = field(default_factory=dict)