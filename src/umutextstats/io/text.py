# src/umutextstats/io/text.py

from typing import Any
import pandas as pd


def ensure_text(value: Any) -> str:
    """Normalize any input value into a safe string for text analysis."""
    if pd.isna(value):
        return ""
    return str(value)