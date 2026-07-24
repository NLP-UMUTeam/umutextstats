from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class EvidenceDescriptor:
    """
    Describe how a dimension's evidence must be interpreted.

    Examples:

    - text spans measured in characters;
    - token spans measured in token indexes;
    - temporal intervals measured in seconds.
    """

    kind: str
    source: str
    unit: str

    def __post_init__(self) -> None:
        if not self.kind.strip():
            raise ValueError(
                "Evidence descriptor kind cannot be empty."
            )

        if not self.source.strip():
            raise ValueError(
                "Evidence descriptor source cannot be empty."
            )

        if not self.unit.strip():
            raise ValueError(
                "Evidence descriptor unit cannot be empty."
            )

    def to_dict(self) -> dict[str, Any]:
        """
        Return a JSON-serializable representation.
        """
        return {
            "kind": self.kind,
            "source": self.source,
            "unit": self.unit,
        }