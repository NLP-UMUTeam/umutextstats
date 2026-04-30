# src/umutextstats/config/models.py

from dataclasses import dataclass, field


@dataclass
class DimensionConfig:
    key: str
    class_name: str | None = None
    strategy: str | None = None
    description: str | None = None
    dictionary: str | None = None
    pattern: str | None = None
    universal: str | None = None
    use_original_input: bool = False
    children: list["DimensionConfig"] = field(default_factory=list)
    params: dict[str, str] = field(default_factory=dict)


@dataclass
class UMUTextStatsConfig:
    directory_folder: str | None = None
    dimensions: list[DimensionConfig] = field(default_factory=list)