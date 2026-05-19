# src/umutextstats/config/yaml_loader.py

from pathlib import Path
from typing import Any

import yaml

from umutextstats.config.models import DimensionConfig, UMUTextStatsConfig


KNOWN_FIELDS = {
    "key",
    "class",
    "class_name",
    "strategy",
    "description",
    "dictionary",
    "pattern",
    "universal",
    "use_original_input",
    "useoriginalinput",
    "percentage",
    "disabled_regexp",
    "disabledregexp",
    "children",
    "dimensions",
}


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default

    if isinstance(value, bool):
        return value

    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y"}

    return bool(value)


def _load_dimension(data: dict[str, Any]) -> DimensionConfig:
    children_data = data.get("children", data.get("dimensions", []))

    params = {
        key: value
        for key, value in data.items()
        if key not in KNOWN_FIELDS
    }

    return DimensionConfig(
        key=data["key"],
        class_name=data.get("class_name") or data.get("class"),
        strategy=data.get("strategy"),
        description=data.get("description"),
        dictionary=data.get("dictionary"),
        pattern=data.get("pattern"),
        universal=data.get("universal"),
        use_original_input=_as_bool(
            data.get("use_original_input", data.get("useoriginalinput")),
            default=False,
        ),
        percentage=_as_bool(data.get("percentage"), default=True),
        disabled_regexp=_as_bool(
            data.get("disabled_regexp", data.get("disabledregexp")),
            default=False,
        ),
        children=[
            _load_dimension(child)
            for child in children_data
        ],
        params=params,
    )


def load_yaml_config(path: str | Path) -> UMUTextStatsConfig:
    path = Path(path)

    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)

    return UMUTextStatsConfig(
        directory_folder=data.get("directory_folder"),
        dimensions=[
            _load_dimension(dimension)
            for dimension in data.get("dimensions", [])
        ],
    )


def load_yaml_config(path: str | Path) -> UMUTextStatsConfig:
    path = Path(path)

    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}

    directory_folder = data.get("directory_folder")
    dimensions_data = list(data.get("dimensions", []))

    for include in data.get("includes", []):
        include_path = path.parent / include

        with include_path.open("r", encoding="utf-8") as file:
            include_data = yaml.safe_load(file) or {}

        dimensions_data.extend(include_data.get("dimensions", []))

    return UMUTextStatsConfig(
        directory_folder=directory_folder,
        dimensions=[
            _load_dimension(dimension)
            for dimension in dimensions_data
        ],
    )