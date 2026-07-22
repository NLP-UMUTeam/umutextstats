from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from umutextstats.config.models import (
    DimensionConfig,
    UMUTextStatsConfig,
)


KNOWN_FIELDS = {
    "key",
    "class",
    "class_name",
    "strategy",
    "description",
    "dictionary",
    "pattern",
    "validation",
    "universal",
    "pos_tag",
    "pos_input_column",
    "input_column",
    "percentage",
    "disabled_regexp",
    "children",
    "dimensions",
    "params",
}


def _as_bool(
    value: Any,
    default: bool = False,
) -> bool:
    """
    Convert common YAML scalar values to booleans.
    """
    if value is None:
        return default

    if isinstance(value, bool):
        return value

    if isinstance(value, str):
        return value.strip().lower() in {
            "true",
            "1",
            "yes",
            "y",
        }

    return bool(value)


def _load_params(
    data: dict[str, Any],
    dimension_key: str,
) -> dict[str, Any]:
    """
    Load dimension-specific parameters.

    Parameters can be declared either directly as unknown top-level
    fields or inside an explicit `params` mapping.

    Both forms may be combined as long as they do not define the same
    parameter more than once.
    """
    direct_params = {
        key: value
        for key, value in data.items()
        if key not in KNOWN_FIELDS
    }

    explicit_params = data.get(
        "params",
        {},
    )

    if explicit_params is None:
        explicit_params = {}

    if not isinstance(
        explicit_params,
        dict,
    ):
        raise ValueError(
            f"Dimension '{dimension_key}' field "
            "'params' must be a mapping."
        )

    duplicated = sorted(
        set(direct_params)
        & set(explicit_params)
    )

    if duplicated:
        duplicated_text = ", ".join(
            duplicated
        )

        raise ValueError(
            f"Dimension '{dimension_key}' defines "
            f"parameter(s) {duplicated_text} both "
            "directly and inside 'params'."
        )

    return {
        **direct_params,
        **explicit_params,
    }


def _load_dimension(
    data: dict[str, Any],
) -> DimensionConfig:
    """
    Load a single dimension configuration from YAML data.

    Dimension-specific parameters may be declared as additional
    top-level fields or inside an explicit `params` mapping.
    """
    if not isinstance(
        data,
        dict,
    ):
        raise ValueError(
            "Each dimension must be a YAML mapping."
        )

    if "key" not in data:
        raise ValueError(
            "Each dimension must define a 'key'."
        )

    dimension_key = str(
        data["key"]
    )

    children_data = data.get(
        "children",
        data.get(
            "dimensions",
            [],
        ),
    )

    if children_data is None:
        children_data = []

    if not isinstance(
        children_data,
        list,
    ):
        raise ValueError(
            f"Dimension '{dimension_key}' field "
            "'children' must be a list."
        )

    params = _load_params(
        data=data,
        dimension_key=dimension_key,
    )

    return DimensionConfig(
        key=dimension_key,
        class_name=(
            data.get("class_name")
            or data.get("class")
        ),
        strategy=data.get("strategy"),
        description=data.get(
            "description"
        ),
        dictionary=data.get(
            "dictionary"
        ),
        pattern=data.get("pattern"),
        universal=data.get(
            "universal"
        ),
        pos_tag=data.get("pos_tag"),
        pos_input_column=data.get(
            "pos_input_column",
            "tagged_pos",
        ),
        input_column=data.get(
            "input_column"
        ),
        validation=data.get(
            "validation"
        ),
        percentage=_as_bool(
            data.get("percentage"),
            default=True,
        ),
        disabled_regexp=_as_bool(
            data.get(
                "disabled_regexp",
                False,
            )
        ),
        children=[
            _load_dimension(child)
            for child in children_data
        ],
        params=params,
    )


def load_yaml_config(
    path: str | Path,
) -> UMUTextStatsConfig:
    """
    Load a UMUTextStats YAML configuration file.

    Included files are resolved relative to the main configuration file.
    """
    path = Path(path)

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        data = yaml.safe_load(
            file
        ) or {}

    if not isinstance(
        data,
        dict,
    ):
        raise ValueError(
            "The YAML configuration root "
            "must be a mapping."
        )

    directory_folder = data.get(
        "directory_folder"
    )

    dimensions_data = list(
        data.get(
            "dimensions",
            [],
        )
    )

    for include in data.get(
        "includes",
        [],
    ):
        include_path = (
            path.parent
            / include
        )

        with include_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            include_data = yaml.safe_load(
                file
            ) or {}

        if not isinstance(
            include_data,
            dict,
        ):
            raise ValueError(
                f"Included configuration "
                f"'{include_path}' must contain "
                "a YAML mapping."
            )

        dimensions_data.extend(
            include_data.get(
                "dimensions",
                [],
            )
        )

    return UMUTextStatsConfig(
        directory_folder=directory_folder,
        dimensions=[
            _load_dimension(
                dimension
            )
            for dimension
            in dimensions_data
        ],
    )