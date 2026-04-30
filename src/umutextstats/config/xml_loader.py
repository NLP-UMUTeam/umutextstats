# src/umutextstats/config/xml_loader.py

import xml.etree.ElementTree as ET
from pathlib import Path

from umutextstats.config.models import DimensionConfig, UMUTextStatsConfig


KNOWN_FIELDS = {
    "key",
    "class",
    "strategy",
    "description",
    "dictionary",
    "pattern",
    "universal",
    "useoriginalinput",
    "dimensions",
}


def _text(element, tag: str) -> str | None:
    child = element.find(tag)

    if child is None or child.text is None:
        return None

    value = child.text.strip()
    return value if value else None


def _bool_text(value: str | None) -> bool:
    return value is not None and value.lower() in {"true", "1", "yes"}


def _parse_extra_params(element) -> dict[str, str]:
    params = {}

    for child in element:
        if child.tag in KNOWN_FIELDS:
            continue

        if child.text is not None:
            value = child.text.strip()
            if value:
                params[child.tag] = value

    return params


def parse_dimension(element) -> DimensionConfig:
    children = []

    dimensions_element = element.find("dimensions")
    if dimensions_element is not None:
        for child_dimension in dimensions_element.findall("dimension"):
            children.append(parse_dimension(child_dimension))

    return DimensionConfig(
        key=_text(element, "key") or "",
        class_name=_text(element, "class"),
        strategy=_text(element, "strategy"),
        description=_text(element, "description"),
        dictionary=_text(element, "dictionary"),
        pattern=_text(element, "pattern"),
        universal=_text(element, "universal"),
        use_original_input=_bool_text(_text(element, "useoriginalinput")),
        children=children,
        params=_parse_extra_params(element),
    )


def load_xml_config(path: str | Path) -> UMUTextStatsConfig:
    path = Path(path)

    tree = ET.parse(path)
    root = tree.getroot()

    dimensions = []

    dimensions_element = root.find("dimensions")
    if dimensions_element is not None:
        for dimension_element in dimensions_element.findall("dimension"):
            dimensions.append(parse_dimension(dimension_element))

    return UMUTextStatsConfig(
        directory_folder=_text(root, "directory_folder"),
        dimensions=dimensions,
    )