# src/umutextstats/config/loader.py

from pathlib import Path
from importlib.resources import files

from umutextstats.config.models import UMUTextStatsConfig
from umutextstats.config.xml_loader import load_xml_config


def default_config_path() -> Path:
    return Path(
        files("umutextstats")
        / "resources"
        / "config"
        / "default.xml"
    )


def load_config(path: str | Path | None = None) -> UMUTextStatsConfig:
    if path is None:
        path = default_config_path()

    return load_xml_config(path)