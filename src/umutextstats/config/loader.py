# src/umutextstats/config/loader.py

from importlib.resources import files
from pathlib import Path

from umutextstats.config.models import UMUTextStatsConfig
from umutextstats.config.xml_loader import load_xml_config
from umutextstats.config.yaml_loader import load_yaml_config


def default_config_path() -> Path:
    return Path(
        files("umutextstats")
        / "resources"
        / "config"
        / "default.yaml"
    )


def load_config(path: str | Path | None = None) -> UMUTextStatsConfig:
    if path is None:
        path = default_config_path()

    path = Path(path)

    if path.suffix in {".yaml", ".yml"}:
        return load_yaml_config(path)

    if path.suffix == ".xml":
        return load_xml_config(path)

    raise ValueError(f"Unsupported config format: {path.suffix}")