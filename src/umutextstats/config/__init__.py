# src/umutextstats/config/__init__.py

from umutextstats.config.loader import load_config, default_config_path
from umutextstats.config.models import DimensionConfig, UMUTextStatsConfig

__all__ = [
    "load_config",
    "default_config_path",
    "DimensionConfig",
    "UMUTextStatsConfig",
]