# src/umutextstats/cache/cache.py

import hashlib
import json
from pathlib import Path

import pandas as pd


class CacheManager:
    def __init__(self, cache_dir: str = ".cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

    def build_key(self, input_path: str, stage: str, params: dict) -> str:
        input_path = Path(input_path)

        payload = {
            "path": str(input_path.resolve()),
            "mtime": input_path.stat().st_mtime,
            "size": input_path.stat().st_size,
            "stage": stage,
            "params": params,
        }

        raw = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        return hashlib.md5(raw).hexdigest()

    def path_for(self, stage: str, key: str) -> Path:
        return self.cache_dir / f"{stage}_{key}.parquet"

    def load(self, stage: str, key: str) -> pd.DataFrame | None:
        path = self.path_for(stage, key)

        if not path.exists():
            return None

        print(f"[CACHE] Loading {stage}: {path}")
        return pd.read_parquet(path)

    def save(self, df: pd.DataFrame, stage: str, key: str) -> None:
        path = self.path_for(stage, key)
        print(f"[CACHE] Saving {stage}: {path}")
        df.to_parquet(path, index=False)