# src/umutextstats/utils/profiler.py

import time
import pandas as pd
from contextlib import contextmanager


class Profiler:
    def __init__(self, enabled: bool = False):
        self.enabled = enabled
        self.records = []

    @contextmanager
    def track(self, stage: str, name: str, **meta):
        if not self.enabled:
            yield
            return

        start = time.perf_counter() 

        try:
            yield
            status = "ok"
        except Exception:
            status = "error"
            raise
        finally:
            elapsed = time.perf_counter() - start
            self.records.append({
                "stage": stage,
                "name": name,
                "status": status,
                "seconds": elapsed,
                **meta,
            })

    def dataframe(self):
        return pd.DataFrame(self.records)