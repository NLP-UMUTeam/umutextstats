# src/umutextstats/dimensions/engine.py

import pandas as pd

from umutextstats.config.models import DimensionConfig, UMUTextStatsConfig
from umutextstats.dimensions.registry import resolve_dimension


class DimensionEngine:
    def __init__(
        self,
        config: UMUTextStatsConfig,
        input_column: str = "text_norm",
        include_unimplemented: bool = True,
    ):
        self.config = config
        self.input_column = input_column
        self.include_unimplemented = include_unimplemented

    def compute(self, df) -> pd.DataFrame:
        data = {}

        if "id" in df.columns:
            data["id"] = df["id"]

        for dimension in self._iter_dimensions(self.config.dimensions):
            key = dimension.key
            dimension_cls = resolve_dimension(dimension.class_name)

            if dimension_cls is None:
                if self.include_unimplemented:
                    data[key] = [""] * len(df)
                continue

            instance = dimension_cls(
                key=key,
                input_column=self.input_column,
            )

            data[key] = instance.compute(df)

        return pd.DataFrame(data)
    def _iter_dimensions(self, dimensions: list[DimensionConfig]):
        for dimension in dimensions:
            yield dimension

            if dimension.children:
                yield from self._iter_dimensions(dimension.children)