from contextlib import nullcontext
from tqdm.auto import tqdm

import pandas as pd

from umutextstats.config.models import DimensionConfig, UMUTextStatsConfig
from umutextstats.dimensions.registry import resolve_dimension, normalize_class_name
from umutextstats.config.params import param, split_param_list
from umutextstats.dimensions.factory import build_dimension_instance

class DimensionEngine:
    def __init__(
        self,
        config: UMUTextStatsConfig,
        input_column: str = "text_norm",
        include_unimplemented: bool = True,
        profiler=None,
        show_progress: bool = True,
    ):
        self.config = config
        self.input_column = input_column
        self.include_unimplemented = include_unimplemented
        self.profiler = profiler
        self.show_progress = show_progress

    def compute(self, df) -> pd.DataFrame:
        data = {}

        if "id" in df.columns:
            data["id"] = df["id"]

        dimensions = list(self._iter_dimensions(self.config.dimensions))

        iterator = tqdm(
            dimensions,
            desc="Dimensions",
            unit="dimension",
            disable=not self.show_progress,
        )

        for dimension in iterator:
            iterator.set_postfix_str(dimension.key)
            self._compute_dimension(df, dimension, data)

        return pd.DataFrame(data)

    def _iter_dimensions(self, dimensions: list[DimensionConfig]):
        for dimension in dimensions:
            yield dimension

            if dimension.children:
                yield from self._iter_dimensions(dimension.children)

    def _compute_dimension(self, df, dimension: DimensionConfig, data: dict):
        
        key = dimension.key
        
        if dimension.key in data:
            return
        
        class_name = normalize_class_name(dimension.class_name)

        with self._track_dimension(key, class_name):
            for child in dimension.children:
                self._compute_dimension(df, child, data)

            if dimension.children:
                data[key] = self._compute_composite_dimension(
                    dimension,
                    data,
                    len(df),
                )
                return

            dimension_cls = resolve_dimension(dimension.class_name)
            
            if class_name == "RatioDimension":
                data[key] = self._compute_ratio_dimension(
                    dimension,
                    data,
                    len(df),
                )
                return

            if dimension_cls is None:
                if self.include_unimplemented:
                    data[key] = [""] * len(df)
                return

            instance = build_dimension_instance (
                dimension=dimension,
                dimension_cls=dimension_cls,
                default_input_column=self.input_column,
            )
            data[key] = instance.compute(df)

    def _compute_composite_dimension(self, dimension, data, n_rows):

        strategy = (dimension.strategy or "CompositeStrategyNone").upper()

        child_keys = [
            child.key
            for child in dimension.children
            if child.key in data
        ]

        if not child_keys:
            return [0] * n_rows

        child_df = pd.DataFrame({
            key: data[key]
            for key in child_keys
        })

        child_df = child_df.apply(pd.to_numeric, errors="coerce").fillna(0)

        if strategy == "COMPOSITESTRATEGYNONE":
            return [None] * n_rows

        if strategy == "COMPOSITESTRATEGYSUM":
            return child_df.sum(axis=1)

        if strategy == "COMPOSITESTRATEGYAVG":
            return child_df.mean(axis=1)

        if strategy == "COMPOSITESTRATEGYMAX":
            return child_df.max(axis=1)

        if strategy == "COMPOSITESTRATEGYMIN":
            return child_df.min(axis=1)

        # fallback seguro
        return child_df.sum(axis=1)

    def _track_dimension(self, key: str, class_name: str | None):
        if self.profiler is None:
            return nullcontext()

        return self.profiler.track(
            stage="dimension",
            name=key,
            class_name=class_name or "",
        )

        
        
    def _compute_ratio_dimension(self, dimension, data, n_rows):
        numerator_keys = split_param_list(
            param(dimension, "numerator", "")
        )
        denominator_keys = split_param_list(
            param(dimension, "denominator", "")
        )

        scale = float(param(dimension, "scale") or 1.0)
        zero_division = float(param(dimension, "zero_division") or 0.0)

        missing = [
            key
            for key in numerator_keys + denominator_keys
            if key not in data
        ]

        if missing:
            raise ValueError(
                f"Missing columns for RatioDimension '{dimension.key}': "
                f"{', '.join(missing)}"
            )

        numerator = self._sum_data_keys(data, numerator_keys, n_rows)
        denominator = self._sum_data_keys(data, denominator_keys, n_rows)

        result = numerator / denominator.replace(0, pd.NA)
        result = (
            pd.to_numeric(result, errors="coerce")
            .fillna(zero_division)
        )

        return result * scale
        

        
        
    def _split_param_list(self, value: str) -> list[str]:
        return [
            item.strip()
            for item in str(value).split("|")
            if item.strip()
        ]


    def _sum_data_keys(self, data, keys, n_rows):
        if not keys:
            return pd.Series([0.0] * n_rows)

        frame = pd.DataFrame({
            key: data[key]
            for key in keys
        })

        return frame.apply(pd.to_numeric, errors="coerce").fillna(0).sum(axis=1)