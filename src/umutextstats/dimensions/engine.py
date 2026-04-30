from contextlib import nullcontext
from tqdm.auto import tqdm

import pandas as pd

from umutextstats.config.models import DimensionConfig, UMUTextStatsConfig
from umutextstats.dimensions.registry import resolve_dimension, normalize_class_name


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

            if dimension_cls is None:
                if self.include_unimplemented:
                    data[key] = [""] * len(df)
                return

            instance = self._build_dimension(dimension, dimension_cls)
            data[key] = instance.compute(df)

    def _compute_composite_dimension(
        self,
        dimension: DimensionConfig,
        data: dict,
        n_rows: int,
    ):
        strategy = (dimension.strategy or "SUM").upper()

        child_keys = [
            child.key
            for child in dimension.children
            if child.key in data
        ]

        if not child_keys:
            return [""] * n_rows

        child_df = pd.DataFrame({
            key: data[key]
            for key in child_keys
        })

        child_df = child_df.apply(pd.to_numeric, errors="coerce")

        if strategy == "SUM":
            return child_df.sum(axis=1, skipna=True)

        if strategy == "AVG":
            return child_df.mean(axis=1, skipna=True)

        if strategy == "MAX":
            return child_df.max(axis=1, skipna=True)

        if strategy == "MIN":
            return child_df.min(axis=1, skipna=True)

        # Unknown strategy
        return [""] * n_rows

    def _track_dimension(self, key: str, class_name: str | None):
        if self.profiler is None:
            return nullcontext()

        return self.profiler.track(
            stage="dimension",
            name=key,
            class_name=class_name or "",
        )

    def _build_dimension(self, dimension: DimensionConfig, dimension_cls):
        if normalize_class_name(dimension.class_name) == "WordPerDictionary":
            return dimension_cls(
                key=dimension.key,
                dictionary_name=dimension.dictionary or "",
                input_column=self.input_column,
                percentage=dimension.percentage,
                use_regex=not dimension.disabled_regexp,
            )
            
        if normalize_class_name(dimension.class_name) == "PatternDimension":
            return dimension_cls(
                key=dimension.key,
                pattern=dimension.pattern or "",
                input_column=self.input_column,
                percentage=dimension.percentage,
            )
            
        if normalize_class_name(dimension.class_name) == "POSTaggingTag":
            return dimension_cls(
                key=dimension.key,
                input_column="tagged_pos",
                postagger_tag=dimension.params.get("postagger_tag"),
                postagger_universal=dimension.universal,
            )

        return dimension_cls(
            key=dimension.key,
            input_column=self.input_column,
        )