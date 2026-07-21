# src/umutextstats/extraction/engine.py

from __future__ import annotations

from contextlib import nullcontext

import pandas as pd
from tqdm.auto import tqdm

from umutextstats.config.models import (
    DimensionConfig,
    UMUTextStatsConfig,
)
from umutextstats.dimensions.composite import CompositeDimension
from umutextstats.dimensions.factory import build_dimension_instance
from umutextstats.dimensions.registry import (
    normalize_class_name,
    resolve_dimension,
)
from umutextstats.extraction.models import (
    BatchDimensionResult,
    ExtractionResult,
)


class ExtractionEngine:
    """
    Compute configured dimensions as structured batch results.

    Atomic dimensions keep using their existing `compute(df)` methods.
    Composite dimensions are calculated from the values of previously
    computed child results.
    """

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

    def compute(
        self,
        df: pd.DataFrame,
    ) -> ExtractionResult:
        """
        Compute all configured dimensions.
        """
        results: dict[str, BatchDimensionResult] = {}

        dimensions = list(
            self._iter_dimensions(self.config.dimensions)
        )

        iterator = tqdm(
            dimensions,
            desc="Dimensions",
            unit="dimension",
            disable=not self.show_progress,
        )

        for dimension in iterator:
            iterator.set_postfix_str(dimension.key)

            self._compute_dimension(
                df=df,
                dimension=dimension,
                results=results,
            )

        ids = None

        if "id" in df.columns:
            ids = df["id"].copy()

        return ExtractionResult(
            ids=ids,
            dimensions=results,
        )

    def _iter_dimensions(
        self,
        dimensions: list[DimensionConfig],
    ):
        """
        Yield dimensions recursively, including child dimensions.
        """
        for dimension in dimensions:
            yield dimension

            if dimension.children:
                yield from self._iter_dimensions(
                    dimension.children
                )

    def _compute_dimension(
        self,
        df: pd.DataFrame,
        dimension: DimensionConfig,
        results: dict[str, BatchDimensionResult],
    ) -> None:
        """
        Compute one atomic or composite dimension.
        """
        key = dimension.key

        if key in results:
            return

        class_name = normalize_class_name(
            dimension.class_name
        )

        with self._track_dimension(
            key=key,
            class_name=class_name,
        ):
            self._compute_children(
                df=df,
                dimension=dimension,
                results=results,
            )

            if dimension.children:
                results[key] = self._compute_composite_dimension(
                    dimension=dimension,
                    results=results,
                    n_rows=len(df),
                    index=df.index,
                )
                return

            instance = self._build_instance(dimension)

            if instance is None:
                self._handle_unimplemented_dimension(
                    key=key,
                    results=results,
                    n_rows=len(df),
                    index=df.index,
                )
                return

            if hasattr(instance, "compute_from_data"):
                computation_values = instance.compute_from_data(
                    data=self._plain_values(results),
                    n_rows=len(df),
                )

                numerators = None
                denominators = None
                evidence = None
                computation_metadata = {}

            else:
                computation = instance.compute_result(df)

                computation_values = computation.values
                numerators = computation.numerators
                denominators = computation.denominators
                evidence = computation.evidence
                computation_metadata = computation.metadata

            metadata = self._dimension_metadata(
                dimension=dimension,
                instance=instance,
            )

            metadata.update(computation_metadata)

            results[key] = BatchDimensionResult(
                key=key,
                values=self._ensure_series(
                    values=computation_values,
                    index=df.index,
                ),
                numerators=(
                    self._ensure_series(
                        values=numerators,
                        index=df.index,
                    )
                    if numerators is not None
                    else None
                ),
                denominators=(
                    self._ensure_series(
                        values=denominators,
                        index=df.index,
                    )
                    if denominators is not None
                    else None
                ),
                evidence=(
                    self._ensure_series(
                        values=evidence,
                        index=df.index,
                    )
                    if evidence is not None
                    else None
                ),
                kind="atomic",
                metadata=metadata,
            )

    def _compute_children(
        self,
        df: pd.DataFrame,
        dimension: DimensionConfig,
        results: dict[str, BatchDimensionResult],
    ) -> None:
        """
        Compute child dimensions before their parent.
        """
        for child in dimension.children:
            self._compute_dimension(
                df=df,
                dimension=child,
                results=results,
            )

    def _compute_composite_dimension(
        self,
        dimension: DimensionConfig,
        results: dict[str, BatchDimensionResult],
        n_rows: int,
        index: pd.Index,
    ) -> BatchDimensionResult:
        """
        Compute a composite dimension from child result values.
        """
        instance = CompositeDimension.from_config(
            dimension=dimension,
            input_column=self.input_column,
        )

        values = instance.compute_from_data(
            data=self._plain_values(results),
            n_rows=n_rows,
        )

        used_children = [
            key
            for key in instance.children
            if key in results
        ]

        missing_children = [
            key
            for key in instance.children
            if key not in results
        ]

        metadata = self._dimension_metadata(
            dimension=dimension,
            instance=instance,
        )

        metadata.update(
            {
                "strategy": instance.strategy,
                "children": list(instance.children),
                "used_children": used_children,
                "missing_children": missing_children,
            }
        )

        return BatchDimensionResult(
            key=dimension.key,
            values=self._ensure_series(
                values=values,
                index=index,
            ),
            kind="composite",
            metadata=metadata,
        )

    def _handle_unimplemented_dimension(
        self,
        key: str,
        results: dict[str, BatchDimensionResult],
        n_rows: int,
        index: pd.Index,
    ) -> None:
        """
        Preserve unresolved dimensions when configured to do so.
        """
        if not self.include_unimplemented:
            return

        results[key] = BatchDimensionResult(
            key=key,
            values=pd.Series(
                [""] * n_rows,
                index=index,
            ),
            kind="unimplemented",
        )

    def _build_instance(
        self,
        dimension: DimensionConfig,
    ):
        """
        Build a configured dimension instance.
        """
        if not dimension.class_name:
            return None

        dimension_cls = resolve_dimension(
            dimension.class_name
        )

        if dimension_cls is None:
            return None

        return build_dimension_instance(
            dimension=dimension,
            dimension_cls=dimension_cls,
            default_input_column=self.input_column,
        )

    def _plain_values(
        self,
        results: dict[str, BatchDimensionResult],
    ) -> dict[str, pd.Series]:
        """
        Return the value Series expected by existing composite dimensions.
        """
        return {
            key: result.values
            for key, result in results.items()
        }

    def _ensure_series(
        self,
        values,
        index: pd.Index,
    ) -> pd.Series:
        """
        Normalize dimension output as a Series aligned with the input.
        """
        if isinstance(values, pd.Series):
            result = values.copy()

            if len(result) != len(index):
                raise ValueError(
                    "Dimension returned a Series with an unexpected "
                    f"length: expected {len(index)}, got {len(result)}"
                )

            result.index = index
            return result

        result = pd.Series(
            values,
            index=index,
        )

        if len(result) != len(index):
            raise ValueError(
                "Dimension returned an output with an unexpected "
                f"length: expected {len(index)}, got {len(result)}"
            )

        return result

    def _track_dimension(
        self,
        key: str,
        class_name: str | None,
    ):
        """
        Return the profiler context for one dimension.
        """
        if self.profiler is None:
            return nullcontext()

        return self.profiler.track(
            stage="dimension",
            name=key,
            class_name=class_name or "",
        )
    

    def _dimension_metadata(
        self,
        dimension: DimensionConfig,
        *,
        instance=None,
    ) -> dict:
        metadata = {
            "class_name": (
                instance.__class__.__name__
                if instance is not None
                else normalize_class_name(dimension.class_name)
            ),
        }

        optional_values = {
            "description": dimension.description,
            "strategy": dimension.strategy,
            "dictionary": dimension.dictionary,
            "pattern": dimension.pattern,
            "universal": dimension.universal,
            "input_column": dimension.input_column,
            "pos_tag": dimension.pos_tag,
            "pos_input_column": dimension.pos_input_column,
            "percentage": dimension.percentage,
            "disabled_regexp": dimension.disabled_regexp,
        }

        metadata.update(
            {
                key: value
                for key, value in optional_values.items()
                if value is not None
            }
        )

        if dimension.children:
            metadata["children"] = [
                child.key
                for child in dimension.children
            ]

        if dimension.params:
            metadata["params"] = dict(dimension.params)

        return metadata