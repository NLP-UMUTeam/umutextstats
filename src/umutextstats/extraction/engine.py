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

    Atomic dimensions are computed from the input DataFrame.

    Derived dimensions are computed from values produced by previously
    evaluated dimensions.

    Composite dimensions are calculated from their configured child
    dimensions.
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
            self._iter_dimensions(
                self.config.dimensions
            )
        )

        iterator = tqdm(
            dimensions,
            desc="Dimensions",
            unit="dimension",
            disable=not self.show_progress,
        )

        for dimension in iterator:
            iterator.set_postfix_str(
                dimension.key
            )

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
            reference_lengths=(
                self._build_reference_lengths(
                    df=df,
                    results=results,
                )
            ),
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
        Compute one atomic, derived, or composite dimension.
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
                results[key] = (
                    self._compute_composite_dimension(
                        dimension=dimension,
                        results=results,
                        n_rows=len(df),
                        index=df.index,
                    )
                )
                return

            instance = self._build_instance(
                dimension
            )

            if instance is None:
                self._handle_unimplemented_dimension(
                    key=key,
                    results=results,
                    n_rows=len(df),
                    index=df.index,
                )
                return

            if hasattr(
                instance,
                "compute_result_from_data",
            ):
                computation = (
                    instance.compute_result_from_data(
                        data=self._plain_values(
                            results
                        ),
                        n_rows=len(df),
                    )
                )

                kind = "derived"

            elif hasattr(
                instance,
                "compute_from_data",
            ):
                computation_values = (
                    instance.compute_from_data(
                        data=self._plain_values(
                            results
                        ),
                        n_rows=len(df),
                    )
                )

                computation = None
                kind = "derived"

            else:
                computation = instance.compute_result(
                    df
                )

                kind = "atomic"

            metadata = self._dimension_metadata(
                dimension=dimension,
                instance=instance,
            )

            if computation is not None:
                computation_values = (
                    computation.values
                )

                numerators = (
                    computation.numerators
                )

                denominators = (
                    computation.denominators
                )

                evidence = computation.evidence

                evidence_descriptor = (
                    computation.evidence_descriptor
                )

                computation_metadata = (
                    computation.metadata
                )

            else:
                numerators = None
                denominators = None
                evidence = None
                computation_metadata = {}

            metadata.update(
                computation_metadata
            )

            evidence_descriptor = (
                self._resolve_evidence_descriptor(
                    instance=instance,
                    explicit_descriptor=evidence_descriptor,
                    evidence=evidence,
                )
            )

            results[key] = BatchDimensionResult(
                key=key,
                values=self._ensure_series(
                    values=computation_values,
                    index=df.index,
                ),
                numerators=self._optional_series(
                    values=numerators,
                    index=df.index,
                ),
                denominators=self._optional_series(
                    values=denominators,
                    index=df.index,
                ),
                evidence=self._optional_series(
                    values=evidence,
                    index=df.index,
                ),
                evidence_descriptor=(
                    evidence_descriptor
                ),
                kind=kind,
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

        computation = instance.compute_result_from_data(
            data=self._plain_values(results),
            n_rows=n_rows,
        )

        metadata = self._dimension_metadata(
            dimension=dimension,
            instance=instance,
        )

        metadata.update(
            computation.metadata
        )

        evidence_descriptor = (
            self._resolve_evidence_descriptor(
                instance=instance,
                explicit_descriptor=(
                    computation.evidence_descriptor
                ),
                evidence=computation.evidence,
            )
        )

        return BatchDimensionResult(
            key=dimension.key,
            values=self._ensure_series(
                values=computation.values,
                index=index,
            ),
            numerators=self._optional_series(
                values=computation.numerators,
                index=index,
            ),
            denominators=self._optional_series(
                values=computation.denominators,
                index=index,
            ),
            evidence=self._optional_series(
                values=computation.evidence,
                index=index,
            ),
            evidence_descriptor=(
                evidence_descriptor
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
            default_input_column=(
                self.input_column
            ),
        )

    @staticmethod
    def _build_reference_lengths(
        df: pd.DataFrame,
        results: dict[
            str,
            BatchDimensionResult,
        ],
    ) -> dict[str, pd.Series]:
        """
        Compute lengths for representations referenced by evidence.

        Only sources explicitly declared through EvidenceDescriptor are
        included.
        """
        sources = {
            result.evidence_descriptor.source
            for result in results.values()
            if result.evidence_descriptor is not None
        }

        missing_sources = sorted(
            source
            for source in sources
            if source not in df.columns
        )

        if missing_sources:
            raise ValueError(
                "Evidence descriptor sources are not "
                "available in the extraction DataFrame: "
                + ", ".join(
                    repr(source)
                    for source in missing_sources
                )
            )

        return {
            source: df[source].apply(
                ExtractionEngine._reference_length
            )
            for source in sorted(sources)
        }

    @staticmethod
    def _reference_length(
        value,
    ) -> int:
        """
        Return the length of one evidence reference representation.
        """
        if value is None:
            return 0

        try:
            if pd.isna(value):
                return 0
        except (
            TypeError,
            ValueError,
        ):
            pass

        return len(
            str(value)
        )

    @staticmethod
    def _plain_values(
        results: dict[
            str,
            BatchDimensionResult,
        ],
    ) -> dict[str, pd.Series]:
        """
        Return value Series expected by derived and composite dimensions.
        """
        return {
            key: result.values
            for key, result in results.items()
        }

    @staticmethod
    def _resolve_evidence_descriptor(
        *,
        instance,
        explicit_descriptor,
        evidence,
    ):
        """
        Resolve the semantic descriptor of a dimension's evidence.

        An explicit descriptor returned by DimensionComputation always takes
        precedence. A class-level default is used only when the computation
        actually returns an evidence batch.
        """
        if explicit_descriptor is not None:
            return explicit_descriptor

        if evidence is None:
            return None

        descriptor_factory = getattr(
            instance,
            "evidence_descriptor",
            None,
        )

        if not callable(descriptor_factory):
            return None

        return descriptor_factory()

    def _optional_series(
        self,
        values,
        index: pd.Index,
    ) -> pd.Series | None:
        """
        Normalize an optional dimension output as an aligned Series.
        """
        if values is None:
            return None

        return self._ensure_series(
            values=values,
            index=index,
        )

    @staticmethod
    def _ensure_series(
        values,
        index: pd.Index,
    ) -> pd.Series:
        """
        Normalize dimension output as a Series aligned with the input.
        """
        if isinstance(
            values,
            pd.Series,
        ):
            result = values.copy()

            if len(result) != len(index):
                raise ValueError(
                    "Dimension returned a Series "
                    "with an unexpected length: "
                    f"expected {len(index)}, "
                    f"got {len(result)}"
                )

            result.index = index
            return result

        result = pd.Series(
            values,
            index=index,
        )

        if len(result) != len(index):
            raise ValueError(
                "Dimension returned an output "
                "with an unexpected length: "
                f"expected {len(index)}, "
                f"got {len(result)}"
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

    @staticmethod
    def _dimension_metadata(
        dimension: DimensionConfig,
        *,
        instance=None,
    ) -> dict:
        """
        Return configuration and runtime metadata for a dimension.
        """
        metadata = {
            "class_name": (
                instance.__class__.__name__
                if instance is not None
                else normalize_class_name(
                    dimension.class_name
                )
            ),
        }

        optional_values = {
            "description": dimension.description,
            "strategy": dimension.strategy,
            "dictionary": dimension.dictionary,
            "pattern": dimension.pattern,
            "universal": dimension.universal,
            "input_column": (
                dimension.input_column
            ),
            "pos_tag": dimension.pos_tag,
            "pos_input_column": (
                dimension.pos_input_column
            ),
            "percentage": (
                dimension.percentage
            ),
            "disabled_regexp": (
                dimension.disabled_regexp
            ),
        }

        metadata.update(
            {
                key: value
                for key, value
                in optional_values.items()
                if value is not None
            }
        )

        if dimension.children:
            metadata["children"] = [
                child.key
                for child
                in dimension.children
            ]

        if dimension.params:
            metadata["params"] = dict(
                dimension.params
            )

        return metadata