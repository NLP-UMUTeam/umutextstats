from __future__ import annotations

import pandas as pd

from umutextstats.dimensions.base import BaseDimension
from umutextstats.dimensions.results import DimensionComputation


class CompositeDimension(BaseDimension):
    """
    Combine previously computed child dimensions using a configured strategy.

    Supported strategies:

    - CompositeStrategyNone
    - CompositeStrategySum
    - CompositeStrategyAvg
    - CompositeStrategyMax
    - CompositeStrategyMin
    """

    def __init__(
        self,
        key: str,
        children: list[str],
        strategy: str = "CompositeStrategyNone",
        input_column: str = "text_norm",
    ):
        super().__init__(
            key=key,
            input_column=input_column,
        )

        self.children = children
        self.strategy = (
            strategy
            or "CompositeStrategyNone"
        )

    @classmethod
    def from_config(
        cls,
        dimension,
        input_column: str = "text_norm",
    ):
        """
        Build the composite dimension from configuration.
        """
        return cls(
            key=dimension.key,
            children=[
                child.key
                for child in dimension.children
            ],
            strategy=(
                dimension.strategy
                or "CompositeStrategyNone"
            ),
            input_column=input_column,
        )

    def compute(
        self,
        df: pd.DataFrame,
    ) -> pd.Series:
        """
        Compute the composite value from DataFrame columns.
        """
        return self.compute_from_data(
            data={
                column: df[column]
                for column in df.columns
            },
            n_rows=len(df),
        )

    def compute_result(
        self,
        df: pd.DataFrame,
    ) -> DimensionComputation:
        """
        Compute a structured composite result from DataFrame columns.
        """
        return self.compute_result_from_data(
            data={
                column: df[column]
                for column in df.columns
            },
            n_rows=len(df),
        )

    def compute_from_data(
        self,
        data: dict[str, pd.Series],
        n_rows: int,
    ) -> pd.Series:
        """
        Compute the composite value from previously calculated dimensions.
        """
        return self.compute_result_from_data(
            data=data,
            n_rows=n_rows,
        ).values

    def compute_result_from_data(
        self,
        data: dict[str, pd.Series],
        n_rows: int,
    ) -> DimensionComputation:
        """
        Compute the composite value and preserve aggregation metadata.
        """
        child_keys = self._available_children(
            data
        )

        missing_children = [
            key
            for key in self.children
            if key not in data
        ]

        metadata = {
            "measure": "composite",
            "strategy": self.strategy,
            "children": list(self.children),
            "used_children": child_keys,
            "missing_children": missing_children,
        }

        if not child_keys:
            return DimensionComputation(
                values=pd.Series(
                    [0.0] * n_rows,
                    dtype=float,
                ),
                metadata=metadata,
            )

        child_df = self._build_child_frame(
            data=data,
            child_keys=child_keys,
        )

        strategy = self.strategy.upper()

        if strategy == "COMPOSITESTRATEGYNONE":
            return DimensionComputation(
                values=pd.Series(
                    [None] * n_rows,
                    index=child_df.index,
                    dtype=object,
                ),
                metadata={
                    **metadata,
                    "aggregation": "none",
                },
            )

        if strategy == "COMPOSITESTRATEGYSUM":
            return DimensionComputation(
                values=child_df.sum(
                    axis=1
                ),
                metadata={
                    **metadata,
                    "aggregation": "sum",
                },
            )

        if strategy == "COMPOSITESTRATEGYAVG":
            numerators = child_df.sum(
                axis=1
            )

            denominators = pd.Series(
                [len(child_keys)] * n_rows,
                index=child_df.index,
                dtype=int,
            )

            values = (
                numerators
                / denominators.replace(
                    0,
                    pd.NA,
                )
            )

            values = (
                pd.to_numeric(
                    values,
                    errors="coerce",
                )
                .fillna(0.0)
                .astype(float)
            )

            return DimensionComputation(
                values=values,
                numerators=numerators,
                denominators=denominators,
                metadata={
                    **metadata,
                    "aggregation": "mean",
                    "numerator_unit": (
                        "sum_of_child_values"
                    ),
                    "normalization_unit": (
                        "used_child_dimensions"
                    ),
                },
            )

        if strategy == "COMPOSITESTRATEGYMAX":
            return DimensionComputation(
                values=child_df.max(
                    axis=1
                ),
                metadata={
                    **metadata,
                    "aggregation": "maximum",
                },
            )

        if strategy == "COMPOSITESTRATEGYMIN":
            return DimensionComputation(
                values=child_df.min(
                    axis=1
                ),
                metadata={
                    **metadata,
                    "aggregation": "minimum",
                },
            )

        return DimensionComputation(
            values=child_df.sum(
                axis=1
            ),
            metadata={
                **metadata,
                "aggregation": "sum",
                "fallback_strategy": True,
            },
        )

    def _available_children(
        self,
        data: dict[str, pd.Series],
    ) -> list[str]:
        """
        Return configured child keys present in the input data.
        """
        return [
            key
            for key in self.children
            if key in data
        ]

    @staticmethod
    def _build_child_frame(
        data: dict[str, pd.Series],
        child_keys: list[str],
    ) -> pd.DataFrame:
        """
        Build a numeric DataFrame containing available child values.
        """
        child_df = pd.DataFrame(
            {
                key: data[key]
                for key in child_keys
            }
        )

        return (
            child_df
            .apply(
                pd.to_numeric,
                errors="coerce",
            )
            .fillna(0)
        )