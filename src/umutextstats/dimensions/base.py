from abc import ABC
from typing import Any

import pandas as pd

from umutextstats.dimensions.results import DimensionComputation

class BaseDimension(ABC):
    """
    Base class for all dimensions.

    Each dimension declares the main DataFrame column it consumes through
    `input_column`. For example: "text", "text_norm", "tagged_pos",
    or "tagged_ner".

    By default, dimensions can compute values row by row through
    `compute_single()`. Dimensions with faster pandas implementations
    should override `compute()`.
    """

    def __init__(
        self,
        key: str,
        input_column: str = "text_norm",
    ):
        self.key = key
        self.input_column = input_column

    def compute(
        self,
        df: pd.DataFrame,
    ) -> pd.Series:
        """
        Compute the dimension for all rows in a DataFrame.

        Default implementation applies `compute_single()` row by row.
        Override this method for vectorized pandas implementations.
        """
        return df.apply(self.compute_single, axis=1)

    def compute_result(
        self,
        df: pd.DataFrame,
    ) -> DimensionComputation:
        """
        Compute the dimension and return its structured batch result.

        Subclasses may override this method to expose numerators,
        denominators, normalization metadata, or other calculation details
        without repeating the main computation.
        """
        return DimensionComputation(
            values=self.compute(df),
        )

    def compute_single(
        self,
        row: pd.Series,
    ) -> Any:
        """
        Compute the dimension for a single DataFrame row.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__}.compute_single() is not implemented"
        )

    def inspect(
        self,
        row: pd.Series,
    ):
        """
        Return an inspection object for a single DataFrame row.

        Subclasses may override this to explain matches, discarded matches,
        patterns, dictionaries, or any other debug information.
        """
        return None

    def get_text(
        self,
        row: pd.Series,
        column: str | None = None,
    ) -> str:
        """
        Retrieve a text value from a row.

        If `column` is not provided, the dimension's `input_column`
        is used.
        """
        return self.get_value_as_text(
            row=row,
            column=column or self.input_column,
        )

    def get_text_series(
        self,
        df: pd.DataFrame,
        column: str | None = None,
    ) -> pd.Series:
        """
        Retrieve a text column from a DataFrame as a clean string Series.

        If `column` is not provided, the dimension's `input_column`
        is used.
        """
        return self.get_series_as_text(
            df=df,
            column=column or self.input_column,
        )

    def get_value(
        self,
        row: pd.Series,
        column: str,
        default: Any = None,
    ) -> Any:
        """
        Safely retrieve a value from a row.

        Missing columns and missing values return `default`.
        """
        if column not in row:
            return default

        value = row[column]

        if pd.isna(value):
            return default

        return value

    def get_series(
        self,
        df: pd.DataFrame,
        column: str,
        default: Any = None,
    ) -> pd.Series:
        """
        Safely retrieve a column from a DataFrame.

        Missing columns return a Series filled with `default`.
        """
        if column not in df.columns:
            return pd.Series(default, index=df.index)

        return df[column]

    def get_value_as_text(
        self,
        row: pd.Series,
        column: str,
    ) -> str:
        """
        Retrieve a row value and convert it to text.

        Missing values are converted to an empty string.
        """
        value = self.get_value(
            row=row,
            column=column,
            default="",
        )

        return "" if value is None else str(value)

    def get_series_as_text(
        self,
        df: pd.DataFrame,
        column: str,
    ) -> pd.Series:
        """
        Retrieve a DataFrame column and convert it to a clean text Series.

        Missing values are replaced by empty strings.
        """
        return (
            self.get_series(
                df=df,
                column=column,
                default="",
            )
            .fillna("")
            .astype(str)
        )