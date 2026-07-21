from umutextstats.config.params import (
    param,
    percentage_param,
)
from umutextstats.dimensions.pattern import PatternDimension


class POSTaggingExpression(PatternDimension):
    def __init__(
        self,
        key: str,
        pattern: str,
        input_column: str = "tagged_pos",
        percentage: bool = True,
    ):
        super().__init__(
            key=key,
            pattern=pattern,
            input_column=input_column,
            percentage=percentage,
        )

    @classmethod
    def from_config(
        cls,
        dimension,
        input_column: str = "tagged_pos",
    ):
        return cls(
            key=dimension.key,
            pattern=param(dimension, "pattern", ""),
            input_column="tagged_pos",
            percentage=percentage_param(dimension),
        )

    def compute_result(
        self,
        df,
    ):
        result = super().compute_result(df)

        if self.percentage:
            result.metadata["normalization_unit"] = (
                "serialized_pos_characters"
            )

        return result