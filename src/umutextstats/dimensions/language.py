from __future__ import annotations

import pandas as pd

from umutextstats.config.params import param
from umutextstats.dimensions.mixins import TextComputeMixin
from umutextstats.dimensions.results import DimensionComputation
from umutextstats.inspection.scalar_inspectable_dimension import (
    ScalarInspectableDimension,
)
from umutextstats.utils.fasttext_loader import FastTextLoader


class LanguageDimension(
    TextComputeMixin,
    ScalarInspectableDimension,
):
    """
    Return the FastText probability assigned to a target language.

    The model is queried for its two most probable language labels. If the
    configured target language is not among those labels, the result is 0.0.
    """

    def __init__(
        self,
        key: str,
        language: str,
        input_column: str = "text_norm",
        missing_value: float | str = "",
    ):
        super().__init__(
            key=key,
            input_column=input_column,
        )

        self.language = language.lower()
        self.missing_value = missing_value
        self.model = FastTextLoader.get_model()

    @classmethod
    def from_config(
        cls,
        dimension,
        input_column: str = "text_norm",
    ):
        """
        Build the dimension from configuration.
        """
        return cls(
            key=dimension.key,
            language=param(
                dimension,
                "language",
                "",
            ),
            input_column=input_column,
        )

    def _compute_text(
        self,
        text: str,
    ) -> float:
        """
        Return the target-language probability for one text.
        """
        return self._predict_target_probability(
            text
        )

    def compute_result(
        self,
        df: pd.DataFrame,
    ) -> DimensionComputation:
        """
        Compute target-language probabilities with model metadata.
        """
        texts = self.get_text_series(df)

        values = texts.apply(
            self._predict_target_probability
        ).astype(float)

        return DimensionComputation(
            values=values,
            metadata={
                "measure": "language_probability",
                "unit": "probability",
                "target_language": self.language,
                "model": "fasttext",
                "prediction_k": 2,
                "fallback": (
                    "zero_when_target_not_in_top_k"
                ),
            },
        )

    def _predict_target_probability(
        self,
        text: str,
    ) -> float:
        """
        Predict the probability of the configured target language.
        """
        text = (
            ""
            if text is None
            else str(text)
        )

        if not text.strip():
            return 0.0

        labels, probabilities = self.model.predict(
            text,
            k=2,
        )

        for label, probability in zip(
            labels,
            probabilities,
        ):
            code = label.replace(
                "__label__",
                "",
            )

            if code.lower() == self.language:
                return float(probability)

        return 0.0

    def inspection_debug_text(
        self,
    ) -> str:
        """
        Return configuration details used during inspection.
        """
        return (
            f"Target language: {self.language}\n"
            "Model: FastText\n"
            "Prediction labels requested: 2"
        )