# src/umutextstats/dimensions/pattern.py

import regex as re

from umutextstats.dimensions.base import BaseDimension


class PatternDimension(BaseDimension):
    def __init__(
        self,
        key: str,
        pattern: str,
        input_column: str = "text_norm",
        percentage: bool = False,
    ):
        super().__init__(key=key, input_column=input_column)
        self.pattern = pattern
        self.percentage = percentage

        try:
            self.regex = re.compile(pattern, re.IGNORECASE)
        except re.error as exc:
            raise ValueError(
                f"Invalid regex in PatternDimension '{key}': {pattern!r}. "
                f"Regex error: {exc}"
            ) from exc

    def compute(self, df):
        texts = df[self.input_column].fillna("").astype(str)

        counts = texts.apply(lambda text: len(self.regex.findall(text)))

        if not self.percentage:
            return counts

        total = texts.str.len()

        result = (100 * counts / total.replace(0, 1)).astype(float)
        result[total == 0] = 0.0

        return result