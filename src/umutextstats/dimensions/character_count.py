from umutextstats.dimensions.base import BaseDimension


class CharacterCountDimension(BaseDimension):
    def __init__(
        self,
        key: str,
        chars: str,
        input_column: str = "text_norm",
    ):
        super().__init__(key=key, input_column=input_column)
        self.chars = set(chars or "")

    def compute(self, df):
        texts = df[self.input_column].fillna("").astype(str)

        counts = texts.apply(self._count_chars)
        total_length = texts.str.len()

        result = (100 * counts / total_length.replace(0, 1)).astype(float)
        result[total_length == 0] = 0.0

        return result

    def _count_chars(self, text: str) -> int:
        if not text or not self.chars:
            return 0

        return sum(text.count(char) for char in self.chars)