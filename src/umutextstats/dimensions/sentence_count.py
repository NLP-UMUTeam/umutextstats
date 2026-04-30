from umutextstats.dimensions.base import BaseDimension


class SentenceCountDimension(BaseDimension):
    def compute(self, df):
        text = df[self.input_column].fillna("").astype(str).str.strip()

        sentence_count = text.str.count(r"[.!?]+")

        # Si hay texto pero no hay puntuación final, cuenta como una frase
        sentence_count = sentence_count.where(
            (text == "") | (sentence_count > 0),
            1,
        )

        return sentence_count