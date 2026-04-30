import re

from umutextstats.dimensions.base import BaseDimension


VOWELS = "aeiouáéíóúü"


def count_syllables_word(word: str) -> int:
    word = word.lower().strip()

    if not word:
        return 0

    groups = re.findall(f"[{VOWELS}]+", word)

    return max(1, len(groups))


def count_syllables_text(text: str) -> int:
    words = re.findall(r"\b[a-záéíóúüñ]+\b", text.lower())

    return sum(count_syllables_word(word) for word in words)


class SyllableCountDimension(BaseDimension):
    def compute(self, df):
        return (
            df[self.input_column]
            .fillna("")
            .astype(str)
            .apply(count_syllables_text)
        )