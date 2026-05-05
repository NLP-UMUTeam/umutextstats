# src/umutextstats/dimensions/error_mispelling_accents.py

from umutextstats.dimensions.base import BaseDimension
from umutextstats.dimensions.enclitics_personal_pronouns import remove_accents
from umutextstats.dimensions.word_count import WORD_REGEX
from umutextstats.utils.accent_map import load_accent_map


class ErrorMispellingAccentsDimension(BaseDimension):
    def __init__(
        self,
        key: str,
        input_column: str = "text_norm",
        language: str = "es",
        accent_map_path: str | None = None,
        percentage: bool = True,
    ):
        super().__init__(key=key, input_column=input_column)
        self.percentage = percentage
        self.accent_map = load_accent_map(language=language, path=accent_map_path)

    def compute(self, df):
        return (
            df[self.input_column]
            .fillna("")
            .astype(str)
            .apply(self._compute_text)
        )

    def _compute_text(self, text: str) -> float:
        words = [word.lower() for word in WORD_REGEX.findall(text)]

        if not words:
            return 0.0

        occurrences = sum(1 for word in words if self._is_accent_error(word))

        if not self.percentage:
            return occurrences

        return (100 * occurrences) / len(words)

    def _is_accent_error(self, word: str) -> bool:
        # Si ya tiene tilde, no lo contamos como error de tilde omitida.
        plain = remove_accents(word)

        if plain != word:
            return False

        return word in self.accent_map