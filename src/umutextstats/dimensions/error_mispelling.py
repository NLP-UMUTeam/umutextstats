import regex as re

from dataclasses import dataclass
from pathlib import Path
from spylls.hunspell import Dictionary

from umutextstats.dimensions.base import BaseDimension
from umutextstats.text.tokenization import get_lexical_tokens

WORD_RE = re.compile(r"\p{L}+")


@dataclass
class WordMatch:
    text: str
    start_pos: int
    end_pos: int

    def group(self, index: int = 0) -> str:
        if index != 0:
            raise IndexError(index)
        return self.text

    def start(self) -> int:
        return self.start_pos

    def end(self) -> int:
        return self.end_pos


class ErrorMispellingDimension(BaseDimension):
    def __init__(
        self,
        key: str,
        input_column: str = "text",
        language: str = "es_ES",
        dictionary_path: str = "/usr/share/hunspell/es_ES",
        missing_value: float | str = "",
    ):
        super().__init__(key=key, input_column=input_column)
        self.language = language
        self.dictionary_path = dictionary_path
        self.missing_value = missing_value
        self.dictionary = None
        self._known_cache: dict[str, bool] = {}

        aff_path = Path(f"{dictionary_path}.aff")
        dic_path = Path(f"{dictionary_path}.dic")

        if aff_path.exists() and dic_path.exists():
            self.dictionary = Dictionary.from_files(dictionary_path)

    def compute(self, df):
        if self.dictionary is None:
            return [self.missing_value] * len(df)

        return (
            df[self.input_column]
            .fillna("")
            .astype(str)
            .apply(self._compute_text)
        )

import regex as re

WORD_RE = re.compile(r"\p{L}+")


def iter_matches(self, text: str):
    text = "" if text is None else str(text)

    if self.dictionary is None:
        return

    for match in WORD_RE.finditer(text):
        word = match.group(0)
        word_norm = word.lower()

        if not self._should_check_word(word, word_norm):
            continue

        if not self._is_known(word_norm):
            yield WordMatch(
                text=word,
                start_pos=match.start(),
                end_pos=match.end(),
            )

    def count_matches(self, text: str) -> int:
        return sum(1 for _ in self.iter_matches(text))

    def _compute_text(self, text: str) -> float:
        text = "" if text is None else str(text)
        words = get_lexical_tokens(text, lowercase=False)

        checked = 0
        errors = 0

        for word in words:
            word_norm = word.lower()

            if not self._should_check_word(word, word_norm):
                continue

            checked += 1

            if not self._is_known(word_norm):
                errors += 1

        if checked == 0:
            return 0.0

        return (100 * errors) / checked

    def _should_check_word(self, word: str, word_norm: str) -> bool:
        if not word:
            return False

        if len(word_norm) <= 1:
            return False

        if not word.isalpha():
            return False

        if word.isupper() and len(word) > 1:
            return False

        if any(c.islower() for c in word) and any(c.isupper() for c in word[1:]):
            return False

        return True

    def _is_known(self, word_norm: str) -> bool:
        if word_norm not in self._known_cache:
            self._known_cache[word_norm] = self.dictionary.lookup(word_norm)

        return self._known_cache[word_norm]