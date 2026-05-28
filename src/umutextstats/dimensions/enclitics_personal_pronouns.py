import unicodedata
import regex as re

from umutextstats.dictionaries import DictionaryLoader
from umutextstats.dimensions.base import BaseDimension
from umutextstats.text.tokenization import get_lexical_tokens
from umutextstats.text.patterns import ENCLITIC_REGEX


def remove_accents(text: str) -> str:
    text = unicodedata.normalize("NFD", text)
    return "".join(ch for ch in text if unicodedata.category(ch) != "Mn")


class EncliticsPersonalPronounsDictionary(BaseDimension):
    def __init__(
        self,
        key: str,
        dictionary_name: str,
        input_column: str = "text_norm",
        dictionary_loader: DictionaryLoader | None = None,
    ):
        super().__init__(key=key, input_column=input_column)

        self.dictionary_name = dictionary_name
        self.dictionary_loader = dictionary_loader or DictionaryLoader()

        entries = self.dictionary_loader.load(dictionary_name)
        self.verbs = [remove_accents(v.lower()) for v in entries.words]

        self.patterns = [
            re.compile(
                rf"(?<!\p{{L}}){re.escape(verb)}{ENCLITIC_REGEX}(?!\p{{L}})",
                re.IGNORECASE,
            )
            for verb in self.verbs
        ]

    def compute(self, df):
        texts = df[self.input_column].fillna("").astype(str)

        counts = texts.apply(self._count_text)
        total_words = texts.apply(lambda text: len(get_lexical_tokens(text)))

        result = (100 * counts / total_words.replace(0, 1)).astype(float)
        result[total_words == 0] = 0.0

        return result

    def _count_text(self, text: str) -> int:
        if not text:
            return 0

        text = remove_accents(text.lower())

        return sum(len(pattern.findall(text)) for pattern in self.patterns)