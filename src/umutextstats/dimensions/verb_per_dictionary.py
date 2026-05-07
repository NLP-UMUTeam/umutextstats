from umutextstats.dictionaries import DictionaryLoader
from umutextstats.dimensions.base import BaseDimension
from umutextstats.text.tokenization import get_lexical_tokens

AUX_VERB_HABER = {
    "ha",
    "habe",
    "habed",
    "haber",
    "habido",
    "habiendo",
    "habremos",
    "habrá",
    "habrán",
    "habrás",
    "habré",
    "habréis",
    "habría",
    "habríais",
    "habríamos",
    "habrían",
    "habrías",
    "habéis",
    "había",
    "habíais",
    "habíamos",
    "habían",
    "habías",
    "han",
    "has",
    "hay",
    "haya",
    "hayamos",
    "hayan",
    "hayas",
    "hayáis",
    "he",
    "hemos",
    "hube",
    "hubiera",
    "hubierais",
    "hubieran",
    "hubieras",
    "hubiere",
    "hubiereis",
    "hubieren",
    "hubieres",
    "hubieron",
    "hubiese",
    "hubieseis",
    "hubiesen",
    "hubieses",
    "hubimos",
    "hubiste",
    "hubisteis",
    "hubiéramos",
    "hubiéremos",
    "hubiésemos",
    "hubo",
}


class VerbPerDictionary(BaseDimension):
    def __init__(
        self,
        key: str,
        dictionary_name: str,
        input_column: str = "text_norm",
        percentage: bool = True,
        dictionary_loader: DictionaryLoader | None = None,
    ):
        super().__init__(key=key, input_column=input_column)
        
        self.dictionary_name = dictionary_name
        self.percentage = percentage
        self.dictionary_loader = dictionary_loader or DictionaryLoader()

        dictionary_names = [
            name.strip()
            for name in dictionary_name.split("|")
            if name.strip()
        ]

        entries = []
        for name in dictionary_names:
            entries = self.dictionary_loader.load(name)


        # This class intentionally ignores regex mode.
        self.words = {entry.lower() for entry in entries.words}

    def compute(self, df):
        return (
            df[self.input_column]
            .fillna("")
            .astype(str)
            .apply(self._compute_text)
        )

    def _compute_text(self, text: str) -> float:
        words = get_lexical_tokens (text)
        total_words = len(words)

        if total_words == 0:
            return 0.0

        occurrences = 0
        aux_verb = ""

        for word in words:
            candidate = f"{aux_verb}{word}"

            if candidate in self.words:
                occurrences += 1

            if word in AUX_VERB_HABER:
                aux_verb = f"{word} "
            else:
                aux_verb = ""

        if not self.percentage:
            return occurrences

        return (100 * occurrences) / total_words