from umutextstats.text.patterns import POS_ITEM_REGEX
from umutextstats.dimensions.word_per_dictionary import WordPerDictionary


ALLOWED_POS = {
    "DET",
    "ADJ",
    "NOUN",
    "PROPN",
    "PRON",
    "VERB",
    "AUX",
}


class GrammaticalGenderDimension(WordPerDictionary):
    def __init__(
        self,
        key: str,
        dictionary_name: str,
        input_column: str = "text_norm",
        tagged_pos_column: str = "tagged_pos",
        percentage: bool = True,
        use_regex: bool = True,
        dictionary_loader=None,
    ):
        super().__init__(
            key=key,
            dictionary_name=dictionary_name,
            input_column=input_column,
            percentage=percentage,
            use_regex=use_regex,
            dictionary_loader=dictionary_loader,
        )
        self.tagged_pos_column = tagged_pos_column

    def compute(self, df):
        return df.apply(self._compute_row, axis=1)

    def _compute_row(self, row) -> float:
        tagged_pos = str(row.get(self.tagged_pos_column, "") or "")

        filtered_words = self._get_words_filtered_by_pos(tagged_pos)

        total_words = len(filtered_words)

        if total_words == 0:
            return 0.0

        filtered_text = " ".join(filtered_words)

        count = self._count_text(filtered_text)

        if not self.percentage:
            return count

        return (100 * count) / total_words

    def _get_words_filtered_by_pos(self, tagged_text: str) -> list[str]:
        words = []

        if not tagged_text:
            return words

        for raw_item in tagged_text.split(", "):
            match = POS_ITEM_REGEX.fullmatch(raw_item.strip())

            if not match:
                continue

            word = match.group("word") or ""
            tag = match.group("tag") or ""

            if tag in ALLOWED_POS:
                words.append(word.lower())

        return words