import regex as re

from umutextstats.text.patterns import POS_ITEM_REGEX
from umutextstats.dimensions.base import BaseDimension
from umutextstats.text.tokenization import get_lexical_tokens


VERB_FORM_BY_MODE = {
    "infinitive": "VerbForm=Inf",
    "gerund": "VerbForm=Ger",
    "participe": "VerbForm=Part",
    "participle": "VerbForm=Part",
}


class PeriphrasisDimension(BaseDimension):
    def __init__(
        self,
        key: str,
        auxiliar_verbs: str,
        input_column: str = "text_norm",
        tagged_pos_column: str = "tagged_pos",
    ):
        super().__init__(key=key, input_column=input_column)
        self.tagged_pos_column = tagged_pos_column
        self.auxiliar_verbs = self._parse_auxiliar_verbs(auxiliar_verbs)

    def compute(self, df):
        return df.apply(self._compute_row, axis=1)

    def _compute_row(self, row) -> int:
        text = str(row.get(self.input_column, "") or "")
        tagged_pos = str(row.get(self.tagged_pos_column, "") or "")

        words = get_lexical_tokens (text)
        tagged_words = self._parse_tagged_pos(tagged_pos)

        if not words or not tagged_words:
            return 0

        occurrences = 0
        index = 0

        while index < len(words):
            matched = False

            for aux in self.auxiliar_verbs:
                if words[index] not in aux["forms"]:
                    continue

                next_index = index + 1

                if aux["linker_variants"]:
                    matched_linker = self._match_linker(
                        words,
                        next_index,
                        aux["linker_variants"],
                    )

                    if matched_linker is None:
                        continue

                    next_index += matched_linker

                if next_index >= len(words):
                    continue

                if next_index < len(tagged_words):
                    if self._matches_verb_mode(tagged_words[next_index], aux["mode"]):
                        occurrences += 1
                        index = next_index
                        matched = True
                        break

            index += 1

        return occurrences

    def _parse_auxiliar_verbs(self, raw: str) -> list[dict]:
        """
        Formato esperado:
            estar(por|para|a punto de)+infinitive
            tener(que)+infinitive
            estar+gerund
        """
        auxiliaries = []

        if not raw:
            return auxiliaries

        for part in raw.split(","):
            part = part.strip()

            if not part or "+" not in part:
                continue

            left, mode = part.split("+", 1)
            mode = mode.strip()

            if mode not in VERB_FORM_BY_MODE:
                continue

            linker_variants = []

            linker_match = re.search(r"\((.*?)\)", left)
            if linker_match:
                linker_variants = [
                    linker.strip().lower().split()
                    for linker in linker_match.group(1).split("|")
                    if linker.strip()
                ]

            verb = re.sub(r"\([^)]+\)", "", left).strip().lower()

            forms = {
                verb,
                f"{verb}me",
                f"{verb}te",
                f"{verb}se",
                f"{verb}nos",
                f"{verb}on",
            }

            auxiliaries.append(
                {
                    "verb": verb,
                    "forms": forms,
                    "linker_variants": linker_variants,
                    "mode": mode,
                }
            )

        return auxiliaries

    def _match_linker(
        self,
        words,
        start_index: int,
        linker_variants: list[list[str]],
    ) -> int | None:
        for linker_tokens in linker_variants:
            end_index = start_index + len(linker_tokens)

            if tuple(words[start_index:end_index]) == tuple(linker_tokens):
                return len(linker_tokens)

        return None

    def _parse_tagged_pos(self, tagged_text: str) -> list[dict[str, str]]:
        if not tagged_text:
            return []

        items = []

        for raw_item in tagged_text.split(", "):
            match = POS_ITEM_REGEX.fullmatch(raw_item.strip())

            if not match:
                continue

            items.append(
                {
                    "word": match.group("word") or "",
                    "tag": match.group("tag") or "",
                    "feats": match.group("feats") or "",
                }
            )

        return items

    def _matches_verb_mode(self, item: dict[str, str], mode: str) -> bool:
        if item["tag"] not in {"VERB", "AUX"}:
            return False

        expected = VERB_FORM_BY_MODE.get(mode)

        if not expected:
            return False

        return expected in item["feats"]