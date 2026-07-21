from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import regex as re

from umutextstats.config.params import param
from umutextstats.inspection.iterable_inspectable_dimension import IterableInspectableDimension
from umutextstats.dimensions.results import DimensionComputation
from umutextstats.inspection.models import DimensionInspection
from umutextstats.text.patterns import POS_ITEM_REGEX
from umutextstats.text.tokenization import get_lexical_tokens


VERB_FORM_BY_MODE = {
    "infinitive": "VerbForm=Inf",
    "gerund": "VerbForm=Ger",
    "participe": "VerbForm=Part",
    "participle": "VerbForm=Part",
}


@dataclass(frozen=True)
class PeriphrasisMatch:
    """
    Regex-like match object used by the inspection layer.
    """

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


class PeriphrasisDimension(IterableInspectableDimension):
    """
    Count verbal periphrases based on lexical tokens and POS annotations.

    This dimension needs two columns:
    - `input_column`: text used to extract lexical tokens.
    - `tagged_pos_column`: POS-tagged text used to validate verb forms.
    """

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

    @classmethod
    def from_config(
        cls,
        dimension,
        input_column: str = "text_norm",
    ):
        return cls(
            key=dimension.key,
            auxiliar_verbs=param(dimension, "auxiliar_verbs", ""),
            input_column=input_column,
            tagged_pos_column=param(
                dimension,
                "tagged_pos_column",
                "tagged_pos",
            ),
        )

    def compute_single(
        self,
        row: pd.Series,
    ) -> int:
        """
        Compute the number of verbal periphrases for a single row.
        """
        return len(self._analyze_row(row))

    def compute(
        self,
        df: pd.DataFrame,
    ) -> pd.Series:
        """
        Compute the number of verbal periphrases for all rows.
        """
        return df.apply(self._compute_row, axis=1)

    def compute_result(
        self,
        df: pd.DataFrame,
    ) -> DimensionComputation:
        """
        Compute periphrasis counts and structured evidence.
        """
        analyses = df.apply(
            self._analyze_row,
            axis=1,
        )

        evidence = analyses.apply(
            lambda matches: [
                self._to_evidence(match)
                for match in matches
            ]
        )

        numerators = evidence.apply(len)

        return DimensionComputation(
            values=numerators.astype(float),
            numerators=numerators,
            evidence=evidence,
            metadata={
                "measure": "count",
                "unit": "verbal_periphrases",
                "evidence_position_unit": "lexical_token_indices",
            },
        )

    @staticmethod
    def _to_evidence(
        match: PeriphrasisMatch,
    ) -> dict:
        """
        Convert a detected periphrasis to structured evidence.

        Positions are lexical-token indices, not character offsets.
        """
        return {
            "text": match.text,
            "token_start": match.start_pos,
            "token_end": match.end_pos,
        }

    def inspect(
        self,
        row: pd.Series,
    ) -> DimensionInspection:
        """
        Inspect the detected periphrases for a single row.
        """
        matches = [
            self._to_inspect_match(match)
            for match in self._analyze_row(row)
        ]

        return DimensionInspection(
            key=self.key,
            class_name=self.__class__.__name__,
            pattern=None,
            dictionary=None,
            matches=matches,
            discarded_matches=[],
            debug_text=self._build_debug_text(
                value=len(matches),
            ),
        )


    def iter_matches(
        self,
        text: str,
    ):
        """
        Return no matches when only plain text is available.

        Correct inspection requires POS annotations, so `inspect()`
        is implemented directly instead.
        """
        return []

    def _compute_row(
        self,
        row: pd.Series,
    ) -> int:
        """
        Compute the dimension from a DataFrame row.
        """
        return len(self._analyze_row(row))

    def _compute_text(
        self,
        text: str,
        tagged_pos: str,
    ) -> int:
        """
        Count periphrases in a text using its POS annotations.
        """
        return sum(
            1
            for _ in self._iter_periphrases(
                text=text,
                tagged_pos=tagged_pos,
            )
        )

    def _iter_periphrases(
        self,
        text: str,
        tagged_pos: str,
    ):
        """
        Yield verbal periphrases detected in text.
        """
        words = get_lexical_tokens(text)
        tagged_words = self._parse_tagged_pos(tagged_pos)

        if not words or not tagged_words:
            return

        index = 0

        while index < len(words):
            matched = False

            for aux in self.auxiliar_verbs:
                if words[index] not in aux["forms"]:
                    continue

                next_index = index + 1

                if aux["linker_variants"]:
                    matched_linker = self._match_linker(
                        words=words,
                        start_index=next_index,
                        linker_variants=aux["linker_variants"],
                    )

                    if matched_linker is None:
                        continue

                    next_index += matched_linker

                if next_index >= len(words):
                    continue

                if next_index >= len(tagged_words):
                    continue

                if not self._matches_verb_mode(
                    tagged_words[next_index],
                    aux["mode"],
                ):
                    continue

                yield PeriphrasisMatch(
                    text=" ".join(words[index: next_index + 1]),
                    start_pos=index,
                    end_pos=next_index + 1,
                )

                index = next_index
                matched = True
                break

            index += 1

    def _parse_auxiliar_verbs(
        self,
        raw: str,
    ) -> list[dict]:
        """
        Parse auxiliary verb configuration.

        Example item:

            estar+gerund
            ir (a)+infinitive
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
        """
        Match optional linker tokens after an auxiliary verb.
        """
        for linker_tokens in linker_variants:
            end_index = start_index + len(linker_tokens)

            if tuple(words[start_index:end_index]) == tuple(linker_tokens):
                return len(linker_tokens)

        return None

    def _parse_tagged_pos(
        self,
        tagged_text: str,
    ) -> list[dict[str, str]]:
        """
        Parse POS-tagged text into a list of token dictionaries.
        """
        if not tagged_text:
            return []

        items = []

        for sentence in tagged_text.split(" || "):
            for raw_item in sentence.split(", "):
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

    def _matches_verb_mode(
        self,
        item: dict[str, str],
        mode: str,
    ) -> bool:
        """
        Check whether a POS item matches the expected verb form.
        """
        if item["tag"] not in {"VERB", "AUX"}:
            return False

        expected = VERB_FORM_BY_MODE.get(mode)

        if not expected:
            return False

        return expected in item["feats"]

    def inspection_debug_text(self) -> str:
        """
        Return configuration details used during inspection.
        """
        return (
            f"Auxiliary patterns: {len(self.auxiliar_verbs)}\n"
            f"Tagged POS column: {self.tagged_pos_column}"
        )
    
    def _analyze_row(
        self,
        row: pd.Series,
    ) -> list[PeriphrasisMatch]:
        """
        Return all periphrases detected for a row.
        """
        text = self.get_text(row)

        tagged_pos = self.get_text(
            row=row,
            column=self.tagged_pos_column,
        )

        return list(
            self._iter_periphrases(
                text=text,
                tagged_pos=tagged_pos,
            )
        )