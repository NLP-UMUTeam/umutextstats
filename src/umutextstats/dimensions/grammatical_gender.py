from __future__ import annotations

import numpy as np
import pandas as pd

from umutextstats.config.params import (
    dictionary_param,
    disabled_regexp_param,
    param,
    percentage_param,
)
from umutextstats.dimensions.results import DimensionComputation
from umutextstats.dimensions.word_per_dictionary import (
    WordPerDictionary,
)
from umutextstats.text.patterns import POS_ITEM_REGEX


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
    """
    Count dictionary matches among words with allowed POS tags.

    Matches are located in the original input text so extraction evidence
    preserves real character offsets. The denominator contains only POS
    items whose tag belongs to `ALLOWED_POS`.
    """

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
            pos_tag=sorted(ALLOWED_POS),
            pos_input_column=tagged_pos_column,
            percentage=percentage,
            use_regex=use_regex,
            dictionary_loader=dictionary_loader,
        )

        self.tagged_pos_column = tagged_pos_column

    @classmethod
    def from_config(
        cls,
        dimension,
        input_column: str = "text_norm",
    ):
        """
        Build the dimension from configuration.
        """
        return cls(
            key=dimension.key,
            dictionary_name=dictionary_param(dimension),
            input_column=input_column,
            tagged_pos_column=param(
                dimension,
                "tagged_pos_column",
                "tagged_pos",
            ),
            percentage=percentage_param(dimension),
            use_regex=not disabled_regexp_param(dimension),
        )

    def compute(
        self,
        df: pd.DataFrame,
    ) -> pd.Series:
        return self.compute_result(df).values

    def compute_single(
        self,
        row: pd.Series,
    ) -> float | int:
        """
        Compute the grammatical-gender score for one row.
        """
        text = self.get_text(row)

        tagged_pos = self.get_text(
            row=row,
            column=self.tagged_pos_column,
        )

        matching_text = self._get_matching_text(
            text=text,
            tagged_pos=tagged_pos,
        )

        matches = self.get_accepted_matches(
            text=matching_text,
            tagged_pos=tagged_pos,
        )

        count = len(matches)

        if not self.percentage:
            return count

        total_words = len(
            self._get_words_filtered_by_pos(
                tagged_pos
            )
        )

        if total_words == 0:
            return 0.0

        return (100.0 * count) / total_words


    def compute_result(
        self,
        df: pd.DataFrame,
    ) -> DimensionComputation:
        """
        Compute values and accepted match evidence.
        """
        texts = self.get_text_series(df)

        tagged_texts = self.get_text_series(
            df=df,
            column=self.tagged_pos_column,
        )

        matching_texts = pd.Series(
            (
                self._get_matching_text(
                    text=text,
                    tagged_pos=tagged_pos,
                )
                for text, tagged_pos in zip(
                    texts,
                    tagged_texts,
                )
            ),
            index=df.index,
            dtype=object,
        )

        accepted_matches = pd.Series(
            (
                self.get_accepted_matches(
                    text=matching_text,
                    tagged_pos=tagged_pos,
                )
                for matching_text, tagged_pos in zip(
                    matching_texts,
                    tagged_texts,
                )
            ),
            index=df.index,
            dtype=object,
        )

        counts = accepted_matches.apply(len)

        evidence = accepted_matches.apply(
            self._matches_to_evidence
        )

        denominators = tagged_texts.apply(
            lambda tagged_pos: len(
                self._get_words_filtered_by_pos(
                    tagged_pos
                )
            )
        )

        if not self.percentage:
            return DimensionComputation(
                values=counts,
                numerators=counts,
                evidence=evidence,
                metadata={
                    "measure": "count",
                    "unit": "matches",
                    "pos_filter": sorted(ALLOWED_POS),
                },
            )

        counts_array = counts.to_numpy(dtype=float)
        denominators_array = denominators.to_numpy(dtype=float)

        percentages = np.zeros_like(
            counts_array,
            dtype=float,
        )

        np.divide(
            100.0 * counts_array,
            denominators_array,
            out=percentages,
            where=denominators_array != 0,
        )

        return DimensionComputation(
            values=pd.Series(
                percentages,
                index=df.index,
            ),
            numerators=counts,
            denominators=denominators,
            evidence=evidence,
            metadata={
                "measure": "rate",
                "normalization_unit": "allowed_pos_words",
                "scale": 100.0,
                "pos_filter": sorted(ALLOWED_POS),
            },
        )


    def inspect(
        self,
        row: pd.Series,
    ):
        """
        Inspect the same accepted matches used by compute and extract.
        """
        text = self.get_text(row)

        tagged_pos = self.get_text(
            row=row,
            column=self.tagged_pos_column,
        )

        matching_text = self._get_matching_text(
            text=text,
            tagged_pos=tagged_pos,
        )

        accepted_matches = self.get_accepted_matches(
            text=matching_text,
            tagged_pos=tagged_pos,
        )

        matches = [
            self._to_inspect_match(match)
            for match in accepted_matches
        ]

        return self._build_inspection(
            matches=matches,
            discarded_matches=[],
        )

    def _get_words_filtered_by_pos(
        self,
        tagged_text: str,
    ) -> list[str]:
        """
        Extract lowercase words whose POS tag is allowed.
        """
        if not tagged_text:
            return []

        words = []

        for sentence in tagged_text.split(" || "):
            for raw_item in sentence.split(", "):
                match = POS_ITEM_REGEX.fullmatch(
                    raw_item.strip()
                )

                if not match:
                    continue

                word = match.group("word") or ""
                tag = match.group("tag") or ""

                if tag in ALLOWED_POS:
                    words.append(
                        word.lower()
                    )

        return words

    def inspection_debug_text(
        self,
    ) -> str:
        """
        Return configuration details used during inspection.
        """
        return (
            f"Loaded dictionary: {self.dictionary_name}\n"
            f"Allowed POS: {', '.join(sorted(ALLOWED_POS))}\n"
            f"Tagged POS column: {self.tagged_pos_column}\n"
            f"Use regex: {self.use_regex}\n"
            f"Percentage: {self.percentage}"
        )
    
    def _get_matching_text(
        self,
        text: str,
        tagged_pos: str,
    ) -> str:
        """
        Return the text used to locate dictionary matches.

        Prefer the configured input text when available. Otherwise,
        reconstruct a filtered text from the POS annotation.
        """
        if text:
            return text

        return " ".join(
            self._get_words_filtered_by_pos(tagged_pos)
        )