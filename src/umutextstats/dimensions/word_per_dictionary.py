from __future__ import annotations

from collections import Counter
from collections.abc import Iterable

import numpy as np
import pandas as pd
import regex as re

from umutextstats.config.params import (
    dictionary_param,
    disabled_regexp_param,
    param,
    percentage_param,
)
from umutextstats.dictionaries import DictionaryLoader
from umutextstats.dimensions.results import DimensionComputation
from umutextstats.inspection.iterable_inspectable_dimension import (
    IterableInspectableDimension,
)
from umutextstats.text.patterns import (
    LEXICAL_TOKEN_REGEX,
    POS_ITEM_REGEX,
)
from umutextstats.text.tokenization import get_lexical_tokens


class WordPerDictionary(IterableInspectableDimension):
    """
    Count words matching one or more configured dictionaries.

    Positive dictionary matches are filtered using dictionary exceptions
    and, optionally, POS annotations. The accepted matches are the source
    of truth for:

    - the final count;
    - the percentage value;
    - inspection output;
    - structured extraction evidence.
    """

    def __init__(
        self,
        key: str,
        dictionary_name: str,
        input_column: str = "text_norm",
        pos_tag: str | list[str] | None = None,
        pos_input_column: str | None = "tagged_pos",
        percentage: bool = True,
        use_regex: bool = True,
        dictionary_loader: DictionaryLoader | None = None,
    ):
        super().__init__(
            key=key,
            input_column=input_column,
        )

        self.dictionary_name = dictionary_name
        self.dictionary = dictionary_name
        self.percentage = percentage
        self.use_regex = use_regex
        self.pos_input_column = pos_input_column
        self.dictionary_loader = (
            dictionary_loader
            or DictionaryLoader()
        )

        if isinstance(pos_tag, str):
            pos_tag = [pos_tag]

        self.pos_tag = pos_tag or None

        dictionary_names = [
            name.strip()
            for name in dictionary_name.split("|")
            if name.strip()
        ]

        entries: list[str] = []
        exceptions: list[str] = []

        for name in dictionary_names:
            dictionary_entries = (
                self.dictionary_loader.load(name)
            )

            entries.extend(
                dictionary_entries.words
            )
            exceptions.extend(
                dictionary_entries.exceptions
            )

        self.entries = entries
        self.exceptions = exceptions

        # Compile dictionary patterns once at initialization time.
        if self.use_regex:
            self.patterns = self._compile_patterns(
                self.entries,
                kind="word",
            )

            self.exception_patterns = (
                self._compile_patterns(
                    self.exceptions,
                    kind="exception",
                )
            )

            self.words = None
            self.exception_words = None

        else:
            self.patterns = None
            self.exception_patterns = None

            self.words = {
                word.lower()
                for word in self.entries
            }

            self.exception_words = {
                word.lower()
                for word in self.exceptions
            }

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
            dictionary_name=dictionary_param(
                dimension
            ),
            input_column=input_column,
            percentage=percentage_param(
                dimension
            ),
            use_regex=not disabled_regexp_param(
                dimension
            ),
            pos_tag=param(
                dimension,
                "pos_tag",
                None,
            ),
            pos_input_column=param(
                dimension,
                "pos_input_column",
                "tagged_pos",
            ),
        )

    def compute_single(
        self,
        row: pd.Series,
    ) -> float | int:
        """
        Compute the dictionary count or percentage for one row.
        """
        text = self.get_text(row)

        tagged_pos = (
            self._get_tagged_pos(row)
            if self.pos_tag
            else None
        )

        accepted_matches = self.get_accepted_matches(
            text=text,
            tagged_pos=tagged_pos,
        )

        count = len(accepted_matches)

        if not self.percentage:
            return count

        word_total = self._get_word_total(
            row=row,
            text=text,
        )

        if word_total == 0:
            return 0.0

        return (100.0 * count) / word_total

    def compute(
        self,
        df: pd.DataFrame,
    ) -> pd.Series:
        """
        Compute dictionary values for all rows.

        The legacy flat engine receives only the final values. The
        structured extraction engine uses `compute_result()` directly.
        """
        return self.compute_result(df).values

    def compute_result(
        self,
        df: pd.DataFrame,
    ) -> DimensionComputation:
        """
        Compute values, numerators, denominators and accepted evidence.

        Dictionary matching is performed once per document. The accepted
        matches are then reused to derive the count and serializable
        evidence.
        """
        texts = self.get_text_series(df)

        if self.pos_tag:
            tagged_texts = self.get_text_series(
                df=df,
                column=(
                    self.pos_input_column
                    or "tagged_pos"
                ),
            )

            accepted_matches = pd.Series(
                (
                    self.get_accepted_matches(
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

        else:
            accepted_matches = texts.apply(
                lambda text: (
                    self.get_accepted_matches(
                        text=text,
                    )
                )
            )

        counts = accepted_matches.apply(len)

        evidence = accepted_matches.apply(
            self._matches_to_evidence
        )

        if not self.percentage:
            return DimensionComputation(
                values=counts,
                numerators=counts,
                evidence=evidence,
                metadata={
                    "measure": "count",
                    "unit": "matches",
                },
            )

        word_totals = self._get_word_totals(
            df=df,
            texts=texts,
        )

        counts_array = counts.to_numpy(
            dtype=float
        )

        word_totals_array = (
            word_totals.to_numpy(
                dtype=float
            )
        )

        percentages = np.zeros_like(
            counts_array,
            dtype=float,
        )

        np.divide(
            100.0 * counts_array,
            word_totals_array,
            out=percentages,
            where=word_totals_array != 0,
        )

        values = pd.Series(
            percentages,
            index=df.index,
        )

        return DimensionComputation(
            values=values,
            numerators=counts,
            denominators=word_totals,
            evidence=evidence,
            metadata={
                "measure": "rate",
                "normalization_unit": "words",
                "scale": 100.0,
            },
        )

    def inspect(
        self,
        row: pd.Series,
    ):
        """
        Inspect accepted dictionary matches for one row.
        """
        text = self.get_text(row)

        tagged_pos = (
            self._get_tagged_pos(row)
            if self.pos_tag
            else None
        )

        accepted_matches = self.get_accepted_matches(
            text=text,
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

    def iter_matches(
        self,
        text: str,
    ):
        """
        Yield accepted matches when no POS context is required.

        POS-filtered matching requires the tagged POS column and is
        therefore handled through `get_accepted_matches()`.
        """
        yield from self.get_accepted_matches(
            text=text,
        )

    def iter_discarded_matches(
        self,
        text: str,
    ):
        """
        Yield dictionary exception matches.
        """
        yield from self.iter_exception_matches(
            text
        )

    def get_accepted_matches(
        self,
        text: str,
        tagged_pos: str | None = None,
    ) -> list:
        """
        Return positive dictionary matches surviving all filters.

        Processing order:

        1. collect positive candidates;
        2. collect exception spans;
        3. discard positive candidates covered by an exception;
        4. apply the optional POS filter.
        """
        positive_matches = list(
            self.iter_positive_matches(text)
        )

        exception_spans = [
            match.span()
            for match in self.iter_exception_matches(
                text
            )
        ]

        accepted_matches = [
            match
            for match in positive_matches
            if not self._is_covered_by_exception(
                match=match,
                exception_spans=exception_spans,
            )
        ]

        if not self.pos_tag:
            return accepted_matches

        if not tagged_pos:
            return []

        return self._filter_matches_by_pos(
            matches=accepted_matches,
            tagged_pos=tagged_pos,
        )

    @staticmethod
    def _is_covered_by_exception(
        match,
        exception_spans: list[
            tuple[int, int]
        ],
    ) -> bool:
        """
        Return whether a positive match is contained in an exception.

        Containment is used instead of arbitrary overlap so that an
        exception only removes positive evidence located inside it.
        """
        match_start, match_end = (
            match.span()
        )

        return any(
            exception_start <= match_start
            and match_end <= exception_end
            for (
                exception_start,
                exception_end,
            ) in exception_spans
        )

    def _filter_matches_by_pos(
        self,
        matches: Iterable,
        tagged_pos: str,
    ) -> list:
        """
        Retain matches supported by the configured POS annotations.

        The current tagged POS representation contains words and tags but
        no source-text offsets. Valid occurrences are therefore assigned
        to textual matches in occurrence order.
        """
        available = self._allowed_pos_counter(
            tagged_pos
        )

        accepted = []

        for match in matches:
            word = match.group(0).lower()

            if available[word] <= 0:
                continue

            available[word] -= 1
            accepted.append(match)

        return accepted

    @staticmethod
    def _matches_to_evidence(
        matches: Iterable,
    ) -> list[dict]:
        """
        Convert accepted matches to JSON-serializable evidence.
        """
        return [
            {
                "text": match.group(0),
                "start": match.start(),
                "end": match.end(),
            }
            for match in matches
        ]

    def iter_positive_matches(
        self,
        text: str,
    ):
        """
        Yield raw positive dictionary candidates.

        Exceptions and POS filters are not applied here.
        """
        text = (
            ""
            if text is None
            else str(text)
        )

        if self.use_regex:
            for pattern in self.patterns:
                yield from pattern.finditer(text)

            return

        for match in LEXICAL_TOKEN_REGEX.finditer(
            text
        ):
            if (
                match.group(0).lower()
                in self.words
            ):
                yield match

    def iter_exception_matches(
        self,
        text: str,
    ):
        """
        Yield raw dictionary exception matches.
        """
        text = (
            ""
            if text is None
            else str(text)
        )

        if self.use_regex:
            for pattern in self.exception_patterns:
                yield from pattern.finditer(
                    text
                )

            return

        for match in LEXICAL_TOKEN_REGEX.finditer(
            text
        ):
            if (
                match.group(0).lower()
                in self.exception_words
            ):
                yield match

    def iter_positive_matches_with_pos(
        self,
        text: str,
        tagged_pos: str,
    ):
        """
        Yield raw positive matches supported by POS annotations.

        This method is retained for backwards compatibility. It does not
        apply dictionary exceptions.
        """
        if not self.pos_tag:
            yield from self.iter_positive_matches(
                text
            )
            return

        allowed = self._allowed_pos_counter(
            tagged_pos
        )

        for match in self.iter_positive_matches(
            text
        ):
            word = match.group(0).lower()

            if allowed[word] <= 0:
                continue

            allowed[word] -= 1
            yield match

    def iter_exception_matches_with_pos(
        self,
        text: str,
        tagged_pos: str,
    ):
        """
        Yield exception matches supported by POS annotations.
        """
        if not self.pos_tag:
            yield from self.iter_exception_matches(
                text
            )
            return

        allowed = self._allowed_pos_counter(
            tagged_pos
        )

        for match in self.iter_exception_matches(
            text
        ):
            word = match.group(0).lower()

            if allowed[word] <= 0:
                continue

            allowed[word] -= 1
            yield match

    def _compile_patterns(
        self,
        entries: list[str],
        kind: str,
    ):
        """
        Compile dictionary entries once.
        """
        patterns = []

        for line_number, entry in enumerate(
            entries,
            start=1,
        ):
            pattern = (
                rf"(?<!\p{{L}})"
                rf"{entry}"
                rf"(?!\p{{L}})"
            )

            try:
                patterns.append(
                    re.compile(
                        pattern,
                        re.IGNORECASE,
                    )
                )
            except re.error as exc:
                raise ValueError(
                    "Invalid regex in dictionary "
                    f"{self.dictionary_name!r} "
                    f"({kind}) at line "
                    f"{line_number}: {entry!r}. "
                    f"Compiled pattern: "
                    f"{pattern!r}. "
                    f"Regex error: {exc}"
                ) from exc

        return patterns

    def _allowed_pos_counter(
        self,
        tagged_pos: str,
    ) -> Counter:
        """
        Count word forms carrying one of the configured POS tags.
        """
        return Counter(
            item["word"].lower()
            for item in self._parse_tagged_pos(
                tagged_pos
            )
            if item["tag"] in self.pos_tag
        )

    def _parse_tagged_pos(
        self,
        tagged_pos: str,
    ) -> list[dict[str, str]]:
        """
        Parse the serialized POS annotation column.
        """
        if not tagged_pos:
            return []

        items = []

        for sentence in tagged_pos.split(
            " || "
        ):
            for raw_item in sentence.split(
                ", "
            ):
                match = POS_ITEM_REGEX.fullmatch(
                    raw_item.strip()
                )

                if not match:
                    continue

                items.append(
                    {
                        "word": (
                            match.group("word")
                            or ""
                        ),
                        "tag": (
                            match.group("tag")
                            or ""
                        ),
                        "feats": (
                            match.group("feats")
                            or ""
                        ),
                    }
                )

        return items

    def _get_tagged_pos(
        self,
        row: pd.Series,
    ) -> str:
        """
        Read the configured POS annotation column from one row.
        """
        return self.get_text(
            row=row,
            column=(
                self.pos_input_column
                or "tagged_pos"
            ),
        )

    def _get_word_total(
        self,
        row: pd.Series,
        text: str,
    ) -> int:
        """
        Get the denominator for one document.
        """
        word_count = self.get_value(
            row=row,
            column="word_count",
        )

        if word_count is not None:
            try:
                return int(word_count)
            except (
                TypeError,
                ValueError,
            ):
                pass

        return len(
            get_lexical_tokens(text)
        )

    def _get_word_totals(
        self,
        df: pd.DataFrame,
        texts: pd.Series,
    ) -> pd.Series:
        """
        Get word-count denominators for all documents.
        """
        if "word_count" in df.columns:
            return pd.to_numeric(
                df["word_count"],
                errors="coerce",
            ).fillna(0)

        return texts.apply(
            lambda text: len(
                get_lexical_tokens(text)
            )
        )

    def _build_inspection(
        self,
        matches,
        discarded_matches,
    ):
        """
        Build the standard inspection result.
        """
        from umutextstats.inspection.models import (
            DimensionInspection,
        )

        return DimensionInspection(
            key=self.key,
            class_name=(
                self.__class__.__name__
            ),
            pattern=None,
            dictionary=self.dictionary_name,
            matches=matches,
            discarded_matches=(
                discarded_matches
            ),
            debug_text=(
                self.inspection_debug_text()
            ),
        )

    def inspection_debug_text(
        self,
    ) -> str:
        """
        Return diagnostic information for inspection.
        """
        parts = [
            (
                "Loaded dictionary: "
                f"{self.dictionary_name}"
            ),
            (
                "Use regex: "
                f"{self.use_regex}"
            ),
            (
                "Percentage: "
                f"{self.percentage}"
            ),
        ]

        if self.pos_tag:
            parts.append(
                "POS filter: "
                + ", ".join(self.pos_tag)
            )

            parts.append(
                "POS input column: "
                f"{self.pos_input_column}"
            )

        return "\n".join(parts)
