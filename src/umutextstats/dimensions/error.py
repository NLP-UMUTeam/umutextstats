from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from spylls.hunspell import Dictionary

from umutextstats.dimensions.enclitics_personal_pronouns import (
    remove_accents,
)
from umutextstats.dimensions.results import DimensionComputation
from umutextstats.inspection.iterable_inspectable_dimension import (
    IterableInspectableDimension,
)
from umutextstats.inspection.scalar_inspectable_dimension import (
    ScalarInspectableDimension,
)
from umutextstats.text.patterns import (
    INITIAL_TOKEN_EXCLUSING_NUMBERS_REGEX,
    INITIAL_TOKEN_REGEX,
    MENTION_REGEX,
    REPEATED_WORD_REGEX,
    SENTENCE_SPAN_REGEX,
    WORD_RE,
)
from umutextstats.text.sentence import get_sentences
from umutextstats.text.tokenization import get_lexical_tokens
from umutextstats.utils.accent_map import load_accent_map


AMBIGUOUS_DIACRITIC_WORDS = {
    "el",
    "tu",
    "mi",
    "si",
    "se",
    "te",
    "de",
    "mas",
    "aun",
    "solo",
}


@dataclass(frozen=True)
class SimpleMatch:
    """
    Regex-like match object used by the inspection layer.
    """

    text: str
    start_pos: int
    end_pos: int

    def group(
        self,
        index: int = 0,
    ) -> str:
        if index != 0:
            raise IndexError(index)

        return self.text

    def start(self) -> int:
        return self.start_pos

    def end(self) -> int:
        return self.end_pos


def matches_to_evidence(
    matches,
) -> list[dict]:
    """
    Convert regex-like matches into serializable evidence.
    """
    return [
        {
            "text": match.group(0),
            "start": match.start(),
            "end": match.end(),
        }
        for match in matches
    ]


class ErrorCapitalizationStartingWithLowerCaseDimension(
    IterableInspectableDimension
):
    """
    Compute the percentage of sentences beginning with a lowercase letter.
    """

    START_SYMBOLS = {
        "¿",
        "¡",
        "[",
        '"',
        "'",
        "-",
        "—",
        "_",
    }

    def __init__(
        self,
        key: str,
        input_column: str = "text_raw",
    ):
        super().__init__(
            key=key,
            input_column=input_column,
        )

    def compute_single(
        self,
        row: pd.Series,
    ) -> float:
        return self._compute_text(
            self.get_text(row)
        )

    def compute(
        self,
        df: pd.DataFrame,
    ) -> pd.Series:
        return self.get_text_series(df).apply(
            self._compute_text
        )

    def compute_result(
        self,
        df: pd.DataFrame,
    ) -> DimensionComputation:
        """
        Compute lowercase-start percentages and sentence evidence.
        """
        texts = self.get_text_series(df)

        analyses = texts.apply(
            self._analyze_text
        )

        numerators = analyses.apply(
            lambda analysis: len(analysis[1])
        )

        denominators = analyses.apply(
            lambda analysis: len(analysis[0])
        )

        values = pd.Series(
            [
                (
                    100.0 * numerator / denominator
                    if denominator
                    else 0.0
                )
                for numerator, denominator in zip(
                    numerators,
                    denominators,
                )
            ],
            index=df.index,
            dtype=float,
        )

        evidence = analyses.apply(
            lambda analysis: matches_to_evidence(
                analysis[1]
            )
        )

        return DimensionComputation(
            values=values,
            numerators=numerators,
            denominators=denominators,
            evidence=evidence,
            metadata={
                "measure": "rate",
                "numerator_unit": (
                    "sentences_starting_with_lowercase"
                ),
                "normalization_unit": "sentences",
                "scale": 100.0,
                "evidence_offset_unit": "text_characters",
            },
        )

    def iter_sentences(
        self,
        text: str,
    ):
        """
        Yield valid sentence spans.
        """
        text = (
            ""
            if text is None
            else str(text)
        )

        for match in SENTENCE_SPAN_REGEX.finditer(
            text
        ):
            raw_sentence = match.group(0)
            sentence = raw_sentence.strip()

            if not sentence:
                continue

            if not any(
                char.isalpha()
                for char in sentence
            ):
                continue

            leading_whitespace = (
                len(raw_sentence)
                - len(raw_sentence.lstrip())
            )

            trailing_whitespace = (
                len(raw_sentence)
                - len(raw_sentence.rstrip())
            )

            start = (
                match.start()
                + leading_whitespace
            )

            end = (
                match.end()
                - trailing_whitespace
            )

            yield SimpleMatch(
                text=sentence,
                start_pos=start,
                end_pos=end,
            )

    def iter_matches(
        self,
        text: str,
    ):
        """
        Yield sentences beginning with lowercase.
        """
        _, errors = self._analyze_text(text)

        yield from errors

    def _analyze_text(
        self,
        text: str,
    ) -> tuple[
        list[SimpleMatch],
        list[SimpleMatch],
    ]:
        """
        Return all valid sentences and lowercase-start errors.
        """
        sentences = list(
            self.iter_sentences(text)
        )

        errors = [
            sentence
            for sentence in sentences
            if self._starts_with_lowercase(
                sentence.group(0)
            )
        ]

        return sentences, errors

    def _compute_text(
        self,
        text: str,
    ) -> float:
        sentences, errors = self._analyze_text(
            text
        )

        if not sentences:
            return 0.0

        return (
            100.0
            * len(errors)
            / len(sentences)
        )

    def _starts_with_lowercase(
        self,
        sentence: str,
    ) -> bool:
        sentence = MENTION_REGEX.sub(
            "",
            sentence,
        ).strip()

        if not sentence:
            return False

        for char in sentence:
            if (
                char in self.START_SYMBOLS
                or char.isspace()
            ):
                continue

            if not char.isalpha():
                return False

            return (
                char == char.lower()
                and char != char.upper()
            )

        return False


class ErrorMispellingAccentsDimension(
    IterableInspectableDimension
):
    """
    Detect words that appear to be missing an accent mark.
    """

    def __init__(
        self,
        key: str,
        input_column: str = "text",
        language: str = "es",
        accent_map_path: str | None = None,
        percentage: bool = True,
    ):
        super().__init__(
            key=key,
            input_column=input_column,
        )

        self.percentage = percentage

        self.accent_map = load_accent_map(
            language=language,
            path=accent_map_path,
        )

    def compute_single(
        self,
        row: pd.Series,
    ) -> float:
        return self._compute_text(
            self.get_text(row)
        )

    def compute(
        self,
        df: pd.DataFrame,
    ) -> pd.Series:
        return self.get_text_series(df).apply(
            self._compute_text
        )

    def compute_result(
        self,
        df: pd.DataFrame,
    ) -> DimensionComputation:
        """
        Compute missing-accent errors and structured evidence.
        """
        texts = self.get_text_series(df)

        matches = texts.apply(
            lambda text: list(
                self.iter_matches(text)
            )
        )

        numerators = matches.apply(len)

        evidence = matches.apply(
            matches_to_evidence
        )

        if not self.percentage:
            return DimensionComputation(
                values=numerators.astype(float),
                numerators=numerators,
                evidence=evidence,
                metadata={
                    "measure": "count",
                    "unit": "missing_accent_errors",
                    "evidence_offset_unit": (
                        "text_characters"
                    ),
                },
            )

        denominators = texts.apply(
            lambda text: len(
                get_lexical_tokens(text)
            )
        )

        values = pd.Series(
            [
                (
                    100.0 * numerator / denominator
                    if denominator
                    else 0.0
                )
                for numerator, denominator in zip(
                    numerators,
                    denominators,
                )
            ],
            index=df.index,
            dtype=float,
        )

        return DimensionComputation(
            values=values,
            numerators=numerators,
            denominators=denominators,
            evidence=evidence,
            metadata={
                "measure": "rate",
                "numerator_unit": (
                    "missing_accent_errors"
                ),
                "normalization_unit": "lexical_tokens",
                "scale": 100.0,
                "evidence_offset_unit": "text_characters",
            },
        )

    def iter_matches(
        self,
        text: str,
    ):
        """
        Yield words detected as missing an accent.
        """
        text = (
            ""
            if text is None
            else str(text)
        )

        for match in WORD_RE.finditer(text):
            word = match.group(0)

            if self._is_accent_error(word):
                yield SimpleMatch(
                    text=word,
                    start_pos=match.start(),
                    end_pos=match.end(),
                )

    def _compute_text(
        self,
        text: str,
    ) -> float:
        matches = list(
            self.iter_matches(text)
        )

        if not self.percentage:
            return float(len(matches))

        total_words = len(
            get_lexical_tokens(text)
        )

        if total_words == 0:
            return 0.0

        return (
            100.0
            * len(matches)
            / total_words
        )

    def _is_accent_error(
        self,
        word: str,
    ) -> bool:
        if word in AMBIGUOUS_DIACRITIC_WORDS:
            return False

        plain = remove_accents(word)

        if plain != word:
            return False

        return word in self.accent_map


class ErrorMispellingDimension(
    IterableInspectableDimension
):
    """
    Compute the percentage of checked words rejected by Hunspell.
    """

    def __init__(
        self,
        key: str,
        input_column: str = "text",
        language: str = "es_ES",
        dictionary_path: str = "/usr/share/hunspell/es_ES",
        missing_value: float | str = "",
    ):
        super().__init__(
            key=key,
            input_column=input_column,
        )

        self.language = language
        self.dictionary_path = dictionary_path
        self.missing_value = missing_value
        self.dictionary = None

        self._known_cache: dict[str, bool] = {}

        aff_path = Path(
            f"{dictionary_path}.aff"
        )

        dic_path = Path(
            f"{dictionary_path}.dic"
        )

        if (
            aff_path.exists()
            and dic_path.exists()
        ):
            self.dictionary = Dictionary.from_files(
                dictionary_path
            )

    def compute_single(
        self,
        row: pd.Series,
    ) -> float | str:
        if self.dictionary is None:
            return self.missing_value

        return self._compute_text(
            self.get_text(row)
        )

    def compute(
        self,
        df: pd.DataFrame,
    ) -> pd.Series:
        if self.dictionary is None:
            return pd.Series(
                [self.missing_value] * len(df),
                index=df.index,
            )

        return self.get_text_series(df).apply(
            self._compute_text
        )

    def compute_result(
        self,
        df: pd.DataFrame,
    ) -> DimensionComputation:
        """
        Compute spelling-error percentages and rejected-word evidence.
        """
        if self.dictionary is None:
            values = pd.Series(
                [self.missing_value] * len(df),
                index=df.index,
            )

            return DimensionComputation(
                values=values,
                metadata={
                    "measure": "rate",
                    "numerator_unit": (
                        "spelling_errors"
                    ),
                    "normalization_unit": (
                        "checked_words"
                    ),
                    "scale": 100.0,
                    "available": False,
                    "reason": (
                        "hunspell_dictionary_unavailable"
                    ),
                    "dictionary_path": (
                        self.dictionary_path
                    ),
                },
            )

        texts = self.get_text_series(df)

        analyses = texts.apply(
            self._analyze_text
        )

        denominators = analyses.apply(
            lambda analysis: analysis[0]
        )

        error_matches = analyses.apply(
            lambda analysis: analysis[1]
        )

        numerators = error_matches.apply(len)

        values = pd.Series(
            [
                (
                    100.0 * numerator / denominator
                    if denominator
                    else 0.0
                )
                for numerator, denominator in zip(
                    numerators,
                    denominators,
                )
            ],
            index=df.index,
            dtype=float,
        )

        evidence = error_matches.apply(
            matches_to_evidence
        )

        return DimensionComputation(
            values=values,
            numerators=numerators,
            denominators=denominators,
            evidence=evidence,
            metadata={
                "measure": "rate",
                "numerator_unit": "spelling_errors",
                "normalization_unit": "checked_words",
                "scale": 100.0,
                "available": True,
                "dictionary_path": self.dictionary_path,
                "evidence_offset_unit": "text_characters",
            },
        )

    def iter_matches(
        self,
        text: str,
    ):
        """
        Yield words rejected by the loaded Hunspell dictionary.
        """
        if self.dictionary is None:
            return

        _, errors = self._analyze_text(text)

        yield from errors

    def _analyze_text(
        self,
        text: str,
    ) -> tuple[int, list[SimpleMatch]]:
        """
        Return the number of checked words and rejected-word matches.
        """
        text = (
            ""
            if text is None
            else str(text)
        )

        checked = 0
        errors = []

        for match in WORD_RE.finditer(text):
            word = match.group(0)
            word_norm = word.lower()

            if not self._should_check_word(
                word,
                word_norm,
            ):
                continue

            checked += 1

            if not self._is_known(word_norm):
                errors.append(
                    SimpleMatch(
                        text=word,
                        start_pos=match.start(),
                        end_pos=match.end(),
                    )
                )

        return checked, errors

    def _compute_text(
        self,
        text: str,
    ) -> float:
        checked, errors = self._analyze_text(
            text
        )

        if checked == 0:
            return 0.0

        return (
            100.0
            * len(errors)
            / checked
        )

    def _should_check_word(
        self,
        word: str,
        word_norm: str,
    ) -> bool:
        if not word:
            return False

        if len(word_norm) <= 1:
            return False

        if not word.isalpha():
            return False

        if (
            word.isupper()
            and len(word) > 1
        ):
            return False

        if (
            any(
                char.islower()
                for char in word
            )
            and any(
                char.isupper()
                for char in word[1:]
            )
        ):
            return False

        return True

    def _is_known(
        self,
        word_norm: str,
    ) -> bool:
        if word_norm not in self._known_cache:
            self._known_cache[word_norm] = (
                self.dictionary.lookup(
                    word_norm
                )
            )

        return self._known_cache[word_norm]


class ErrorMiscTwoOrMoreEqualWordsDimension(
    ScalarInspectableDimension
):
    """
    Count consecutive repeated-word occurrences.
    """

    def compute_single(
        self,
        row: pd.Series,
    ) -> int:
        return self._compute_text(
            self.get_text(row)
        )

    def compute(
        self,
        df: pd.DataFrame,
    ) -> pd.Series:
        return self.get_text_series(df).apply(
            self._compute_text
        )

    def compute_result(
        self,
        df: pd.DataFrame,
    ) -> DimensionComputation:
        """
        Compute repeated-word error counts.
        """
        values = self.compute(df)

        return DimensionComputation(
            values=values,
            numerators=values.copy(),
            metadata={
                "measure": "count",
                "unit": "repeated_word_errors",
            },
        )

    def _compute_text(
        self,
        text: str,
    ) -> int:
        sentences = get_sentences(text)

        if not sentences:
            return 0

        return sum(
            self._count_repeated_words(sentence)
            for sentence in sentences
        )

    def _count_repeated_words(
        self,
        sentence: str,
    ) -> int:
        return sum(
            1
            for _ in REPEATED_WORD_REGEX.finditer(
                sentence
            )
        )


class ErrorStyleSentencesStartingWithNumbers(
    IterableInspectableDimension
):
    """
    Compute the percentage of sentences beginning with a number.
    """

    def __init__(
        self,
        key: str,
        input_column: str = "text_raw",
    ):
        super().__init__(
            key=key,
            input_column=input_column,
        )

    def compute_single(
        self,
        row: pd.Series,
    ) -> float:
        return self._compute_text(
            self.get_text(row)
        )

    def compute(
        self,
        df: pd.DataFrame,
    ) -> pd.Series:
        return self.get_text_series(df).apply(
            self._compute_text
        )

    def compute_result(
        self,
        df: pd.DataFrame,
    ) -> DimensionComputation:
        """
        Compute number-start percentages and sentence evidence.
        """
        texts = self.get_text_series(df)

        analyses = texts.apply(
            self._analyze_text
        )

        numerators = analyses.apply(
            lambda analysis: len(analysis[1])
        )

        denominators = analyses.apply(
            lambda analysis: len(analysis[0])
        )

        values = pd.Series(
            [
                (
                    100.0 * numerator / denominator
                    if denominator
                    else 0.0
                )
                for numerator, denominator in zip(
                    numerators,
                    denominators,
                )
            ],
            index=df.index,
            dtype=float,
        )

        evidence = analyses.apply(
            lambda analysis: matches_to_evidence(
                analysis[1]
            )
        )

        return DimensionComputation(
            values=values,
            numerators=numerators,
            denominators=denominators,
            evidence=evidence,
            metadata={
                "measure": "rate",
                "numerator_unit": (
                    "sentences_starting_with_numbers"
                ),
                "normalization_unit": "sentences",
                "scale": 100.0,
                "evidence_offset_unit": "text_characters",
            },
        )

    def iter_sentences(
        self,
        text: str,
    ):
        """
        Yield valid sentence spans.
        """
        text = (
            ""
            if text is None
            else str(text)
        )

        for match in SENTENCE_SPAN_REGEX.finditer(
            text
        ):
            raw_sentence = match.group(0)
            sentence = raw_sentence.strip()

            if not sentence:
                continue

            if not any(
                char.isalnum()
                for char in sentence
            ):
                continue

            leading_whitespace = (
                len(raw_sentence)
                - len(raw_sentence.lstrip())
            )

            trailing_whitespace = (
                len(raw_sentence)
                - len(raw_sentence.rstrip())
            )

            start = (
                match.start()
                + leading_whitespace
            )

            end = (
                match.end()
                - trailing_whitespace
            )

            yield SimpleMatch(
                text=sentence,
                start_pos=start,
                end_pos=end,
            )

    def iter_matches(
        self,
        text: str,
    ):
        """
        Yield sentences beginning with a number.
        """
        _, errors = self._analyze_text(text)

        yield from errors

    def _analyze_text(
        self,
        text: str,
    ) -> tuple[
        list[SimpleMatch],
        list[SimpleMatch],
    ]:
        """
        Return all valid sentences and number-start errors.
        """
        sentences = list(
            self.iter_sentences(text)
        )

        errors = [
            sentence
            for sentence in sentences
            if self._starts_with_number(
                sentence.group(0)
            )
        ]

        return sentences, errors

    def _compute_text(
        self,
        text: str,
    ) -> float:
        sentences, errors = self._analyze_text(
            text
        )

        if not sentences:
            return 0.0

        return (
            100.0
            * len(errors)
            / len(sentences)
        )

    def _starts_with_number(
        self,
        sentence: str,
    ) -> bool:
        match = INITIAL_TOKEN_REGEX.search(
            sentence
        )

        if not match:
            return False

        return match.group(0)[0].isdigit()


class ErrorStyleSentencesStartingWithTheSameWord(
    ScalarInspectableDimension
):
    """
    Compute the percentage of sentences whose first word is repeated.
    """

    def __init__(
        self,
        key: str,
        input_column: str = "text_raw",
    ):
        super().__init__(
            key=key,
            input_column=input_column,
        )

    def compute_single(
        self,
        row: pd.Series,
    ) -> float:
        return self._compute_text(
            self.get_text(row)
        )

    def compute(
        self,
        df: pd.DataFrame,
    ) -> pd.Series:
        return self.get_text_series(df).apply(
            self._compute_text
        )

    def compute_result(
        self,
        df: pd.DataFrame,
    ) -> DimensionComputation:
        """
        Compute repeated-first-word percentages.
        """
        texts = self.get_text_series(df)

        analyses = texts.apply(
            self._analyze_text
        )

        numerators = analyses.apply(
            lambda analysis: analysis[1]
        )

        denominators = analyses.apply(
            lambda analysis: analysis[0]
        )

        values = pd.Series(
            [
                (
                    100.0 * numerator / denominator
                    if denominator
                    else 0.0
                )
                for numerator, denominator in zip(
                    numerators,
                    denominators,
                )
            ],
            index=df.index,
            dtype=float,
        )

        return DimensionComputation(
            values=values,
            numerators=numerators,
            denominators=denominators,
            metadata={
                "measure": "rate",
                "numerator_unit": (
                    "sentences_with_repeated_first_word"
                ),
                "normalization_unit": "sentences",
                "scale": 100.0,
            },
        )

    def _analyze_text(
        self,
        text: str,
    ) -> tuple[int, int]:
        """
        Return total sentences and repeated-first-word occurrences.
        """
        sentences = self._split_sentences(
            text
        )

        first_words = [
            first_word
            for sentence in sentences
            if (
                first_word
                := self._first_word(sentence)
            )
            is not None
        ]

        stats = Counter(first_words)

        occurrences = sum(
            count
            for count in stats.values()
            if count > 1
        )

        return len(sentences), occurrences

    def _compute_text(
        self,
        text: str,
    ) -> float:
        total_sentences, occurrences = (
            self._analyze_text(text)
        )

        if total_sentences == 0:
            return 0.0

        return (
            100.0
            * occurrences
            / total_sentences
        )

    def _split_sentences(
        self,
        text: str,
    ) -> list[str]:
        text = (
            ""
            if text is None
            else str(text)
        )

        sentences = []

        for match in SENTENCE_SPAN_REGEX.finditer(
            text
        ):
            sentence = match.group(0).strip()

            if not sentence:
                continue

            if not any(
                char.isalpha()
                for char in sentence
            ):
                continue

            sentences.append(sentence)

        return sentences

    def _first_word(
        self,
        sentence: str,
    ) -> str | None:
        match = (
            INITIAL_TOKEN_EXCLUSING_NUMBERS_REGEX.search(
                sentence
            )
        )

        if not match:
            return None

        return match.group(0).lower()