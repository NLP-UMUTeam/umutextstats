# src/umutextstats/text/sentence.py

from functools import lru_cache

import regex as re

from umutextstats.text.patterns import (
    ABBREVIATIONS,
    SENTENCE_END_REGEX,
    SENTENCE_SPAN_REGEX,
)


@lru_cache(maxsize=50_000)
def get_sentences(text: str) -> tuple[str, ...]:
    """
    Extract sentence-like text segments.

    Examples
    --------
    >>> get_sentences("Hola mundo. Adiós!")
    ("Hola mundo.", "Adiós!")
    """

    text = "" if text is None else str(text)

    sentences = [
        sentence.strip()
        for sentence in SENTENCE_SPAN_REGEX.findall(text)
        if sentence.strip()
    ]

    return tuple(sentences)


@lru_cache(maxsize=50_000)
def count_sentence_endings(text: str) -> int:
    """
    Count raw sentence-ending punctuation marks.

    Examples
    --------
    >>> count_sentence_endings("Hola. Adiós!")
    2
    """

    text = "" if text is None else str(text)

    return len(SENTENCE_END_REGEX.findall(text))


@lru_cache(maxsize=50_000)
def count_sentences(text: str) -> int:
    """
    Count linguistic sentences with basic heuristics.

    Rules
    -----
    - Empty text -> 0
    - Text without sentence-ending punctuation -> 1
    - Ignore decimal separators: 3.14
    - Ignore common abbreviations: Sr., Dr., etc.

    Examples
    --------
    >>> count_sentences("Hola mundo.")
    1

    >>> count_sentences("Hola mundo. Adiós.")
    2

    >>> count_sentences("Vale 3.14 euros.")
    1

    >>> count_sentences("El Sr. García vino.")
    1
    """

    text = "" if text is None else str(text)
    text = text.strip()

    if not text:
        return 0

    count = 0

    for match in SENTENCE_END_REGEX.finditer(text):
        start, end = match.span()
        punct = match.group()

        # Ignore decimal separators: 3.14
        if (
            punct == "."
            and start > 0
            and end < len(text)
            and text[start - 1].isdigit()
            and text[end].isdigit()
        ):
            continue

        # Ignore abbreviations: Sr., Dr., etc.
        prefix = text[:start].rstrip()

        last_token_match = re.search(
            r"[\p{L}.]+$",
            prefix,
            re.UNICODE,
        )

        if last_token_match:
            last_token = (
                last_token_match.group()
                .lower()
                .strip(".")
            )

            if last_token in ABBREVIATIONS:
                continue

        count += 1

    # Minimum one sentence for non-empty text
    return count if count > 0 else 1