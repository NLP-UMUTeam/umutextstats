from functools import lru_cache

from umutextstats.text.patterns import LEXICAL_TOKEN_REGEX, SYLLABIFIABLE_WORD_REGEX


@lru_cache(maxsize=50_000)
def get_lexical_tokens(
    text: str,
    lowercase: bool = True,
) -> tuple[str, ...]:
    return tuple(
        token
        for token, _, _ in get_lexical_token_spans(
            text,
            lowercase=lowercase,
        )
    )


@lru_cache(maxsize=50_000)
def get_lexical_token_spans(
    text: str,
    lowercase: bool = True,
) -> tuple[tuple[str, int, int], ...]:
    """
    Return lexical tokens together with character offsets.

    Each item is:

        (token, start, end)

    Offsets refer to the original input string and follow the
    half-open interval convention: [start, end).
    """
    text = "" if text is None else str(text)

    items = []

    for match in LEXICAL_TOKEN_REGEX.finditer(
        text
    ):
        token = match.group(0)

        if lowercase:
            token = token.lower()

        items.append(
            (
                token,
                match.start(),
                match.end(),
            )
        )

    return tuple(items)


@lru_cache(maxsize=50_000)
def get_syllabifiable_words(text: str, lowercase: bool = True) -> tuple[str, ...]:
    text = "" if text is None else str(text)
    words = SYLLABIFIABLE_WORD_REGEX.findall(text)

    if lowercase:
        words = [word.lower() for word in words]

    return tuple(words)