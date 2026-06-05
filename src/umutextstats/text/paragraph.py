from umutextstats.text.patterns import LEXICAL_TOKEN_REGEX, PARAGRAPH_SEPARATOR_REGEX


def split_paragraphs(text: str) -> list[str]:
    """Split text into non-empty paragraphs separated by one or more blank lines."""
    text = text.strip()

    if not text:
        return []

    return [
        paragraph.strip()
        for paragraph in PARAGRAPH_SEPARATOR_REGEX.split(text)
        if paragraph.strip()
    ]


def count_paragraph_words(paragraph: str) -> int:
    """Count lexical tokens in a paragraph."""
    return len(LEXICAL_TOKEN_REGEX.findall(paragraph))


def paragraph_lengths(text: str) -> list[int]:
    """Return paragraph lengths measured in lexical tokens."""
    return [count_paragraph_words(paragraph) for paragraph in split_paragraphs(text)]


def iter_paragraph_spans(text: str):
    text = "" if text is None else str(text)
    start = 0

    for separator in PARAGRAPH_SEPARATOR_REGEX.finditer(text):
        end = separator.start()
        paragraph = text[start:end]

        stripped = paragraph.strip()
        if stripped:
            leading = len(paragraph) - len(paragraph.lstrip())
            trailing = len(paragraph) - len(paragraph.rstrip())

            yield stripped, start + leading, end - trailing

        start = separator.end()

    paragraph = text[start:]
    stripped = paragraph.strip()

    if stripped:
        leading = len(paragraph) - len(paragraph.lstrip())
        trailing = len(paragraph) - len(paragraph.rstrip())

        yield stripped, start + leading, len(text) - trailing