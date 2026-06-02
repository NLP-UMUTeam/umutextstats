import regex as re


"""
Abbreviations commonly found in Spanish text.

Used to avoid treating some periods as sentence boundaries,
for example: "Dr.", "Sra.", "etc.".
"""
ABBREVIATIONS = {
    "dr", "dra",
    "etc",
    "prof", "profa",
    "sr", "sra", "srta",
}


"""
Pattern for detecting Spanish enclitic pronouns attached to verbs.

Examples:
- dámelo
- explicárselo
- comernos
"""
ENCLITIC_REGEX = r"([mts]e(l[aoe]s?)?|l[oae]s?|[mn]os?)"


"""
Matches dependency parser items encoded as:

    word__(deprel)(head)

Example:
    casa__(obj)(comprar)
"""
DEPENDENCY_ITEM_REGEX = re.compile(
    r"(?P<word>.+?)__\((?P<deprel>[^)]*)\)\((?P<head>[^)]*)\)"
)


"""
Matches the first alphanumeric token in a text.

It allows numbers with one decimal separator, for example:
- 3
- 3,14
- 3.14
- palabra
"""
INITIAL_TOKEN_REGEX = re.compile(
    r"\b[\p{L}\p{N}]+(?:[.,]\d+)?\b",
    re.UNICODE,
)


"""
Matches the first alphabetic word in a text.

Unlike INITIAL_TOKEN_REGEX, this excludes numbers.
"""
INITIAL_TOKEN_EXCLUSING_NUMBERS_REGEX = re.compile(
    r"\b[\p{L}]+\b",
    re.UNICODE,
)


"""
Matches lexical tokens useful for linguistic analysis.

It captures:
- words with letters and/or numbers
- hyphenated words
- apostrophe forms
- decimal numbers with comma or period separators

Examples:
- casa
- siglo-XXI
- d'artagnan
- 3,14
- 1.000,50
"""
LEXICAL_TOKEN_REGEX = re.compile(
    r"""
    (?V1)
    (?:
        \b\p{N}+(?:[.,]\p{N}+)+\b
        |
        \b[\p{L}\p{N}]+(?:[-'’][\p{L}\p{N}]+)*\b
    )
    """,
    re.IGNORECASE | re.UNICODE | re.VERBOSE,
)


"""
Matches the first non-whitespace token in a text.

A token is defined as any contiguous sequence of non-space
characters, without applying linguistic tokenization rules.

Examples:
- hola
- @usuario
- https://example.com
- palabra!!!
"""
FIRST_TOKEN_REGEX = re.compile(r"\S+", re.UNICODE)



"""
Matches mentions in social-media-style text.

Examples:
- @usuario
- @UMUTextStats
"""
MENTION_REGEX = re.compile(
    r"@\w+",
    re.UNICODE,
)


"""
Matches a mention only when it appears at the beginning of the text.

Useful for removing initial replies in tweets or social media posts.

Example:
    "@usuario gracias por la ayuda"
"""
LEADING_MENTION_REGEX = re.compile(
    r"^\s*@\w+\s*",
    re.UNICODE,
)


"""
Matches named-entity recognition items encoded as:

    TAG(text)

Example:
    PER(María)
    LOC(Murcia)
"""
NER_ITEM_REGEX = re.compile(
    r"(?P<tag>[A-Z]+)\((?P<text>.*?)\)"
)


"""
Matches paragraph separators defined as one or more blank lines.
"""
PARAGRAPH_SEPARATOR_REGEX = re.compile(r"\n\s*\n+", re.UNICODE)


"""
Matches part-of-speech tagged items encoded as:

    word__(TAG)
    word__(TAG)(features)

Examples:
    casa__(NOUN)
    el__(DET)(PronType=Art)
"""
POS_ITEM_REGEX = re.compile(
    r"(?P<word>.+?)__\((?P<tag>[^)]*)\)(?:\((?P<feats>[^)]*)\))?"
)


"""
Matches sentence-ending punctuation.

Used to count or detect sentence boundaries.

Examples:
- .
- !
- ?
- ...
"""
SENTENCE_END_REGEX = re.compile(
    r"[.!?]+",
    re.UNICODE,
)


"""
Matches sentence-like spans.

It captures text up to optional sentence-ending punctuation.
"""
SENTENCE_SPAN_REGEX = re.compile(
    r"[^.!?]+[.!?]*",
    re.UNICODE,
)


"""
Matches words that can reasonably be passed to a syllabification algorithm.

Only alphabetic words are accepted, including hyphenated and apostrophe forms.
Numbers are excluded.

Examples:
- casa
- dámelo
- teórico-práctico
"""
SYLLABIFIABLE_WORD_REGEX = re.compile(
    r"""
    (?V1)
    \b
    \p{L}+
    (?:[-'’]\p{L}+)*
    \b
    """,
    re.IGNORECASE | re.UNICODE | re.VERBOSE,
)


"""
Matches URLs in text.

Examples:
- https://example.com
- http://example.com
- www.example.com
"""
URL_REGEX = re.compile(
    r"https?://\S+|www\.\S+",
    re.IGNORECASE,
)


"""
Matches simple word-like tokens, optionally preceded by @.

This is a looser tokenization pattern that can keep mentions as tokens.

Examples:
- palabra
- 123
- @usuario
"""
WORD_TOKEN_REGEX = re.compile(
    r"@?\b[\p{L}\p{N}]+\b",
    re.UNICODE,
)