import regex as re
 

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


SENTENCE_END_REGEX = re.compile(r"[.!?]+", re.UNICODE)
SENTENCE_REGEX = re.compile (r"[^.!?]+[.!?]*", re.UNICODE)

POS_ITEM_REGEX = re.compile(
    r"(?P<word>.+?)__\((?P<tag>[^)]*)\)(?:\((?P<feats>[^)]*)\))?"
)

ENCLITIC_PATTERN = r"([mts]e(l[aoe]s?)?|l[oae]s?|[mn]os?)"

ABBREVIATIONS = {
    "sr", "sra", "srta",
    "dr", "dra",
    "prof", "profa",
    "etc",
}