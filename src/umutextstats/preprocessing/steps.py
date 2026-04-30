# src/umutextstats/preprocessing/steps.py

import re
from abc import ABC, abstractmethod
from html import unescape


# =========================
# Base class
# =========================

class NormalizationStep(ABC):
    def __init__(self, enabled: bool = True):
        self.enabled = enabled

    def __call__(self, text: str) -> str:
        if not self.enabled:
            return text
        return self.apply(text)

    @abstractmethod
    def apply(self, text: str) -> str:
        pass


# =========================
# Basic cleaning
# =========================

class ReplaceLineBreaksStep(NormalizationStep):
    _linebreak_re = re.compile(r"[\r\n]+")

    def apply(self, text: str) -> str:
        def repl(match):
            before = text[:match.start()].rstrip()

            if not before:
                return " "

            prev_char = before[-1]

            if prev_char in ".!?":
                return " "

            return ". "

        return self._linebreak_re.sub(repl, text)


class StripHtmlStep(NormalizationStep):
    _tag_re = re.compile(r"<[^>]+>")

    def apply(self, text: str) -> str:
        text = unescape(text)
        return self._tag_re.sub(" ", text)


class TrimStep(NormalizationStep):
    def apply(self, text: str) -> str:
        return text.strip()


# =========================
# Social media cleaning
# =========================

class RemoveUrlsStep(NormalizationStep):
    _url_re = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)

    def apply(self, text: str) -> str:
        return self._url_re.sub("", text)


class RemoveMentionsStep(NormalizationStep):
    _mention_re = re.compile(r"@\w+:?", re.UNICODE)

    def apply(self, text: str) -> str:
        return self._mention_re.sub("", text)


class RemoveTwitterMarkersStep(NormalizationStep):
    def apply(self, text: str) -> str:
        return text.replace("#", "").replace("@", "")

class RemoveRetweetMarkerStep(NormalizationStep):
    _rt_re = re.compile(r"^\s*RT\s+", re.IGNORECASE)

    def apply(self, text: str) -> str:
        return self._rt_re.sub("", text)


# =========================
# Text normalization
# =========================

class RemoveRepeatedCharsStep(NormalizationStep):
    _repeated_re = re.compile(r"(.)\1{2,}", re.UNICODE)

    def apply(self, text: str) -> str:
        return self._repeated_re.sub(r"\1", text)


class LowercaseStep(NormalizationStep):
    def apply(self, text: str) -> str:
        return text.lower().strip()


class CleanupSpacesStep(NormalizationStep):
    _multi_dot_re = re.compile(r"\.+")
    _multi_space_re = re.compile(r"\s+")
    _space_before_punctuation_re = re.compile(r"\s+([.,;:!?])")

    def apply(self, text: str) -> str:
        text = text.replace(" . ", ". ")
        text = self._space_before_punctuation_re.sub(r"\1", text)
        text = self._multi_dot_re.sub(".", text)
        text = self._multi_space_re.sub(" ", text)
        return text.strip()


# =========================
# Extensible / optional steps
# =========================

class ReplaceEmojisStep(NormalizationStep):
    def __init__(self, enabled=True):
        super().__init__(enabled)

        try:
            import emoji
            self.emoji = emoji
        except ImportError:
            self.enabled = False

    def apply(self, text: str) -> str:
        text = self.emoji.demojize(text, delimiters=(" /", "/ "))
        text = text.replace("//", "/ /")
        return text


class ExpandSmsStep(NormalizationStep):
    """
    Expande lenguaje tipo SMS: q -> que, xq -> porque
    """
    def __init__(self, replacements: dict[str, str], enabled=True):
        super().__init__(enabled)
        self.replacements = [
            (re.compile(rf"(?<!\w){re.escape(k)}(?!\w)", re.IGNORECASE), v)
            for k, v in replacements.items()
        ]

    def apply(self, text: str) -> str:
        for pattern, repl in self.replacements:
            text = pattern.sub(repl, text)
        return text