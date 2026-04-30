# src/umutextstats/preprocessing/normalizer.py

from umutextstats.preprocessing.steps import (
    CleanupSpacesStep,
    ExpandSmsStep,
    LowercaseStep,
    NormalizationStep,
    RemoveMentionsStep,
    RemoveRepeatedCharsStep,
    RemoveTwitterMarkersStep,
    RemoveRetweetMarkerStep,
    RemoveUrlsStep,
    ReplaceEmojisStep,
    ReplaceLineBreaksStep,
    StripHtmlStep,
    TrimStep,
)


class TextNormalizer:
    def __init__(self, steps: list[NormalizationStep]):
        self.steps = steps

    def normalize(self, text: str | None) -> str:
        if text is None:
            text = ""

        text = str(text)

        for step in self.steps:
            text = step(text)

        return text


def default_normalizer(
    preserve_case: bool = False,
    remove_repeated_chars: bool = True,
    replace_emojis: bool = False,
    expand_sms: bool = False,
    sms_replacements: dict[str, str] | None = None,
) -> TextNormalizer:
    steps: list[NormalizationStep] = [
        StripHtmlStep(),
        TrimStep(),
        RemoveUrlsStep(),
        RemoveMentionsStep(),
        ReplaceLineBreaksStep(),
    ]

    if replace_emojis:
        steps.append(ReplaceEmojisStep())

    if remove_repeated_chars:
        steps.append(RemoveRepeatedCharsStep())

    if expand_sms:
        steps.append(ExpandSmsStep(sms_replacements or {}))

    steps.append(RemoveTwitterMarkersStep())
    steps.append(RemoveRetweetMarkerStep())
    

    if not preserve_case:
        steps.append(LowercaseStep())

    steps.append(CleanupSpacesStep())

    return TextNormalizer(steps)