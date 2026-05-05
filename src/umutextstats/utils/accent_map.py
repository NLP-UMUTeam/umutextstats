# src/umutextstats/utils/accent_map.py

from importlib.resources import files
from pathlib import Path

from umutextstats.dimensions.enclitics_personal_pronouns import remove_accents


_ACCENT_MAPS: dict[str, dict[str, str]] = {}


def default_accent_map_path(language: str = "es") -> Path:
    return Path(
        files("umutextstats")
        / "resources"
        / "accent_maps"
        / f"{language}.txt"
    )


def load_accent_map(language: str = "es", path: str | Path | None = None) -> dict[str, str]:
    cache_key = f"{language}:{path or ''}"

    if cache_key in _ACCENT_MAPS:
        return _ACCENT_MAPS[cache_key]

    source = Path(path) if path else default_accent_map_path(language)

    accent_map = {}

    if source.exists():
        for raw_line in source.read_text(encoding="utf-8").splitlines():
            word = raw_line.strip().lower()

            if not word or word.startswith("#"):
                continue

            plain = remove_accents(word)

            if plain != word:
                accent_map[plain] = word

    _ACCENT_MAPS[cache_key] = accent_map
    return accent_map