# src/umutextstats/dictionaries/loader.py

from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path


@dataclass
class DictionaryEntries:
    words: list[str]
    exceptions: list[str]


def default_dictionaries_path() -> Path:
    return Path(files("umutextstats") / "resources" / "dictionaries")


class DictionaryLoader:
    def __init__(self, dictionaries_path: str | Path | None = None):
        self.dictionaries_path = (
            Path(dictionaries_path)
            if dictionaries_path
            else default_dictionaries_path()
        )
        self._cache: dict[str, DictionaryEntries] = {}
    
    def load(self, name: str) -> DictionaryEntries:
        if name in self._cache:
            return self._cache[name]

        path = self.dictionaries_path / f"{name}.txt"

        if not path.exists():
            entries = DictionaryEntries(words=[], exceptions=[])
            self._cache[name] = entries
            return entries

        words = []
        exceptions = []

        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()

            if not line or line.startswith("#"):
                continue

            if line.startswith("NEG:"):
                exception = line[1:].strip()
                if exception:
                    exceptions.append(exception)
            else:
                words.append(line)

        entries = DictionaryEntries(words=words, exceptions=exceptions)
        self._cache[name] = entries
        return entries