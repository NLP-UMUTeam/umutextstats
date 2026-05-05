from pathlib import Path
import unicodedata
from collections import defaultdict


ACCENTED = set("áéíóúüÁÉÍÓÚÜ")


def remove_accents(text: str) -> str:
    text = unicodedata.normalize("NFD", text)
    return "".join(ch for ch in text if unicodedata.category(ch) != "Mn")


def has_accent(word: str) -> bool:
    return any(ch in ACCENTED for ch in word)


def main():
    source = Path("src/umutextstats/resources/dictionaries/es.txt")
    output = Path("src/umutextstats/resources/accent_maps/es.txt")
    ambiguous_output = Path("src/umutextstats/resources/accent_maps/es_ambiguous.txt")

    groups = defaultdict(set)

    for raw in source.read_text(encoding="utf-8").splitlines():
        word = raw.strip().lower()

        if not word or word.startswith("#"):
            continue

        if not has_accent(word):
            continue

        plain = remove_accents(word)

        if plain != word:
            groups[plain].add(word)

    clean = {}
    ambiguous = {}

    for plain, accented_forms in groups.items():
        # Si solo hay una forma acentuada, es segura
        if len(accented_forms) == 1:
            clean[plain] = next(iter(accented_forms))
        else:
            ambiguous[plain] = sorted(accented_forms)

    output.write_text(
        "\n".join(sorted(clean.values())) + "\n",
        encoding="utf-8",
    )

    ambiguous_output.write_text(
        "\n".join(
            f"{plain}\t{','.join(forms)}"
            for plain, forms in sorted(ambiguous.items())
        ) + "\n",
        encoding="utf-8",
    )

    print(f"clean: {len(clean)}")
    print(f"ambiguous: {len(ambiguous)}")


if __name__ == "__main__":
    main()