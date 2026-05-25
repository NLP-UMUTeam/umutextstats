# scripts/annotate_feature_cases.py

from pathlib import Path

import yaml

from umutextstats.config.loader import load_config
from umutextstats.nlp.stanza_annotator import (
    StanzaAnnotator,
    format_tagged_pos,
)


def iter_validation_files(dimensions):
    for dimension in dimensions:
        validation = getattr(dimension, "validation", None)

        if validation:
            cases_path = validation.get("cases")

            if cases_path:
                yield Path(cases_path)

        yield from iter_validation_files(dimension.children)


def main():
    config = load_config()
    root = Path(__file__).parent.parent

    paths = sorted(set(iter_validation_files(config.dimensions)))

    annotator = StanzaAnnotator()

    changed = 0

    for relative_path in paths:
        path = root / relative_path

        if not path.exists():
            continue

        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        cases = data.get("cases", [])

        texts_to_annotate = []
        case_indexes = []

        for i, case in enumerate(cases):
            annotations = case.get("annotations") or {}

            if annotations.get("tagged_pos"):
                continue

            texts_to_annotate.append(case["text"])
            case_indexes.append(i)

        if not texts_to_annotate:
            continue

        docs = annotator.annotate_texts(texts_to_annotate)

        for case_index, doc in zip(case_indexes, docs):
            case = cases[case_index]
            annotations = case.setdefault("annotations", {})
            annotations["tagged_pos"] = format_tagged_pos(doc)
            changed += 1

        path.write_text(
            yaml.safe_dump(
                data,
                allow_unicode=True,
                sort_keys=False,
            ),
            encoding="utf-8",
        )

    print(f"Annotated cases: {changed}")


if __name__ == "__main__":
    main()