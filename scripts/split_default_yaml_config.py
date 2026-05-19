# scripts/split_default_yaml_config.py

from pathlib import Path

import yaml


INPUT = Path("src/umutextstats/resources/config/default.yaml")
OUTPUT_DIR = Path("src/umutextstats/resources/config/es")


def main() -> None:
    data = yaml.safe_load(INPUT.read_text(encoding="utf-8"))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    includes = []

    for dimension in data["dimensions"]:
        key = dimension["key"]
        output_path = OUTPUT_DIR / f"{key}.yaml"

        output_data = {
            "dimensions": [dimension],
        }

        output_path.write_text(
            yaml.safe_dump(
                output_data,
                allow_unicode=True,
                sort_keys=False,
                width=1000,
            ),
            encoding="utf-8",
        )

        includes.append(f"es/{key}.yaml")

    index_data = {
        "directory_folder": data.get("directory_folder"),
        "includes": includes,
    }

    INPUT.write_text(
        yaml.safe_dump(
            index_data,
            allow_unicode=True,
            sort_keys=False,
            width=1000,
        ),
        encoding="utf-8",
    )

    print(f"Split {len(includes)} top-level dimensions into {OUTPUT_DIR}")
    print(f"Updated index file: {INPUT}")


if __name__ == "__main__":
    main()