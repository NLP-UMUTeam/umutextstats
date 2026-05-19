from pathlib import Path

from umutextstats.config.yaml_loader import load_yaml_config


def test_load_yaml_config_with_includes(tmp_path: Path):
    included = tmp_path / "phonetics.yaml"
    included.write_text(
        """
dimensions:
  - key: phonetics
    class: CompositeDimension
    children:
      - key: phonetics-expressive-lengthening
        class: PatternDimension
        pattern: '(.)\\1{3,}'
""",
        encoding="utf-8", 
    )

    index = tmp_path / "default.yaml"
    index.write_text(
        """
directory_folder: es
includes:
  - phonetics.yaml
""",
        encoding="utf-8",
    )

    config = load_yaml_config(index)

    assert config.directory_folder == "es"
    assert len(config.dimensions) == 1
    assert config.dimensions[0].key == "phonetics"
    assert config.dimensions[0].children[0].key == "phonetics-expressive-lengthening"