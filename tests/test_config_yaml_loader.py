from pathlib import Path

from umutextstats.config.yaml_loader import load_yaml_config


def test_load_yaml_config(tmp_path: Path):
    path = tmp_path / "config.yaml"
    path.write_text(
        """
directory_folder: es
dimensions:
  - key: phonetics
    class: CompositeDimension
    strategy: CompositeStrategySum
    children:
      - key: phonetics-expressive-lengthening
        class: PatternDimension
        pattern: '(.)\\1{3,}'
        input_column: text_norm
""",
        encoding="utf-8",
    )

    config = load_yaml_config(path)

    assert config.directory_folder == "es"
    assert config.dimensions[0].key == "phonetics"
    assert (
        config.dimensions[0].children[0].key
        == "phonetics-expressive-lengthening"
    )
    assert (
        config.dimensions[0].children[0].class_name
        == "PatternDimension"
    )


def test_load_yaml_config_stores_extra_fields_as_params(
    tmp_path: Path,
):
    path = tmp_path / "config.yaml"
    path.write_text(
        """
dimensions:
  - key: nouns
    class: POSTaggingTag
    tag: NOUN

  - key: noun-verb-ratio
    class: RatioDimension
    numerator: nouns
    denominator: verbs
    scale: 100
    zero_division: 0
""",
        encoding="utf-8",
    )

    config = load_yaml_config(path)

    nouns = config.dimensions[0]
    ratio = config.dimensions[1]

    assert nouns.params == {
        "tag": "NOUN",
    }

    assert ratio.params == {
        "numerator": "nouns",
        "denominator": "verbs",
        "scale": 100,
        "zero_division": 0,
    }


def test_load_yaml_config_accepts_explicit_params(
    tmp_path: Path,
):
    path = tmp_path / "config.yaml"

    path.write_text(
        """
dimensions:
  - key: nouns
    class: POSTaggingTag
    params:
      tag: NOUN
""",
        encoding="utf-8",
    )

    config = load_yaml_config(path)

    assert config.dimensions[0].params == {
        "tag": "NOUN",
    }