from pathlib import Path

from umutextstats.config.loader import load_config


DEFAULT_YAML = Path("src/umutextstats/resources/config/default.yaml")
CONFIG_DIR = Path("src/umutextstats/resources/config")


def test_default_yaml_uses_includes():
    text = DEFAULT_YAML.read_text(encoding="utf-8")

    assert "includes:" in text
    assert "dimensions:" not in text


def test_default_yaml_includes_exist():
    config_index = load_config(DEFAULT_YAML)

    assert config_index.dimensions

    # Comprueba que las rutas del índice existen físicamente
    import yaml

    data = yaml.safe_load(DEFAULT_YAML.read_text(encoding="utf-8"))

    for include in data["includes"]:
        assert (CONFIG_DIR / include).exists()


def test_default_yaml_loads_expected_top_level_dimensions():
    config = load_config(DEFAULT_YAML)

    keys = [dimension.key for dimension in config.dimensions]

    assert keys == [
        "phonetics",
        "morphosyntax",
        "errors",
        "semantics",
        "pragmatics",
        "stylometry",
        "lexical",
        "psycholinguistic-processes",
        "register",
        "digital-communication",
        "syntax",
    ]