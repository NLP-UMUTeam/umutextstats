# src/umutextstats/dimensions/registry.py

from typing import Type

from umutextstats.dimensions.base import BaseDimension

# Importa aquí las dimensiones implementadas
from umutextstats.dimensions.length import LengthDimension
from umutextstats.dimensions.sentence_count import SentenceCountDimension
from umutextstats.dimensions.syllable_count import SyllableCountDimension
from umutextstats.dimensions.word_count import WordCountDimension


# =========================
# Registry
# =========================

DIMENSION_REGISTRY: dict[str, Type[BaseDimension]] = {
    "LengthDimension": LengthDimension,
    "SentenceCountDimension": SentenceCountDimension,
    "SyllableCountDimension": SyllableCountDimension,
    "WordCountDimension": WordCountDimension,
}


# =========================
# Helpers
# =========================

def normalize_class_name(class_name: str | None) -> str | None:
    """
    Normaliza nombres heredados del XML/PHP.

    Ejemplo:
    UMUTextStats\\Dimensions\\LengthDimension → LengthDimension
    """
    if not class_name:
        return None

    return class_name.replace("\\", "/").split("/")[-1]


def resolve_dimension(class_name: str | None) -> Type[BaseDimension] | None:
    """
    Devuelve la clase Python correspondiente a una dimensión.
    """
    normalized = normalize_class_name(class_name)

    if not normalized:
        return None

    return DIMENSION_REGISTRY.get(normalized)


def list_available_dimensions() -> list[str]:
    """
    Devuelve las dimensiones registradas (debug / info).
    """
    return sorted(DIMENSION_REGISTRY.keys())