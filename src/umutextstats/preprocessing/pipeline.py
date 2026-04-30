# src/umutextstats/preprocessing/pipeline.py

from umutextstats.cache import CacheManager
from umutextstats.preprocessing import preprocess_dataframe
from umutextstats.preprocessing.normalizer import TextNormalizer, default_normalizer


def preprocess_dataframe_cached(
    df,
    input_path: str,
    input_column: str = "text_raw",
    output_column: str = "text_norm",
    normalizer: TextNormalizer | None = None,
    show_progress: bool = True,
    use_cache: bool = True,
    cache: CacheManager | None = None,
):
    normalizer = normalizer or default_normalizer()
    cache = cache or CacheManager()

    params = {
        "input_column": input_column,
        "output_column": output_column,
        "normalizer": repr([step.__class__.__name__ for step in normalizer.steps]),
    }

    key = cache.build_key(input_path, "preprocess", params)

    if use_cache:
        cached = cache.load("preprocess", key)
        if cached is not None:
            return cached

    df = preprocess_dataframe(
        df,
        input_column=input_column,
        output_column=output_column,
        normalizer=normalizer,
        show_progress=show_progress,
    )

    if use_cache:
        cache.save(df, "preprocess", key)

    return df