# src/umutextstats/preprocessing/__init__.py

from tqdm.auto import tqdm

from umutextstats.preprocessing.normalizer import TextNormalizer, default_normalizer


def preprocess_dataframe(
    df,
    input_column: str = "text_raw",
    output_column: str = "text_norm",
    normalizer: TextNormalizer | None = None,
    show_progress: bool = True,
):
    if input_column not in df.columns:
        raise ValueError(f"Input column '{input_column}' not found in dataframe")

    # Lazy enough: normalizer only created when this function is actually computing.
    normalizer = normalizer or default_normalizer()

    if show_progress:
        tqdm.pandas(desc="Preprocessing")
        df[output_column] = df[input_column].progress_apply(normalizer.normalize)
    else:
        df[output_column] = df[input_column].apply(normalizer.normalize)

    return df