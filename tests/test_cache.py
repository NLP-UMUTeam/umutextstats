import pandas as pd

from umutextstats.cache import CacheManager


def test_save_and_load_cache(tmp_path):
    cache = CacheManager(tmp_path)

    df = pd.DataFrame({"a": [1, 2, 3]})

    key = "test123"

    cache.save(df, "stage", key)
    loaded = cache.load("stage", key)

    assert loaded is not None
    assert loaded.equals(df)
    
    
def test_cache_key_changes_with_params(tmp_path):
    cache = CacheManager(tmp_path)

    path = tmp_path / "file.csv"
    path.write_text("dummy")

    key1 = cache.build_key(path, "stage", {"a": 1})
    key2 = cache.build_key(path, "stage", {"a": 2})

    assert key1 != key2
    
    
def test_cache_key_same_for_same_input(tmp_path):
    cache = CacheManager(tmp_path)

    path = tmp_path / "file.csv"
    path.write_text("dummy")

    key1 = cache.build_key(path, "stage", {"a": 1})
    key2 = cache.build_key(path, "stage", {"a": 1})

    assert key1 == key2
    
    
from umutextstats.preprocessing.pipeline import preprocess_dataframe_cached


def test_preprocess_uses_cache(tmp_path):
    cache = CacheManager(tmp_path)

    df = pd.DataFrame({"text_raw": ["Hola mundo"]})

    input_path = tmp_path / "input.csv"
    input_path.write_text("dummy")

    # Primera ejecución (crea cache)
    df1 = preprocess_dataframe_cached(
        df.copy(),
        input_path=str(input_path),
        cache=cache,
    )

    # Segunda ejecución (debería cargar cache)
    df2 = preprocess_dataframe_cached(
        df.copy(),
        input_path=str(input_path),
        cache=cache,
    )

    assert df1.equals(df2)
    
    