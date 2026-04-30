from umutextstats.io import read_input
from umutextstats.preprocessing.pipeline import preprocess_dataframe_cached
from umutextstats.nlp import annotate_dataframe_with_stanza
from umutextstats.cache import CacheManager
from umutextstats.config import load_config
from umutextstats.dimensions import DimensionEngine

input_path = "dataset.csv"

cache = CacheManager(".cache")
config = load_config()

df = read_input(input_path, text_column="text")

df = preprocess_dataframe_cached(
    df,
    input_path=input_path,
    cache=cache,
)

df = annotate_dataframe_with_stanza(
    df,
    input_path=input_path,
    cache=cache,
)

features = DimensionEngine(
    config=config,
    input_column="text_norm",
    include_unimplemented=True,
).compute(df)

features.to_csv("features.csv", index=False)

print(features.head())