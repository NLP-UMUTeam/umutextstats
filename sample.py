from umutextstats.io import read_input
from umutextstats.preprocessing.pipeline import preprocess_dataframe_cached
from umutextstats.nlp import annotate_dataframe_with_stanza
from umutextstats.cache import CacheManager

input_path = "dataset.csv"

cache = CacheManager(".cache")

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

print(df.head())