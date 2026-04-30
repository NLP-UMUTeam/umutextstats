from umutextstats.io import read_input
from umutextstats.preprocessing.pipeline import preprocess_dataframe_cached
from umutextstats.nlp import annotate_dataframe_with_stanza
from umutextstats.cache import CacheManager
from umutextstats.config import load_config
from umutextstats.dimensions import DimensionEngine
from umutextstats.utils.profiler import Profiler
from umutextstats.output import write_output

input_path = "dataset-10.csv"

cache = CacheManager(".cache")
config = load_config()

profiler = Profiler(enabled=True)

with profiler.track("io", "read_input"):
    df = read_input(input_path, text_column="text")

with profiler.track("preprocessing", "normalize"):
    df = preprocess_dataframe_cached(
        df,
        input_path=input_path,
        cache=cache,
    )

with profiler.track("nlp", "stanza"):
    df = annotate_dataframe_with_stanza(
        df,
        input_path=input_path,
        cache=cache,
    )

engine = DimensionEngine(
    config=config,
    input_column="text_norm",
    include_unimplemented=True,
    profiler=profiler,
    show_progress=True,
)

with profiler.track("dimensions", "compute_all"):
    features = engine.compute(df)


stats = profiler.dataframe()

write_output(features, "features.csv")
write_output(profiler.dataframe(), "stats.json")