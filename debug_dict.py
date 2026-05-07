import pandas as pd

from umutextstats.dimensions.word_per_dictionary import WordPerDictionary

df = pd.read_csv("dataset-small.csv")

df["text_norm"] = df["tweet"].fillna("").astype(str)
df["word_count"] = df["text_norm"].str.split().apply(len)

dim = WordPerDictionary(
    key="offensive",
    dictionary_name="offensive",
)

scores = dim.compute(df)

print("nonzero:", (scores != 0).sum())
print("ratio:", (scores != 0).mean())
print(df.loc[scores != 0, "tweet"].head(20).to_string(index=False))