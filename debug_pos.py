import pandas as pd

from umutextstats.nlp import annotate_dataframe_with_stanza
from umutextstats.dimensions.pos_tagging_tag import POSTaggingTag

df = pd.DataFrame(
    {
        "text_norm": [
            "Yo comí pan ayer.",
            "Ella canta muy bien.",
            "Los perros grandes corren rápido.",
        ]
    }
)

df = annotate_dataframe_with_stanza(
    df,
    input_path="debug_pos.py",
    use_cache=False,
)

print("\nTAGGED POS")
print(df["tagged_pos"].to_string(index=False))

tests = [
    ("verbs", POSTaggingTag(key="verbs", postagger_tag="VERB")),
    ("present_verbs", POSTaggingTag(key="present_verbs", postagger_tag="VERB", postagger_universal="Tense=Pres")),
    ("past_verbs", POSTaggingTag(key="past_verbs", postagger_tag="VERB", postagger_universal="Tense=Past")),
    ("singular", POSTaggingTag(key="singular", postagger_universal="Number=Sing")),
    ("masculine", POSTaggingTag(key="masculine", postagger_universal="Gender=Masc")),
    ("feminine", POSTaggingTag(key="feminine", postagger_universal="Gender=Fem")),
]

print("\nRESULTS")
for name, dim in tests:
    print(name)
    print(dim.compute(df).to_string(index=False))