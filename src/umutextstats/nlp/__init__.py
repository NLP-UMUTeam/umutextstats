from umutextstats.cache import CacheManager
from umutextstats.nlp.stanza_annotator import (
    StanzaAnnotator,
    format_tagged_ner,
    format_tagged_pos,
)


def annotate_dataframe_with_stanza(
    df,
    input_path: str,
    input_column: str = "text_norm",
    doc_column: str = "stanza_doc",
    pos_column: str = "tagged_pos",
    ner_column: str = "tagged_ner",
    annotator: StanzaAnnotator | None = None,
    use_cache: bool = True,
    cache: CacheManager | None = None,
):
    cache = cache or CacheManager()

    params = {
        "input_column": input_column,
        "pos_column": pos_column,
        "ner_column": ner_column,
        "lang": annotator.lang if annotator else StanzaAnnotator.lang,
        "processors": annotator.processors if annotator else StanzaAnnotator.processors,
        "use_gpu": annotator.use_gpu if annotator else StanzaAnnotator.use_gpu,
        "pos_batch_size": annotator.pos_batch_size if annotator else StanzaAnnotator.pos_batch_size,
        "ner_batch_size": annotator.ner_batch_size if annotator else StanzaAnnotator.ner_batch_size,
        "batch_size_docs": annotator.batch_size_docs if annotator else StanzaAnnotator.batch_size_docs,
    }

    key = cache.build_key(input_path, "stanza", params)

    if use_cache:
        cached = cache.load("stanza", key)
        if cached is not None:
            return cached

    # Lazy initialization: Stanza only loads if cache is missing
    annotator = annotator or StanzaAnnotator()

    df = annotator.annotate_dataframe(
        df,
        input_column=input_column,
        output_column=doc_column,
    )

    df[pos_column] = df[doc_column].apply(format_tagged_pos)
    df[ner_column] = df[doc_column].apply(format_tagged_ner)

    if doc_column in df.columns:
        df = df.drop(columns=[doc_column])

    if use_cache:
        cache.save(df, "stanza", key)

    return df