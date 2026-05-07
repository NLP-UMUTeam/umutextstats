# src/umutextstats/nlp/stanza_annotator.py

from dataclasses import dataclass

from tqdm.auto import tqdm


@dataclass
class StanzaAnnotator:
    lang: str = "es"
    processors: str = "tokenize,mwt,pos,ner"
    use_gpu: bool = False
    pos_batch_size: int = 8000
    ner_batch_size: int = 32
    batch_size_docs: int = 128
    download_models: bool = False

    def __post_init__(self):
        try:
            import stanza
        except ImportError as exc:
            raise ImportError(
                "Stanza is not installed. Install it with: pip install -e '.[nlp]'"
            ) from exc

        self.stanza = stanza

        if self.download_models:
            stanza.download(self.lang)

        self.nlp = stanza.Pipeline(
            lang=self.lang,
            processors=self.processors,
            use_gpu=self.use_gpu,
            pos_batch_size=self.pos_batch_size,
            ner_batch_size=self.ner_batch_size,
        )

    def annotate_texts(self, texts: list[str]):
        docs = []

        for i in tqdm(
            range(0, len(texts), self.batch_size_docs),
            desc="Stanza por lotes",
        ):
            batch = texts[i : i + self.batch_size_docs]

            in_docs = [
                self.stanza.Document([], text=text or "")
                for text in batch
            ]

            out_docs = self.nlp(in_docs)
            docs.extend(out_docs)

        return docs

    def annotate_dataframe(
        self,
        df,
        input_column: str = "text_norm",
        output_column: str = "stanza_doc",
    ):
        if output_column in df.columns:
            return df

        texts = df[input_column].fillna("").astype(str).tolist()
        df[output_column] = self.annotate_texts(texts)

        return df


def format_tagged_pos(doc) -> str:
    items = []

    for sent in doc.sentences:
        for word in sent.words:
            tag = word.upos or word.xpos or ""
            feats = word.feats or ""

            if feats:
                items.append(f"{word.text}__({tag})({feats})")
            else:
                items.append(f"{word.text}__({tag})")

    return ", ".join(items)


def format_tagged_ner(doc) -> str:
    return ", ".join(
        ", ".join(
            f"{ent.type}({ent.text})"
            for ent in sent.ents
        )
        for sent in doc.sentences
    )