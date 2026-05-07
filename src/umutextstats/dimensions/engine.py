from contextlib import nullcontext
from tqdm.auto import tqdm

import pandas as pd

from umutextstats.config.models import DimensionConfig, UMUTextStatsConfig
from umutextstats.dimensions.registry import resolve_dimension, normalize_class_name


class DimensionEngine:
    def __init__(
        self,
        config: UMUTextStatsConfig,
        input_column: str = "text_norm",
        include_unimplemented: bool = True,
        profiler=None,
        show_progress: bool = True,
    ):
        self.config = config
        self.input_column = input_column
        self.include_unimplemented = include_unimplemented
        self.profiler = profiler
        self.show_progress = show_progress

    def compute(self, df) -> pd.DataFrame:
        data = {}

        if "id" in df.columns:
            data["id"] = df["id"]

        dimensions = list(self._iter_dimensions(self.config.dimensions))

        iterator = tqdm(
            dimensions,
            desc="Dimensions",
            unit="dimension",
            disable=not self.show_progress,
        )

        for dimension in iterator:
            iterator.set_postfix_str(dimension.key)
            self._compute_dimension(df, dimension, data)

        return pd.DataFrame(data)

    def _iter_dimensions(self, dimensions: list[DimensionConfig]):
        for dimension in dimensions:
            yield dimension

            if dimension.children:
                yield from self._iter_dimensions(dimension.children)

    def _compute_dimension(self, df, dimension: DimensionConfig, data: dict):
        
        key = dimension.key
        
        if dimension.key in data:
            return
        
        class_name = normalize_class_name(dimension.class_name)

        with self._track_dimension(key, class_name):
            for child in dimension.children:
                self._compute_dimension(df, child, data)

            if dimension.children:
                data[key] = self._compute_composite_dimension(
                    dimension,
                    data,
                    len(df),
                )
                return

            dimension_cls = resolve_dimension(dimension.class_name)

            if dimension_cls is None:
                if self.include_unimplemented:
                    data[key] = [""] * len(df)
                return

            instance = self._build_dimension(dimension, dimension_cls)
            data[key] = instance.compute(df)

    def _compute_composite_dimension(self, dimension, data, n_rows):
        strategy = (dimension.strategy or "CompositeStrategySum").upper()

        child_keys = [
            child.key
            for child in dimension.children
            if child.key in data
        ]

        if not child_keys:
            return [0] * n_rows

        child_df = pd.DataFrame({
            key: data[key]
            for key in child_keys
        })

        child_df = child_df.apply(pd.to_numeric, errors="coerce").fillna(0)

        if strategy == "COMPOSITESTRATEGYSUM":
            return child_df.sum(axis=1)

        if strategy == "COMPOSITESTRATEGYAVG":
            return child_df.mean(axis=1)

        if strategy == "COMPOSITESTRATEGYMAX":
            return child_df.max(axis=1)

        if strategy == "COMPOSITESTRATEGYMIN":
            return child_df.min(axis=1)

        # fallback seguro
        return child_df.sum(axis=1)

    def _track_dimension(self, key: str, class_name: str | None):
        if self.profiler is None:
            return nullcontext()

        return self.profiler.track(
            stage="dimension",
            name=key,
            class_name=class_name or "",
        )

    def _build_dimension(self, dimension: DimensionConfig, dimension_cls):

        class_name = normalize_class_name(dimension.class_name)

        input_column = (
            "text_raw"
            if dimension.use_original_input
            else self.input_column
        )

        params = dimension.params

        if class_name == "WordPerDictionary":
            return dimension_cls(
                key=dimension.key,
                dictionary_name=self._dictionary_param(dimension),
                input_column=input_column,
                percentage=self._percentage_param(dimension),
                use_regex=not self._disabled_regexp_param(dimension),
            )

        if class_name == "PatternDimension":
            return dimension_cls(
                key=dimension.key,
                pattern=self._param(dimension, "pattern", ""),
                input_column=input_column,
                percentage=self._percentage_param(dimension),
            )

        if class_name == "POSTaggingTag":
            return dimension_cls(
                key=dimension.key,
                input_column="tagged_pos",
                postagger_tag=self._param(dimension, "tag"),
                postagger_universal=self._param(dimension, "universal"),
            )

        if class_name == "POSTaggingExpression":
            return dimension_cls(
                key=dimension.key,
                pattern=self._param(dimension, "pattern", ""),
                input_column="tagged_pos",
                percentage=self._percentage_param(dimension),
            )

        if class_name == "NERTaggingTag":
            return dimension_cls(
                key=dimension.key,
                tag=self._param(dimension, "tag"),
                input_column="tagged_ner",
            )

        if class_name == "EncliticsPersonalPronounsDictionary":
            return dimension_cls(
                key=dimension.key,
                dictionary_name=self._dictionary_param(dimension),
                input_column=input_column,
            )

        if class_name == "PeriphrasisDimension":
            return dimension_cls(
                key=dimension.key,
                auxiliar_verbs=self._param(dimension, "auxiliar_verbs", ""),
                input_column=input_column,
                tagged_pos_column="tagged_pos",
            )

        if class_name == "CharacterCountDimension":
            chars = self._param(dimension, "character", "")

            if chars == "SPACE":
                chars = " "

            return dimension_cls(
                key=dimension.key,
                chars=chars,
                input_column=input_column,
            )

        if class_name == "ErrorCapitalizationStartingWithLowerCaseDimension":
            return dimension_cls(
                key=dimension.key,
                input_column="text_raw",
            )

        if class_name == "ErrorStyleSentencesStartingWithNumbers":
            return dimension_cls(
                key=dimension.key,
                input_column="text_raw",
            )

        if class_name == "ErrorStyleSentencesStartingWithTheSameWord":
            return dimension_cls(
                key=dimension.key,
                input_column="text_raw",
            )

        if class_name == "GrammaticalGenderDimension":
            return dimension_cls(
                key=dimension.key,
                dictionary_name=self._dictionary_param(dimension),
                input_column=input_column,
                tagged_pos_column="tagged_pos",
                percentage=self._percentage_param(dimension),
                use_regex=not self._disabled_regexp_param(dimension),
            )

        if class_name == "LanguageDimension":
            return dimension_cls(
                key=dimension.key,
                language=self._param(dimension, "language", ""),
                input_column=input_column,
            )

        if class_name in {"RTIEDimension", "RTIEDeviationDimension"}:
            return dimension_cls(
                key=dimension.key,
                input_column=input_column,
                separator=self._param(dimension, "separator", "by-chunks"),
            )

        if class_name == "SentencePerDictionary":
            return dimension_cls(
                key=dimension.key,
                dictionary_name=self._dictionary_param(dimension),
                input_column=input_column,
                percentage=self._percentage_param(dimension),
            )

        if class_name == "TwitterReplyToDimension":
            return dimension_cls(
                key=dimension.key,
                dictionary_name=self._dictionary_param(dimension),
                input_column="text_raw",
            )

        if class_name == "VerbPerDictionary":
            return dimension_cls(
                key=dimension.key,
                dictionary_name=self._dictionary_param(dimension),
                input_column=input_column,
                percentage=self._percentage_param(dimension),
            )

        if class_name == "WordUniqueDimension":
            return dimension_cls(
                key=dimension.key,
                input_column=input_column,
                percentage=self._percentage_param(dimension),
            )

        if class_name == "WordCase":
            return dimension_cls(
                key=dimension.key,
                comparator=(
                    self._param(dimension, "word_comparator")
                    or self._param(dimension, "comparator")
                    or "upper"
                ),
                input_column="text_raw",
            )

        if class_name == "WordLengthDimension":
            return dimension_cls(
                key=dimension.key,
                length=self._param(dimension, "length"),
                comparator=self._param(dimension, "comparator", "="),
                input_column=input_column,
                percentage=self._percentage_param(dimension),
            )

        return dimension_cls(
            key=dimension.key,
            input_column=input_column,
        )
        
    def _param(self, dimension, name: str, default=None):
        value = dimension.params.get(name)

        if value is not None:
            return value

        return getattr(dimension, name, default)

    def _bool_value(self, value, default: bool = False) -> bool:
        if value is None:
            return default

        if isinstance(value, bool):
            return value

        if isinstance(value, str):
            return value.strip().lower() in {"true", "1", "yes", "y"}

        return bool(value)

    def _dictionary_param(self, dimension) -> str:
        return (
            dimension.params.get("dictionary")
            or dimension.params.get("dictionaries")
            or getattr(dimension, "dictionary", None)
            or ""
        )

    def _percentage_param(self, dimension) -> bool:
        value = dimension.params.get("percentage", None)
        return self._bool_value(value, default=dimension.percentage)

    def _disabled_regexp_param(self, dimension) -> bool:
        value = (
            dimension.params.get("disabledregexp")
            if "disabledregexp" in dimension.params
            else dimension.params.get("disabled_regexp", None)
        )

        return self._bool_value(value, default=dimension.disabled_regexp)