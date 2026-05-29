import pandas as pd

from umutextstats.io.text import ensure_text


class CSVInputResolver:
    extensions = {".csv"}

    def supports(self, path):
        return path.suffix.lower() in self.extensions

    def read(self, path, text_column: str):
        df = pd.read_csv(path)

        if text_column not in df.columns:
            raise ValueError(f"Text column '{text_column}' not found in CSV")

        df = df.copy()

        df["text_raw"] = df[text_column].map(ensure_text)
        df["text"] = df["text_raw"]

        if "id" not in df.columns:
            df.insert(0, "id", range(len(df)))

        return df