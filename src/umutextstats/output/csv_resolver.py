# src/umutextstats/output/csv_resolver.py

class CSVOutputResolver:
    extensions = {".csv"}
    formats = {"csv"}

    def supports(self, path, output_format=None):
        if output_format:
            return output_format.lower() in self.formats

        return path.suffix.lower() in self.extensions

    def write(self, df, path, **kwargs):
        df.to_csv(path, index=False, **kwargs)
        return path