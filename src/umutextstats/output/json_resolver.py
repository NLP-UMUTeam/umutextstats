# src/umutextstats/output/json_resolver.py

class JSONOutputResolver:
    extensions = {".json"}
    formats = {"json"}

    def supports(self, path, output_format=None):
        if output_format:
            return output_format.lower() in self.formats

        return path.suffix.lower() in self.extensions

    def write(self, df, path, orient="records", force_ascii=False, indent=2, **kwargs):
        df.to_json(
            path,
            orient=orient,
            force_ascii=force_ascii,
            indent=indent,
            **kwargs,
        )
        return path