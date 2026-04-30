# src/umutextstats/output/writer.py

from pathlib import Path

from umutextstats.output.resolvers import OutputResolverRegistry


class OutputWriter: 
    def __init__(self, registry: OutputResolverRegistry):
        self.registry = registry

    def write(self, df, path: str, output_format: str | None = None, **kwargs):
        path = Path(path)
        resolver = self.registry.resolve(path, output_format)
        return resolver.write(df, path, **kwargs)