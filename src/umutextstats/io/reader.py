from pathlib import Path
from umutextstats.io.resolvers import InputResolverRegistry


class InputReader:
    def __init__(self, registry: InputResolverRegistry):
        self.registry = registry

    def read(self, path: str, text_column: str):
        path = Path(path)
        resolver = self.registry.resolve(path)
        return resolver.read(path, text_column=text_column)