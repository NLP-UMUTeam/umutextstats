# src/umutextstats/output/resolvers.py

from pathlib import Path


class OutputResolverRegistry:
    def __init__(self):
        self._resolvers = []

    def register(self, resolver):
        self._resolvers.append(resolver)

    def resolve(self, path: str | Path, output_format: str | None = None):
        path = Path(path)

        for resolver in self._resolvers:
            if resolver.supports(path, output_format):
                return resolver

        raise ValueError(
            f"No output resolver found for path={path} format={output_format}"
        )