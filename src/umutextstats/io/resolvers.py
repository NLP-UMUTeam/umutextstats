from pathlib import Path


class InputResolverRegistry:
    def __init__(self):
        self._resolvers = []

    def register(self, resolver):
        self._resolvers.append(resolver)

    def resolve(self, path: Path):
        for resolver in self._resolvers:
            if resolver.supports(path):
                return resolver

        raise ValueError(f"No input resolver found for file: {path}")