# src/umutextstats/io/__init__.py

from umutextstats.io.reader import InputReader
from umutextstats.io.resolvers import InputResolverRegistry
from umutextstats.io.csv_resolver import CSVInputResolver


def read_input(path: str, text_column: str):
    registry = InputResolverRegistry()
    registry.register(CSVInputResolver())

    reader = InputReader(registry)
    return reader.read(path, text_column=text_column)