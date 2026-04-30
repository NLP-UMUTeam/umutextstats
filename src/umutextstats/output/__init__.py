# src/umutextstats/output/__init__.py

from umutextstats.output.csv_resolver import CSVOutputResolver
from umutextstats.output.json_resolver import JSONOutputResolver
from umutextstats.output.resolvers import OutputResolverRegistry
from umutextstats.output.writer import OutputWriter


def write_output(
    df,
    path: str,
    output_format: str | None = None,
    **kwargs,
):
    registry = OutputResolverRegistry()
    registry.register(CSVOutputResolver())
    registry.register(JSONOutputResolver())

    writer = OutputWriter(registry)

    return writer.write(
        df,
        path=path,
        output_format=output_format,
        **kwargs,
    )


__all__ = [
    "write_output",
    "OutputWriter",
    "OutputResolverRegistry",
    "CSVOutputResolver",
    "JSONOutputResolver",
]