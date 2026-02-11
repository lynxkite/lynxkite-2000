"""Operations for reading and writing files."""

import enum
import fsspec
from lynxkite_core import ops

from .. import core
import networkx as nx
import pandas as pd


op = ops.op_registration(core.ENV)


class FileFormat(enum.StrEnum):
    csv = "csv"
    parquet = "parquet"
    json = "json"
    excel = "excel"


@op(
    "Import file",
    color="green",
    icon="file-filled",
    params=[
        ops.ParameterGroup(
            name="file_format_group",
            selector=ops.Parameter(name="file_format", type=FileFormat, default=FileFormat.csv),
            groups={
                "csv": [
                    ops.Parameter.basic("columns", type=str, default="<from file>"),
                    ops.Parameter.basic("separator", type=str, default="<auto>"),
                ],
                "parquet": [],
                "json": [],
                "excel": [ops.Parameter.basic("sheet_name", type=str, default="Sheet1")],
            },
            default=FileFormat.csv,
        ),
    ],
    slow=True,
)
def import_file(
    *, file_path: ops.PathStr, table_name: str, file_format: FileFormat = FileFormat.csv, **kwargs
) -> core.Bundle:
    """Read the contents of the a file into a `Bundle`.

    Args:
        file_path: Path to the file to import.
        table_name: Name to use for identifying the table in the bundle.
        file_format: Format of the file. Has to be one of the values in the `FileFormat` enum.

    Returns:
        Bundle: Bundle with a single table with the contents of the file.
    """
    if file_format == "csv":
        names = kwargs.get("columns", "<from file>")
        names = pd.api.extensions.no_default if names == "<from file>" else names.split(",")
        sep = kwargs.get("separator", "<auto>")
        sep = pd.api.extensions.no_default if sep == "<auto>" else sep.replace("\\t", "\t")
        df = pd.read_csv(file_path, names=names, sep=sep)  # ty: ignore[invalid-argument-type]
    elif file_format == "json":
        with open(file_path, "r") as f:
            df = pd.read_json(f)
    elif file_format == "parquet":
        df = pd.read_parquet(file_path)
    elif file_format == "excel":
        df = pd.read_excel(file_path, sheet_name=kwargs.get("sheet_name", "Sheet1"))
    else:
        raise ValueError(f"Unsupported file format: {file_format}")
    return core.Bundle(dfs={table_name: df})


@op("Export to file", icon="file-filled")
def export_to_file(
    bundle: core.Bundle,
    *,
    table_name: str,
    filename: ops.PathStr,
    file_format: FileFormat = FileFormat.csv,
):
    """Exports a DataFrame to a file.

    Args:
        bundle: The bundle containing the DataFrame to export.
        table_name: The name of the DataFrame in the bundle to export.
        filename: The name of the file to export to.
        file_format: The format of the file to export to. Defaults to CSV.
    """

    df = bundle.dfs[table_name]
    if file_format == FileFormat.csv:
        df.to_csv(filename, index=False)
    elif file_format == FileFormat.json:
        df.to_json(filename, orient="records", lines=True)
    elif file_format == FileFormat.parquet:
        df.to_parquet(filename, index=False)
    elif file_format == FileFormat.excel:
        df.to_excel(filename, index=False)
    else:
        raise ValueError(f"Unsupported file format: {file_format}")


@op("Import Parquet", color="green", icon="file-filled")
def import_parquet(*, filename: ops.PathStr):
    """Imports a Parquet file."""
    return pd.read_parquet(filename)


@op("Import CSV", slow=True, color="green", icon="file-filled")
def import_csv(*, filename: ops.PathStr, columns: str = "<from file>", separator: str = "<auto>"):
    """Imports a CSV file."""
    names = pd.api.extensions.no_default if columns == "<from file>" else columns.split(",")
    sep = pd.api.extensions.no_default if separator == "<auto>" else separator
    return pd.read_csv(filename, names=names, sep=sep)  # ty: ignore[invalid-argument-type]


@op("Import GraphML", slow=True, color="green", icon="topology-star-3")
def import_graphml(*, filename: ops.PathStr):
    """Imports a GraphML file."""
    files = fsspec.open_files(filename, compression="infer")
    for f in files:
        if ".graphml" in f.path:
            with f as f:
                return nx.read_graphml(f)
    raise ValueError(f"No .graphml file found at {filename}")


@op("Graph from OSM", slow=True)
def import_osm(*, location: str):
    import osmnx as ox

    return ox.graph.graph_from_place(location, network_type="drive")
