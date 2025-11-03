"""Graph analytics operations."""

import enum
import os
import fsspec
from lynxkite_core import ops
from collections import deque

from . import core
import grandcypher
import matplotlib.cm
import matplotlib.colors
import networkx as nx
import pandas as pd
import polars as pl
import json


op = ops.op_registration(core.ENV)


class FileFormat(enum.StrEnum):
    csv = "csv"
    parquet = "parquet"
    json = "json"
    excel = "excel"


@op(
    "Import file",
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
        df = pd.read_csv(file_path, names=names, sep=sep)
    elif file_format == "json":
        with open(file_path, "r") as f:
            df = pd.read_json(f)
    elif file_format == "parquet":
        df = pd.read_parquet(file_path)
    elif file_format == "excel":
        df = pd.read_excel(file_path, sheet_name=kwargs.get("sheet_name", "Sheet1"))
    else:
        df = ValueError(f"Unsupported file format: {file_format}")
    return core.Bundle(dfs={table_name: df})


@op("Export to file")
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


@op("Import Parquet")
def import_parquet(*, filename: ops.PathStr):
    """Imports a Parquet file."""
    return pd.read_parquet(filename)


@op("Import CSV", slow=True)
def import_csv(*, filename: ops.PathStr, columns: str = "<from file>", separator: str = "<auto>"):
    """Imports a CSV file."""
    return pd.read_csv(
        filename,
        names=pd.api.extensions.no_default if columns == "<from file>" else columns.split(","),
        sep=pd.api.extensions.no_default if separator == "<auto>" else separator,
    )


@op("Import GraphML", slow=True)
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


@op("Discard loop edges")
def discard_loop_edges(graph: nx.Graph):
    graph = graph.copy()
    graph.remove_edges_from(nx.selfloop_edges(graph))
    return graph


@op("Discard parallel edges")
def discard_parallel_edges(graph: nx.Graph):
    return nx.DiGraph(graph)


@op("SQL")
def sql(bundle: core.Bundle, *, query: ops.LongStr, save_as: str = "result"):
    """Run a SQL query on the DataFrames in the bundle. Save the results as a new DataFrame."""
    bundle = bundle.copy()
    if os.environ.get("NX_CUGRAPH_AUTOCONFIG", "").strip().lower() == "true":
        with pl.Config() as cfg:
            cfg.set_verbose(True)
            res = pl.SQLContext(bundle.dfs).execute(query).collect(engine="gpu").to_pandas()
            # TODO: Currently `collect()` moves the data from cuDF to Polars. Then we convert it to Pandas,
            # which (hopefully) puts it back into cuDF. Hopefully we will be able to keep it in cuDF.
    else:
        res = pl.SQLContext(bundle.dfs).execute(query).collect().to_pandas()
    bundle.dfs[save_as] = res
    return bundle


@op("Cypher")
def cypher(bundle: core.Bundle, *, query: ops.LongStr, save_as: str = "result"):
    """Run a Cypher query on the graph in the bundle. Save the results as a new DataFrame."""
    bundle = bundle.copy()
    graph = bundle.to_nx()
    res = grandcypher.GrandCypher(graph).run(query)
    bundle.dfs[save_as] = pd.DataFrame(res)
    return bundle


@op("Sample graph")
def sample_graph(graph: nx.Graph, *, nodes: int = 100):
    """Takes a (preferably connected) subgraph."""
    sample = set()
    to_expand = deque([next(graph.nodes.keys().__iter__())])
    while to_expand and len(sample) < nodes:
        node = to_expand.pop()
        for n in graph.neighbors(node):
            if n not in sample:
                sample.add(n)
                to_expand.append(n)
            if len(sample) == nodes:
                break
    return nx.Graph(graph.subgraph(sample))


@op("Sample table")
def sample_table(b: core.Bundle, *, table_name: core.TableName = "meta", fraction: float = 0.1):
    b = b.copy()
    b.dfs[table_name] = b.dfs[table_name].sample(frac=fraction)
    return b


def _map_color(value):
    if pd.api.types.is_numeric_dtype(value):
        cmap = matplotlib.cm.get_cmap("viridis")
        value = (value - value.min()) / (value.max() - value.min())
        rgba = cmap(value.values)
        return [
            "#{:02x}{:02x}{:02x}".format(int(r * 255), int(g * 255), int(b * 255))
            for r, g, b in rgba[:, :3]
        ]
    else:
        cmap = matplotlib.cm.get_cmap("Paired")
        categories = pd.Index(value.unique())
        assert isinstance(cmap, matplotlib.colors.ListedColormap)
        colors = cmap.colors[: len(categories)]
        return [
            "#{:02x}{:02x}{:02x}".format(int(r * 255), int(g * 255), int(b * 255))
            for r, g, b in [colors[min(len(colors) - 1, categories.get_loc(v))] for v in value]
        ]


@op("Visualize graph", view="visualization")
def visualize_graph(
    graph: core.Bundle,
    *,
    color_nodes_by: core.NodePropertyName = None,
    label_by: core.NodePropertyName = None,
    color_edges_by: core.EdgePropertyName = None,
):
    nodes = core.df_for_frontend(graph.dfs["nodes"], 10_000)
    if color_nodes_by:
        nodes["color"] = _map_color(nodes[color_nodes_by])
    for cols in ["x y", "long lat"]:
        x, y = cols.split()
        if (
            x in nodes.columns
            and nodes[x].dtype == "float64"
            and y in nodes.columns
            and nodes[y].dtype == "float64"
        ):
            cx, cy = nodes[x].mean(), nodes[y].mean()
            dx, dy = nodes[x].std(), nodes[y].std()
            # Scale up to avoid float precision issues and because eCharts omits short edges.
            scale_x = 100 / max(dx, dy)
            scale_y = scale_x
            if y == "lat":
                scale_y *= -1
            pos = {
                node_id: ((row[x] - cx) * scale_x, (row[y] - cy) * scale_y)
                for node_id, row in nodes.iterrows()
            }
            curveness = 0  # Street maps are better with straight streets.
            break
    else:
        pos = nx.spring_layout(graph.to_nx(), iterations=max(1, int(10000 / len(nodes))))
        curveness = 0.3
    nodes = nodes.to_records()
    deduped_edges = graph.dfs["edges"].drop_duplicates(
        ["source", "target"]
    )  # ty: ignore[no-matching-overload] (https://github.com/astral-sh/ty/issues/1132)
    edges = core.df_for_frontend(deduped_edges, 10_000)
    if color_edges_by:
        edges["color"] = _map_color(edges[color_edges_by])
    edges = edges.to_records()
    v = {
        "animationDuration": 500,
        "animationEasingUpdate": "quinticInOut",
        "tooltip": {"show": True},
        "series": [
            {
                "type": "graph",
                # Mouse zoom/panning is disabled for now. It interacts badly with ReactFlow.
                # "roam": True,
                "lineStyle": {
                    "color": "gray",
                    "curveness": curveness,
                },
                "emphasis": {
                    "focus": "adjacency",
                    "lineStyle": {
                        "width": 10,
                    },
                },
                "label": {"position": "top", "formatter": "{b}"},
                "data": [
                    {
                        "id": str(n.id),
                        "x": float(pos[n.id][0]),
                        "y": float(pos[n.id][1]),
                        # Adjust node size to cover the same area no matter how many nodes there are.
                        "symbolSize": 50 / len(nodes) ** 0.5,
                        "itemStyle": {"color": n.color} if color_nodes_by else {},
                        "label": {"show": label_by is not None},
                        "name": str(getattr(n, label_by, "")) if label_by else None,
                        "value": str(getattr(n, color_nodes_by, "")) if color_nodes_by else None,
                    }
                    for n in nodes
                ],
                "links": [
                    {
                        "source": str(r.source),
                        "target": str(r.target),
                        "lineStyle": {"color": r.color} if color_edges_by else {},
                        "value": str(getattr(r, color_edges_by, "")) if color_edges_by else None,
                    }
                    for r in edges
                ],
            },
        ],
    }
    return v


@op("View tables", view="table_view")
def view_tables(bundle: core.Bundle, *, _tables_open: str = "", limit: int = 100):
    _tables_open = _tables_open  # The frontend uses this parameter to track which tables are open.
    return bundle.to_dict(limit=limit)


@op(
    "Organize",
    view="graph_creation_view",
    outputs=["output"],
)
def organize(bundles: list[core.Bundle], *, relations: str = ""):
    """Merge multiple inputs and construct graphs from the tables.

    To create a graph, import tables for edges and nodes, and combine them in this operation.
    """
    bundle = core.Bundle()
    for b in bundles:
        bundle.dfs.update(b.dfs)
        bundle.relations.extend(b.relations)
        bundle.other.update(b.other)
    if relations.strip():
        bundle.relations = [
            core.RelationDefinition(**r)  # ty: ignore[missing-argument]
            for r in json.loads(relations).values()
        ]
    return ops.Result(output=bundle, display=bundle.to_dict(limit=100))
