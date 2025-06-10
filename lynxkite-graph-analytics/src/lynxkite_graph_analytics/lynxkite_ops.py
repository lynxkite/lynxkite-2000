"""Graph analytics operations."""

import enum
import os
import types
import fsspec
from lynxkite.core import ops
from collections import deque

from . import core
import grandcypher
import matplotlib
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
)
def import_file(
    *, file_path: str, table_name: str, file_format: FileFormat, **kwargs
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
        sep = pd.api.extensions.no_default if sep == "<auto>" else sep
        df = pd.read_csv(file_path, names=names, sep=sep)
    elif file_format == "json":
        df = pd.read_json(file_path)
    elif file_format == "parquet":
        df = pd.read_parquet(file_path)
    elif file_format == "excel":
        df = pd.read_excel(file_path, sheet_name=kwargs.get("sheet_name", "Sheet1"))
    else:
        df = ValueError(f"Unsupported file format: {file_format}")
    return core.Bundle(dfs={table_name: df})


@op("Import Parquet")
def import_parquet(*, filename: str):
    """Imports a Parquet file."""
    return pd.read_parquet(filename)


@op("Import CSV", slow=True)
def import_csv(*, filename: str, columns: str = "<from file>", separator: str = "<auto>"):
    """Imports a CSV file."""
    return pd.read_csv(
        filename,
        names=pd.api.extensions.no_default if columns == "<from file>" else columns.split(","),
        sep=pd.api.extensions.no_default if separator == "<auto>" else separator,
    )


@op("Import GraphML", slow=True)
def import_graphml(*, filename: str):
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


@op("Organize")
def organize(bundle: list[core.Bundle], *, code: ops.LongStr) -> core.Bundle:
    """Lets you rename/copy/delete DataFrames, and modify relations.

    TODO: Merge this with "Create graph".
    """
    bundle = bundle.copy()
    exec(code, globals(), {"bundle": bundle})
    return bundle


@op("Merge Bundles")
def merge_bundles(bundle_a: core.Bundle, bundle_b: core.Bundle, *, suffix: str) -> core.Bundle:
    """Merges two bundles into one.

    The resulting bundle will have all DataFrames from both bundles, and relations from both bundles.
    If a DataFrame with the same name exists in both bundles, the one from `bundle_b` will be used.
    """
    bundle = bundle_a.copy()
    # If both bundles have a DataFrame with the same name, modify the name of the one from bundle_b.
    for name in bundle_b.dfs.keys():
        if name in bundle.dfs:
            new_name = f"{name}_{suffix}"
            bundle.dfs[new_name] = bundle_b.dfs[name]
        else:
            bundle.dfs[name] = bundle_b.dfs[name]
    bundle.relations.extend(bundle_b.relations)
    return bundle


@op("Make indexed bundle")
def make_indexed_bundle(
    bundle: core.Bundle, *, get_item_function: ops.LongStr, len_function: ops.LongStr
) -> core.Bundle:
    """Converts the bundle to an indexed bundle, using the specified index column."""
    if not get_item_function:
        return bundle
    bundle = bundle.copy()
    local_vars = {}
    exec(get_item_function, globals(), local_vars)
    bundle._getitem_fn = types.MethodType(local_vars["custom_getitem"], bundle)
    exec(len_function, globals(), local_vars)
    bundle._len_fn = types.MethodType(local_vars["custom_len"], bundle)
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
        colors = cmap.colors[: len(categories)]
        return [
            "#{:02x}{:02x}{:02x}".format(int(r * 255), int(g * 255), int(b * 255))
            for r, g, b in [colors[min(len(colors) - 1, categories.get_loc(v))] for v in value]
        ]


@op("Visualize graph", view="visualization")
def visualize_graph(
    graph: core.Bundle,
    *,
    color_nodes_by: ops.NodeAttribute = None,
    label_by: ops.NodeAttribute = None,
    color_edges_by: ops.EdgeAttribute = None,
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
    edges = core.df_for_frontend(graph.dfs["edges"].drop_duplicates(["source", "target"]), 10_000)
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
    "Create graph",
    view="graph_creation_view",
    outputs=["output"],
)
def create_graph(bundle: core.Bundle, *, relations: str = None) -> core.Bundle:
    """Replace relations of the given bundle

    relations is a stringified JSON, instead of a dict, because complex Yjs types (arrays, maps)
    are not currently supported in the UI.

    Args:
        bundle: Bundle to modify
        relations (str, optional): Set of relations to set for the bundle. The parameter
            should be a JSON object where the keys are relation names and the values are
            a dictionary representation of a `RelationDefinition`.
            Defaults to None.

    Returns:
        Bundle: The input bundle with the new relations set.
    """
    bundle = bundle.copy()
    if not (relations is None or relations.strip() == ""):
        bundle.relations = [core.RelationDefinition(**r) for r in json.loads(relations).values()]
    return ops.Result(output=bundle, display=bundle.to_dict(limit=100))


@op("Biomedical foundation graph (PLACEHOLDER)")
def biomedical_foundation_graph(*, filter_nodes: str):
    """Loads the gigantic Lynx-maintained knowledge graph. Includes drugs, diseases, genes, proteins, etc."""
    return None


@op("Test indexed bundle")
def test_indexed_bundle(bundle: core.Bundle, *, index: int) -> ops.Result:
    """Tests the indexed bundle by getting an item at the specified index."""
    if not hasattr(bundle, "_getitem_fn"):
        return ops.Result(bundle, error="Bundle is not indexed.")
    try:
        print(f"Len is {len(bundle)}")
        item = bundle[index]
        print(item)
        return ops.Result(item)
    except Exception:
        # Return the full stack trace in the error message for debugging.
        import traceback

        error_message = traceback.format_exc()
        return ops.Result(bundle, error=str(error_message))
