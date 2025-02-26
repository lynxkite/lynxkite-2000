"""Graph analytics operations."""

import os
import fsspec
from lynxkite.core import ops
from collections import deque
from . import core
import grandcypher
import joblib
import matplotlib
import networkx as nx
import pandas as pd
import polars as pl
import json


mem = joblib.Memory("../joblib-cache")
op = ops.op_registration(core.ENV)


@op("Import Parquet")
def import_parquet(*, filename: str):
    """Imports a Parquet file."""
    return pd.read_parquet(filename)


@mem.cache
@op("Import CSV")
def import_csv(
    *, filename: str, columns: str = "<from file>", separator: str = "<auto>"
):
    """Imports a CSV file."""
    return pd.read_csv(
        filename,
        names=pd.api.extensions.no_default
        if columns == "<from file>"
        else columns.split(","),
        sep=pd.api.extensions.no_default if separator == "<auto>" else separator,
    )


@mem.cache
@op("Import GraphML")
def import_graphml(*, filename: str):
    """Imports a GraphML file."""
    files = fsspec.open_files(filename, compression="infer")
    for f in files:
        if ".graphml" in f.path:
            with f as f:
                return nx.read_graphml(f)
    raise ValueError(f"No .graphml file found at {filename}")


@mem.cache
@op("Graph from OSM")
def import_osm(*, location: str):
    import osmnx as ox

    return ox.graph.graph_from_place(location, network_type="drive")


@op("Create scale-free graph")
def create_scale_free_graph(*, nodes: int = 10):
    """Creates a scale-free graph with the given number of nodes."""
    return nx.scale_free_graph(nodes)


@op("Compute PageRank")
@core.nx_node_attribute_func("pagerank")
def compute_pagerank(graph: nx.Graph, *, damping=0.85, iterations=100):
    # TODO: This requires scipy to be installed.
    return nx.pagerank(graph, alpha=damping, max_iter=iterations)


@op("Compute betweenness centrality")
@core.nx_node_attribute_func("betweenness_centrality")
def compute_betweenness_centrality(graph: nx.Graph, *, k=10):
    return nx.betweenness_centrality(graph, k=k)


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
            res = (
                pl.SQLContext(bundle.dfs)
                .execute(query)
                .collect(engine="gpu")
                .to_pandas()
            )
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


@op("Organize bundle")
def organize_bundle(bundle: core.Bundle, *, code: ops.LongStr):
    """Lets you rename/copy/delete DataFrames, and modify relations.

    TODO: Use a declarative solution instead of Python code. Add UI.
    """
    bundle = bundle.copy()
    exec(code, globals(), {"bundle": bundle})
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
            for r, g, b in [
                colors[min(len(colors) - 1, categories.get_loc(v))] for v in value
            ]
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
        pos = nx.spring_layout(
            graph.to_nx(), iterations=max(1, int(10000 / len(nodes)))
        )
        curveness = 0.3
    nodes = nodes.to_records()
    edges = core.df_for_frontend(
        graph.dfs["edges"].drop_duplicates(["source", "target"]), 10_000
    )
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
                        "value": str(getattr(n, color_nodes_by, ""))
                        if color_nodes_by
                        else None,
                    }
                    for n in nodes
                ],
                "links": [
                    {
                        "source": str(r.source),
                        "target": str(r.target),
                        "lineStyle": {"color": r.color} if color_edges_by else {},
                        "value": str(getattr(r, color_edges_by, ""))
                        if color_edges_by
                        else None,
                    }
                    for r in edges
                ],
            },
        ],
    }
    return v


@op("View tables", view="table_view")
def view_tables(bundle: core.Bundle, *, limit: int = 100):
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
        bundle.relations = [
            core.RelationDefinition(**r) for r in json.loads(relations).values()
        ]
    return ops.Result(output=bundle, display=bundle.to_dict(limit=100))
