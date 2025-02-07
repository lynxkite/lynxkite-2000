"""Graph analytics operations. To be split into separate files when we have more."""

import os
from lynxkite.core import ops, workspace
from collections import deque
import dataclasses
import functools
import grandcypher
import joblib
import matplotlib
import networkx as nx
import pandas as pd
import polars as pl
import traceback
import typing
import zipfile

mem = joblib.Memory("../joblib-cache")
ENV = "LynxKite Graph Analytics"
op = ops.op_registration(ENV)


@dataclasses.dataclass
class RelationDefinition:
    """Defines a set of edges."""

    df: str  # The DataFrame that contains the edges.
    source_column: (
        str  # The column in the edge DataFrame that contains the source node ID.
    )
    target_column: (
        str  # The column in the edge DataFrame that contains the target node ID.
    )
    source_table: str  # The DataFrame that contains the source nodes.
    target_table: str  # The DataFrame that contains the target nodes.
    source_key: str  # The column in the source table that contains the node ID.
    target_key: str  # The column in the target table that contains the node ID.


@dataclasses.dataclass
class Bundle:
    """A collection of DataFrames and other data.

    Can efficiently represent a knowledge graph (homogeneous or heterogeneous) or tabular data.
    It can also carry other data, such as a trained model.
    """

    dfs: dict[str, pd.DataFrame] = dataclasses.field(default_factory=dict)
    relations: list[RelationDefinition] = dataclasses.field(default_factory=list)
    other: dict[str, typing.Any] = None

    @classmethod
    def from_nx(cls, graph: nx.Graph):
        edges = nx.to_pandas_edgelist(graph)
        d = dict(graph.nodes(data=True))
        nodes = pd.DataFrame(d.values(), index=d.keys())
        nodes["id"] = nodes.index
        if "index" in nodes.columns:
            nodes.drop(columns=["index"], inplace=True)
        return cls(
            dfs={"edges": edges, "nodes": nodes},
            relations=[
                RelationDefinition(
                    df="edges",
                    source_column="source",
                    target_column="target",
                    source_table="nodes",
                    target_table="nodes",
                    source_key="id",
                    target_key="id",
                )
            ],
        )

    @classmethod
    def from_df(cls, df: pd.DataFrame):
        return cls(dfs={"df": df})

    def to_nx(self):
        # TODO: Use relations.
        graph = nx.DiGraph()
        if "nodes" in self.dfs:
            graph.add_nodes_from(
                self.dfs["nodes"].set_index("id").to_dict("index").items()
            )
        graph.add_edges_from(
            self.dfs["edges"][["source", "target"]].itertuples(index=False, name=None)
        )
        return graph

    def copy(self):
        """Returns a medium depth copy of the bundle. The Bundle is completely new, but the DataFrames and RelationDefinitions are shared."""
        return Bundle(
            dfs=dict(self.dfs),
            relations=list(self.relations),
            other=dict(self.other) if self.other else None,
        )


def nx_node_attribute_func(name):
    """Decorator for wrapping a function that adds a NetworkX node attribute."""

    def decorator(func):
        @functools.wraps(func)
        def wrapper(graph: nx.Graph, **kwargs):
            graph = graph.copy()
            attr = func(graph, **kwargs)
            nx.set_node_attributes(graph, attr, name)
            return graph

        return wrapper

    return decorator


def disambiguate_edges(ws):
    """If an input plug is connected to multiple edges, keep only the last edge."""
    seen = set()
    for edge in reversed(ws.edges):
        if (edge.target, edge.targetHandle) in seen:
            ws.edges.remove(edge)
        seen.add((edge.target, edge.targetHandle))


@ops.register_executor(ENV)
async def execute(ws):
    catalog: dict[str, ops.Op] = ops.CATALOGS[ENV]
    disambiguate_edges(ws)
    outputs = {}
    failed = 0
    while len(outputs) + failed < len(ws.nodes):
        for node in ws.nodes:
            if node.id in outputs:
                continue
            # TODO: Take the input/output handles into account.
            inputs = [edge.source for edge in ws.edges if edge.target == node.id]
            if all(input in outputs for input in inputs):
                # All inputs for this node are ready, we can compute the output.
                inputs = [outputs[input] for input in inputs]
                data = node.data
                params = {**data.params}
                op = catalog.get(data.title)
                if not op:
                    data.error = "Operation not found in catalog"
                    failed += 1
                    continue
                try:
                    # Convert inputs types  to match operation signature.
                    for i, (x, p) in enumerate(zip(inputs, op.inputs.values())):
                        if p.type == nx.Graph and isinstance(x, Bundle):
                            inputs[i] = x.to_nx()
                        elif p.type == Bundle and isinstance(x, nx.Graph):
                            inputs[i] = Bundle.from_nx(x)
                        elif p.type == Bundle and isinstance(x, pd.DataFrame):
                            inputs[i] = Bundle.from_df(x)
                    output = op(*inputs, **params)
                except Exception as e:
                    traceback.print_exc()
                    data.error = str(e)
                    failed += 1
                    continue
                if len(op.inputs) == 1 and op.inputs.get("multi") == "*":
                    # It's a flexible input. Create n+1 handles.
                    data.inputs = {f"input{i}": None for i in range(len(inputs) + 1)}
                data.error = None
                outputs[node.id] = output
                if (
                    op.type == "visualization"
                    or op.type == "table_view"
                    or op.type == "image"
                ):
                    data.display = output


@op("Import Parquet")
def import_parquet(*, filename: str):
    """Imports a Parquet file."""
    return pd.read_parquet(filename)


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


@op("Import GraphML")
@mem.cache
def import_graphml(*, filename: str):
    """Imports a GraphML file."""
    if filename.endswith(".zip"):
        with zipfile.ZipFile(filename, "r") as z:
            for fn in z.namelist():
                if fn.endswith(".graphml"):
                    with z.open(fn) as f:
                        G = nx.read_graphml(f)
                        break
            else:
                raise ValueError("No GraphML file found in the ZIP archive.")
    else:
        G = nx.read_graphml(filename)
    return G


@op("Graph from OSM")
@mem.cache
def import_osm(*, location: str):
    import osmnx as ox

    return ox.graph.graph_from_place(location, network_type="drive")


@op("Create scale-free graph")
def create_scale_free_graph(*, nodes: int = 10):
    """Creates a scale-free graph with the given number of nodes."""
    return nx.scale_free_graph(nodes)


@op("Compute PageRank")
@nx_node_attribute_func("pagerank")
def compute_pagerank(graph: nx.Graph, *, damping=0.85, iterations=100):
    # TODO: This requires scipy to be installed.
    return nx.pagerank(graph, alpha=damping, max_iter=iterations)


@op("Compute betweenness centrality")
@nx_node_attribute_func("betweenness_centrality")
def compute_betweenness_centrality(graph: nx.Graph, *, k=10):
    return nx.betweenness_centrality(graph, k=k, backend="cugraph")


@op("Discard loop edges")
def discard_loop_edges(graph: nx.Graph):
    graph = graph.copy()
    graph.remove_edges_from(nx.selfloop_edges(graph))
    return graph


@op("Discard parallel edges")
def discard_parallel_edges(graph: nx.Graph):
    return nx.DiGraph(graph)


@op("SQL")
def sql(bundle: Bundle, *, query: ops.LongStr, save_as: str = "result"):
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
def cypher(bundle: Bundle, *, query: ops.LongStr, save_as: str = "result"):
    """Run a Cypher query on the graph in the bundle. Save the results as a new DataFrame."""
    bundle = bundle.copy()
    graph = bundle.to_nx()
    res = grandcypher.GrandCypher(graph).run(query)
    bundle.dfs[save_as] = pd.DataFrame(res)
    return bundle


@op("Organize bundle")
def organize_bundle(bundle: Bundle, *, code: ops.LongStr):
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
        rgba = cmap(value)
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
def visualize_graph(graph: Bundle, *, color_nodes_by: ops.NodeAttribute = None):
    nodes = graph.dfs["nodes"].copy()
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
    edges = graph.dfs["edges"].drop_duplicates(["source", "target"])
    edges = edges.to_records()
    v = {
        "animationDuration": 500,
        "animationEasingUpdate": "quinticInOut",
        "series": [
            {
                "type": "graph",
                "roam": True,
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
                "data": [
                    {
                        "id": str(n.id),
                        "x": float(pos[n.id][0]),
                        "y": float(pos[n.id][1]),
                        # Adjust node size to cover the same area no matter how many nodes there are.
                        "symbolSize": 50 / len(nodes) ** 0.5,
                        "itemStyle": {"color": n.color} if color_nodes_by else {},
                    }
                    for n in nodes
                ],
                "links": [
                    {"source": str(r.source), "target": str(r.target)} for r in edges
                ],
            },
        ],
    }
    return v


def collect(df: pd.DataFrame):
    if isinstance(df, pl.LazyFrame):
        df = df.collect()
    if isinstance(df, pl.DataFrame):
        return [[d[c] for c in df.columns] for d in df.to_dicts()]
    return df.values.tolist()


@op("View tables", view="table_view")
def view_tables(bundle: Bundle, *, limit: int = 100):
    v = {
        "dataframes": {
            name: {
                "columns": [str(c) for c in df.columns],
                "data": collect(df)[:limit],
            }
            for name, df in bundle.dfs.items()
        },
        "relations": bundle.relations,
        "other": bundle.other,
    }
    return v
