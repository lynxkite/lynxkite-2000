"""Graph analytics executor and data types."""

import inspect
import os
import pathlib
from lynxkite_core import ops, workspace
from lynxkite_core.executors.one_by_one import mount_gradio
import dataclasses
import functools
import networkx as nx
import pandas as pd
import polars as pl
import traceback
import typing
import urllib.parse

if typing.TYPE_CHECKING:
    import fastapi

ENV = "LynxKite Graph Analytics"

# Annotated types with format "dropdown" let you specify the available options
# as a query on the input_metadata. These query expressions are JMESPath expressions.
TableName = typing.Annotated[
    str, {"format": "dropdown", "metadata_query": "[].dataframes[].keys(@)[]"}
]
"""A type annotation to be used for parameters of an operation. TableName is
rendered as a dropdown in the frontend, listing all DataFrames in the Bundle.
The table name is passed to the operation as a string."""

NodePropertyName = typing.Annotated[
    str, {"format": "dropdown", "metadata_query": "[].dataframes[].nodes[].columns[]"}
]
"""A type annotation to be used for parameters of an operation. NodePropertyName is
rendered as a dropdown in the frontend, listing the columns of the "nodes" DataFrame.
The column name is passed to the operation as a string."""

EdgePropertyName = typing.Annotated[
    str, {"format": "dropdown", "metadata_query": "[].dataframes[].edges[].columns[]"}
]
"""A type annotation to be used for parameters of an operation. EdgePropertyName is
rendered as a dropdown in the frontend, listing the columns of the "edges" DataFrame.
The column name is passed to the operation as a string."""

OtherName = typing.Annotated[str, {"format": "dropdown", "metadata_query": "[].other.keys(@)[]"}]
"""A type annotation to be used for parameters of an operation. OtherName is
rendered as a dropdown in the frontend, listing the keys on the "other" part of the Bundle.
The key is passed to the operation as a string."""

# Parameter names in angle brackets, like <table_name>, will be replaced with the parameter
# values. (This is not part of JMESPath.)
# ColumnNameByTableName will list the columns of the DataFrame with the name
# specified by the `table_name` parameter.
ColumnNameByTableName = typing.Annotated[
    str, {"format": "dropdown", "metadata_query": "[].dataframes[].<table_name>.columns[]"}
]
"""A type annotation to be used for parameters of an operation. ColumnNameByTableName is
rendered as a dropdown in the frontend, listing the columns of the DataFrame
named by the "table_name" parameter. The column name is passed to the operation as a string."""


@dataclasses.dataclass
class RelationDefinition:
    """
    Defines a set of edges.

    Attributes:
        df: The name of the DataFrame that contains the edges.
        source_column: The column in the edge DataFrame that contains the source node ID.
        target_column: The column in the edge DataFrame that contains the target node ID.
        source_table: The name of the DataFrame that contains the source nodes.
        target_table: The name of the DataFrame that contains the target nodes.
        source_key: The column in the source table that contains the node ID.
        target_key: The column in the target table that contains the node ID.
        name: Descriptive name for the relation.
    """

    df: str
    source_column: str
    target_column: str
    source_table: str
    target_table: str
    source_key: str
    target_key: str
    name: str | None = None


@dataclasses.dataclass
class Bundle:
    """A collection of DataFrames and other data.

    Can efficiently represent a knowledge graph (homogeneous or heterogeneous) or tabular data.

    By convention, if it contains a single DataFrame, it is called `df`.
    If it contains a homogeneous graph, it is represented as two DataFrames called `nodes` and
    `edges`.

    Attributes:
        dfs: Named DataFrames.
        relations: Metadata that describes the roles of each DataFrame.
            Can be empty, if the bundle is just one or more DataFrames.
        other: Other data, such as a trained model.
    """

    dfs: dict[str, pd.DataFrame] = dataclasses.field(default_factory=dict)
    relations: list[RelationDefinition] = dataclasses.field(default_factory=list)
    other: dict[str, typing.Any] = dataclasses.field(default_factory=dict)

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
            df = self.dfs["nodes"]
            if df.index.name != "id":
                df = df.set_index("id")
            graph.add_nodes_from(df.to_dict("index").items())
        if "edges" in self.dfs:
            edges = self.dfs["edges"]
            graph.add_edges_from(
                [
                    (
                        e["source"],
                        e["target"],
                        {k: e[k] for k in edges.columns if k not in ["source", "target"]},
                    )
                    for e in edges.to_records()
                ]
            )
        return graph

    def copy(self):
        """
        Returns a shallow copy of the bundle. The Bundle and its containers are new, but
        the DataFrames and RelationDefinitions are shared. (The contents of `other` are also shared.)
        """
        return Bundle(
            dfs=dict(self.dfs),
            relations=list(self.relations),
            other=dict(self.other),
        )

    def to_dict(self, limit: int = 100):
        """JSON-serializable representation of the bundle, including some data."""
        return {
            "dataframes": {
                name: {
                    "columns": [str(c) for c in df.columns],
                    "data": df_for_frontend(df, limit).values.tolist(),
                }
                for name, df in self.dfs.items()
            },
            "relations": [dataclasses.asdict(relation) for relation in self.relations],
            "other": {k: str(v) for k, v in self.other.items()},
        }

    def metadata(self):
        """JSON-serializable information about the bundle, metadata only."""
        return {
            "dataframes": {
                name: {
                    "key": name,
                    "columns": sorted(str(c) for c in df.columns),
                }
                for name, df in self.dfs.items()
            },
            "relations": [dataclasses.asdict(relation) for relation in self.relations],
            "other": {
                k: {"key": k, **getattr(v, "metadata", lambda: {})()} for k, v in self.other.items()
            },
        }


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


def disambiguate_edges(ws: workspace.Workspace):
    """If an input plug is connected to multiple edges, keep only the last edge."""
    catalog = ops.CATALOGS[ws.env]
    nodes = {node.id: node for node in ws.nodes}
    seen = set()
    for edge in reversed(ws.edges):
        dst_node = nodes[edge.target]
        op = catalog.get(dst_node.data.op_id)
        if not op:
            continue
        t = op.get_input(edge.targetHandle).type
        if t is list or typing.get_origin(t) is list:
            # Takes multiple bundles as an input. No need to disambiguate.
            continue
        if (edge.target, edge.targetHandle) in seen:
            i = ws.edges.index(edge)
            del ws.edges[i]
            if ws._crdt:
                del ws._crdt["edges"][i]
        seen.add((edge.target, edge.targetHandle))


# Outputs are tracked by node ID and output ID.
Outputs = dict[tuple[str, str], typing.Any]


@typing.runtime_checkable
class Service(typing.Protocol):
    async def get(self, request: "fastapi.Request") -> dict:
        """Handles a GET request. The unparsed part of the URL is available as request.state.remaining_path."""
        ...

    async def post(self, request: "fastapi.Request") -> dict:
        """Handles a POST request. The unparsed part of the URL is available as request.state.remaining_path."""
        ...

    def get_description(self, url: str) -> str:
        return f"URL: [{url}]({url})"


@dataclasses.dataclass
class WorkspaceResult:
    outputs: Outputs
    services: dict[str, Service]


@ops.register_executor(ENV)
async def execute(
    ws: workspace.Workspace, ctx: workspace.WorkspaceExecutionContext | None = None
) -> WorkspaceResult:
    catalog = ops.CATALOGS[ws.env]
    disambiguate_edges(ws)
    wsres = WorkspaceResult(outputs={}, services={})
    nodes = {node.id: node for node in ws.nodes}
    todo = set(nodes.keys())
    progress = True
    while progress:
        progress = False
        for id in list(todo):
            node = nodes[id]
            inputs_done = [
                (edge.source, edge.sourceHandle) in wsres.outputs
                for edge in ws.edges
                if edge.target == id
            ]
            if all(inputs_done):
                # All inputs for this node are ready, we can compute the output.
                todo.remove(id)
                progress = True
                await _execute_node(ctx, node, ws, catalog, wsres)
    return wsres


async def await_if_needed(obj):
    if inspect.isawaitable(obj):
        obj = await obj
    return obj


def _to_bundle(x):
    if isinstance(x, nx.Graph):
        x = Bundle.from_nx(x)
    elif isinstance(x, pd.DataFrame):
        x = Bundle.from_df(x)
    assert isinstance(x, Bundle), f"Input must be a graph or dataframe. Got: {x}"
    return x


async def _execute_node(
    ctx: workspace.WorkspaceExecutionContext | None,
    node: workspace.WorkspaceNode,
    ws: workspace.Workspace,
    catalog: ops.Catalog,
    wsres: WorkspaceResult,
):
    params = {**node.data.params}
    op = catalog.get(node.data.op_id)
    if not op:
        node.publish_error("Operation not found in catalog")
        return
    node.publish_started()
    input_map = {}
    for edge in ws.edges:
        if edge.target == node.id:
            input_map.setdefault(edge.targetHandle, []).append(
                wsres.outputs[edge.source, edge.sourceHandle]
            )
    # Convert inputs types to match operation signature.
    try:
        inputs = []
        missing = []
        for p in op.inputs:
            is_list = typing.get_origin(p.type) is list
            if p.name not in input_map:
                opt_type = ops.get_optional_type(p.type)
                if opt_type is not None:
                    inputs.append(None)
                elif is_list:
                    inputs.append([])
                else:
                    missing.append(p.name)
                continue
            x = input_map[p.name]
            if p.type == list[Bundle]:
                x = [_to_bundle(i) for i in x]
            elif is_list:
                pass
            else:
                [x] = x  # There should never be multiple inputs.
            if p.type == nx.Graph:
                if isinstance(x, Bundle):
                    x = x.to_nx()
                assert isinstance(x, nx.Graph), f"Input must be a graph. Got: {x}"
            elif p.type == Bundle:
                x = _to_bundle(x)
            if p.type == pd.DataFrame:
                if isinstance(x, nx.Graph):
                    x = Bundle.from_nx(x)
                if isinstance(x, Bundle):
                    assert len(x.dfs) == 1, (
                        f"Bundle must contain a single DataFrame. Found: {sorted(x.dfs.keys())}"
                    )
                    [x] = list(x.dfs.values())
                assert isinstance(x, pd.DataFrame), f"Input must be a DataFrame. Got: {x}"
            inputs.append(x)
    except Exception as e:
        if not os.environ.get("LYNXKITE_SUPPRESS_OP_ERRORS"):
            print(f"Failed to execute node {node.id}:")
            traceback.print_exc()
        node.publish_error(e)
        return
    if missing:
        node.publish_error(f"Missing input: {', '.join(missing)}")
        return
    # Execute op.
    try:
        result = op(*inputs, **params)
        result.output = await await_if_needed(result.output)
        result.display = await await_if_needed(result.display)
    except Exception as e:
        if not os.environ.get("LYNXKITE_SUPPRESS_OP_ERRORS"):
            traceback.print_exc()
        result = ops.Result(error=str(e))
    result.input_metadata = [_get_metadata(i) for i in inputs]
    try:
        if node.type == "service":
            assert len(op.outputs) == 0, f"Unexpected outputs for service node {node.id}"
            assert isinstance(result.output, Service), (
                f"{node.id} must return a Service. Current output: {result.output}"
            )
            wsres.services[node.id] = result.output
            url = f"/api/service/lynxkite_graph_analytics/{ws.path}/{node.id}"
            url = urllib.parse.quote_plus(url)
            markdown = result.output.get_description(url)
            result.display = {
                "dataframes": {"service": {"columns": ["markdown"], "data": [[markdown]]}}
            }
            result.output = None
        elif node.type == "gradio" and result.output and ctx and ctx.app:
            url = f"/api/lynxkite_graph_analytics/{ws.path}/{node.id}"
            await mount_gradio(ctx.app, result.output, url)
            result.display = {"backend": urllib.parse.quote(url)}
            result.output = None
        elif len(op.outputs) > 1:
            assert isinstance(result.output, dict), f"Multi-output op {node.id} must return a dict"
            for k, v in result.output.items():
                wsres.outputs[node.id, k] = v
        elif len(op.outputs) == 1:
            [k] = op.outputs
            wsres.outputs[node.id, k.name] = result.output
    except Exception as e:
        if not os.environ.get("LYNXKITE_SUPPRESS_OP_ERRORS"):
            traceback.print_exc()
        result = ops.Result(error=str(e))
    node.publish_result(result)


def _get_metadata(x) -> dict:
    if hasattr(x, "metadata"):
        return x.metadata()
    return {}


def df_for_frontend(df: pd.DataFrame, limit: int) -> pd.DataFrame:
    """Returns a DataFrame with values that are safe to send to the frontend."""
    df = df[:limit]
    if isinstance(df, pl.LazyFrame):
        df = df.collect()
    if isinstance(df, pl.DataFrame):
        df = df.to_pandas()
    # Convert non-numeric columns to strings.
    for c in df.columns:
        if not pd.api.types.is_numeric_dtype(df[c]):
            df[c] = df[c].astype(str)
    return df


async def get_node_service(request: "fastapi.Request") -> Service:
    parts = request.url.path.split("/")[4:]
    cwd = pathlib.Path()
    # The workspace path (which may include slashes) is followed by the rest of the URL (which may include slashes).
    # To tell them apart, we take path elements until we find a file.
    path = cwd
    i = 0
    while path.is_dir():
        path = path / parts[i]
        i += 1
    assert path.is_relative_to(cwd), f"Path '{path}' is invalid"
    assert path.exists(), f"Workspace {path} does not exist"
    ws = workspace.Workspace.load(str(path))
    assert ws.env == ENV, f"Workspace {path} is not a LynxKite Graph Analytics workspace"
    node_id = parts[i]
    ops.load_user_scripts(str(path))
    ws.update_metadata()
    ws.normalize()
    executor = ops.EXECUTORS[ws.env]
    wsres = await executor(ws)
    [node] = [n for n in ws.nodes if n.id == node_id]
    assert not node.data.error, f"Node {node_id} has an error: {node.data.error}"
    request.state.remaining_path = "/".join(parts[i + 1 :])
    return wsres.services[node_id]


async def api_service_post(request):
    service = await get_node_service(request)
    return await service.post(request)


async def api_service_get(request):
    """
    Boxes can expose HTTP endpoints.

    Example:
      ...
      class ChatBackend:
        def get(self, request: fastapi.Request):
          return f"Hello from {request.state.remaining_path}"
        def post(self, request: fastapi.Request):
          print("POST received for", request.state.remaining_path)
      @op("Chat backend", outputs=[], view="service")
      def chat_backend(input: Bundle):
          return ChatBackend()

      curl ${LYNXKITE_URL}/api/service/lynxkite_graph_analytics/Example.lynxkite.json/Chat%20backend%201/models
    """
    service = await get_node_service(request)
    return await service.get(request)
