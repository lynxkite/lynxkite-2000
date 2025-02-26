"""Graph analytics executor and data types."""

from lynxkite.core import ops
import dataclasses
import functools
import networkx as nx
import pandas as pd
import polars as pl
import traceback
import typing


ENV = "LynxKite Graph Analytics"


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
    name: str | None = None  # Descriptive name for the relation.


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
                        {
                            k: e[k]
                            for k in edges.columns
                            if k not in ["source", "target"]
                        },
                    )
                    for e in edges.to_records()
                ]
            )
        return graph

    def copy(self):
        """Returns a medium depth copy of the bundle. The Bundle is completely new, but the DataFrames and RelationDefinitions are shared."""
        return Bundle(
            dfs=dict(self.dfs),
            relations=list(self.relations),
            other=dict(self.other) if self.other else None,
        )

    def to_dict(self, limit: int = 100):
        return {
            "dataframes": {
                name: {
                    "columns": [str(c) for c in df.columns],
                    "data": df_for_frontend(df, limit).values.tolist(),
                }
                for name, df in self.dfs.items()
            },
            "relations": [dataclasses.asdict(relation) for relation in self.relations],
            "other": self.other,
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


def disambiguate_edges(ws):
    """If an input plug is connected to multiple edges, keep only the last edge."""
    seen = set()
    for edge in reversed(ws.edges):
        if (edge.target, edge.targetHandle) in seen:
            ws.edges.remove(edge)
        seen.add((edge.target, edge.targetHandle))


@ops.register_executor(ENV)
async def execute(ws):
    catalog: dict[str, ops.Op] = ops.CATALOGS[ws.env]
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
                params = {**node.data.params}
                op = catalog.get(node.data.title)
                if not op:
                    node.publish_error("Operation not found in catalog")
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
                    result = op(*inputs, **params)
                except Exception as e:
                    traceback.print_exc()
                    node.publish_error(e)
                    failed += 1
                    continue
                outputs[node.id] = result.output
                node.publish_result(result)


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
