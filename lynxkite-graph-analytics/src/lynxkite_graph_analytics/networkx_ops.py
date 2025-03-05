"""Automatically wraps all NetworkX functions as LynxKite operations."""

import collections
import types
from lynxkite.core import ops
import functools
import inspect
import networkx as nx
import re

import pandas as pd

ENV = "LynxKite Graph Analytics"


class UnsupportedParameterType(Exception):
    pass


_UNSUPPORTED = object()
_SKIP = object()

nx.ladder_graph


def doc_to_type(name: str, t: str) -> type:
    t = t.lower()
    t = re.sub("[(][^)]+[)]", "", t).strip().strip(".")
    if " " in name or "http" in name:
        return _UNSUPPORTED  # Not a parameter type.
    if t.endswith(", optional"):
        w = doc_to_type(name, t.removesuffix(", optional").strip())
        if w is _UNSUPPORTED:
            return _SKIP
        return w if w is _SKIP else w | None
    if t in [
        "a digraph or multidigraph",
        "a graph g",
        "graph",
        "graphs",
        "networkx graph instance",
        "networkx graph",
        "networkx undirected graph",
        "nx.graph",
        "undirected graph",
        "undirected networkx graph",
    ] or t.startswith("networkx graph"):
        return nx.Graph
    elif t in [
        "digraph-like",
        "digraph",
        "directed graph",
        "networkx digraph",
        "networkx directed graph",
        "nx.digraph",
    ]:
        return nx.DiGraph
    elif t == "node":
        return _UNSUPPORTED
    elif t == '"node (optional)"':
        return _SKIP
    elif t == '"edge"':
        return _UNSUPPORTED
    elif t == '"edge (optional)"':
        return _SKIP
    elif t in ["class", "data type"]:
        return _UNSUPPORTED
    elif t in ["string", "str", "node label"]:
        return str
    elif t in ["string or none", "none or string", "string, or none"]:
        return str | None
    elif t in ["int", "integer"]:
        return int
    elif t in ["bool", "boolean"]:
        return bool
    elif t == "tuple":
        return _UNSUPPORTED
    elif t == "set":
        return _UNSUPPORTED
    elif t == "list of floats":
        return _UNSUPPORTED
    elif t == "list of floats or float":
        return float
    elif t in ["dict", "dictionary"]:
        return _UNSUPPORTED
    elif t == "scalar or dictionary":
        return float
    elif t == "none or dict":
        return _SKIP
    elif t in ["function", "callable"]:
        return _UNSUPPORTED
    elif t in [
        "collection",
        "container of nodes",
        "list of nodes",
    ]:
        return _UNSUPPORTED
    elif t in [
        "container",
        "generator",
        "iterable",
        "iterator",
        "list or iterable container",
        "list or iterable",
        "list or set",
        "list or tuple",
        "list",
    ]:
        return _UNSUPPORTED
    elif t == "generator of sets":
        return _UNSUPPORTED
    elif t == "dict or a set of 2 or 3 tuples":
        return _UNSUPPORTED
    elif t == "set of 2 or 3 tuples":
        return _UNSUPPORTED
    elif t == "none, string or function":
        return str | None
    elif t == "string or function" and name == "weight":
        return str
    elif t == "integer, float, or none":
        return float | None
    elif t in [
        "float",
        "int or float",
        "integer or float",
        "integer, float",
        "number",
        "numeric",
        "real",
        "scalar",
    ]:
        return float
    elif t in ["integer or none", "int or none"]:
        return int | None
    elif name == "seed":
        return int | None
    elif name == "weight":
        return str
    elif t == "object":
        return _UNSUPPORTED
    return _SKIP


def types_from_doc(doc: str) -> dict[str, type]:
    types = {}
    for line in doc.splitlines():
        if ":" in line:
            a, b = line.split(":", 1)
            for a in a.split(","):
                a = a.strip()
                types[a] = doc_to_type(a, b)
    return types


def wrapped(name: str, func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        for k, v in kwargs.items():
            if v == "None":
                kwargs[k] = None
        res = func(*args, **kwargs)
        # Figure out what the returned value is.
        if isinstance(res, nx.Graph):
            return res
        if isinstance(res, types.GeneratorType):
            res = list(res)
        if name in ["articulation_points"]:
            graph = args[0].copy()
            nx.set_node_attributes(graph, 0, name=name)
            nx.set_node_attributes(graph, {r: 1 for r in res}, name=name)
            return graph
        if isinstance(res, collections.abc.Sized):
            if len(res) == 0:
                return pd.DataFrame()
            for a in args:
                if isinstance(a, nx.Graph):
                    if a.number_of_nodes() == len(res):
                        graph = a.copy()
                        nx.set_node_attributes(graph, values=res, name=name)
                        return graph
                    if a.number_of_edges() == len(res):
                        graph = a.copy()
                        nx.set_edge_attributes(graph, values=res, name=name)
                        return graph
            return pd.DataFrame({name: res})
        return pd.DataFrame({name: [res]})

    return wrapper


def _get_params(func) -> dict | None:
    sig = inspect.signature(func)
    # Get types from docstring.
    types = types_from_doc(func.__doc__)
    # Always hide these.
    for k in ["backend", "backend_kwargs", "create_using"]:
        types[k] = _SKIP
    # Add in types based on signature.
    for k, param in sig.parameters.items():
        if k in types:
            continue
        if param.annotation is not param.empty:
            types[k] = param.annotation
        if k in ["i", "j", "n"]:
            types[k] = int
    params = {}
    for name, param in sig.parameters.items():
        _type = types.get(name, _UNSUPPORTED)
        if _type is _UNSUPPORTED:
            raise UnsupportedParameterType(name)
        if _type is _SKIP or _type in [nx.Graph, nx.DiGraph]:
            continue
        params[name] = ops.Parameter.basic(
            name=name,
            default=str(param.default)
            if type(param.default) in [str, int, float]
            else None,
            type=_type,
        )
    return params


_REPLACEMENTS = [
    ("Barabasi Albert", "Barabasi–Albert"),
    ("Bellman Ford", "Bellman–Ford"),
    ("Bethe Hessian", "Bethe–Hessian"),
    ("Bfs", "BFS"),
    ("Dag ", "DAG "),
    ("Dfs", "DFS"),
    ("Dorogovtsev Goltsev Mendes", "Dorogovtsev–Goltsev–Mendes"),
    ("Erdos Renyi", "Erdos–Renyi"),
    ("Floyd Warshall", "Floyd–Warshall"),
    ("Gnc", "G(n,c)"),
    ("Gnm", "G(n,m)"),
    ("Gnp", "G(n,p)"),
    ("Gnr", "G(n,r)"),
    ("Havel Hakimi", "Havel–Hakimi"),
    ("Hkn", "H(k,n)"),
    ("Hnm", "H(n,m)"),
    ("Kl ", "KL "),
    ("Moebius Kantor", "Moebius–Kantor"),
    ("Pagerank", "PageRank"),
    ("Scale Free", "Scale-Free"),
    ("Vf2Pp", "VF2++"),
    ("Watts Strogatz", "Watts–Strogatz"),
    ("Weisfeiler Lehman", "Weisfeiler–Lehman"),
]


def register_networkx(env: str):
    cat = ops.CATALOGS.setdefault(env, {})
    counter = 0
    for name, func in nx.__dict__.items():
        if hasattr(func, "graphs"):
            try:
                params = _get_params(func)
            except UnsupportedParameterType:
                continue
            inputs = {k: ops.Input(name=k, type=nx.Graph) for k in func.graphs}
            nicename = "NX › " + name.replace("_", " ").title()
            for a, b in _REPLACEMENTS:
                nicename = nicename.replace(a, b)
            op = ops.Op(
                func=wrapped(name, func),
                name=nicename,
                params=params,
                inputs=inputs,
                outputs={"output": ops.Output(name="output", type=nx.Graph)},
                type="basic",
            )
            cat[nicename] = op
            counter += 1
    print(f"Registered {counter} NetworkX operations.")


register_networkx(ENV)
