"""Automatically wraps all NetworkX functions as LynxKite operations."""

import collections
from lynxkite.core import ops
import functools
import inspect
import networkx as nx
import re

import pandas as pd

ENV = "LynxKite Graph Analytics"


class UnsupportedType(Exception):
    pass


nx.ladder_graph


def doc_to_type(name: str, t: str) -> type:
    t = t.lower()
    t = re.sub("[(][^)]+[)]", "", t).strip().strip(".")
    if " " in name or "http" in name:
        return None  # Not a parameter type.
    if t.endswith(", optional"):
        w = doc_to_type(name, t.removesuffix(", optional").strip())
        if w is None:
            return None
        return w | None
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
        raise UnsupportedType(t)
    elif t == '"node (optional)"':
        return None
    elif t == '"edge"':
        raise UnsupportedType(t)
    elif t == '"edge (optional)"':
        return None
    elif t in ["class", "data type"]:
        raise UnsupportedType(t)
    elif t in ["string", "str", "node label"]:
        return str
    elif t in ["string or none", "none or string", "string, or none"]:
        return str | None
    elif t in ["int", "integer"]:
        return int
    elif t in ["bool", "boolean"]:
        return bool
    elif t == "tuple":
        raise UnsupportedType(t)
    elif t == "set":
        raise UnsupportedType(t)
    elif t == "list of floats":
        raise UnsupportedType(t)
    elif t == "list of floats or float":
        return float
    elif t in ["dict", "dictionary"]:
        raise UnsupportedType(t)
    elif t == "scalar or dictionary":
        return float
    elif t == "none or dict":
        return None
    elif t in ["function", "callable"]:
        raise UnsupportedType(t)
    elif t in [
        "collection",
        "container of nodes",
        "list of nodes",
    ]:
        raise UnsupportedType(t)
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
        raise UnsupportedType(t)
    elif t == "generator of sets":
        raise UnsupportedType(t)
    elif t == "dict or a set of 2 or 3 tuples":
        raise UnsupportedType(t)
    elif t == "set of 2 or 3 tuples":
        raise UnsupportedType(t)
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
        raise UnsupportedType(t)
    return None


def types_from_doc(doc: str) -> dict[str, type]:
    types = {}
    for line in doc.splitlines():
        if ":" in line:
            a, b = line.split(":", 1)
            for a in a.split(","):
                a = a.strip()
                t = doc_to_type(a, b)
                if t is not None:
                    types[a] = t
    return types


def wrapped(name: str, func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        for k, v in kwargs.items():
            if v == "None":
                kwargs[k] = None
        res = func(*args, **kwargs)
        if isinstance(res, nx.Graph):
            return res
        # Figure out what the returned value is.
        if isinstance(res, nx.Graph):
            return res
        if isinstance(res, collections.abc.Sized):
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


def register_networkx(env: str):
    cat = ops.CATALOGS.setdefault(env, {})
    counter = 0
    for name, func in nx.__dict__.items():
        if hasattr(func, "graphs"):
            sig = inspect.signature(func)
            try:
                types = types_from_doc(func.__doc__)
            except UnsupportedType:
                continue
            for k, param in sig.parameters.items():
                if k in types:
                    continue
                if param.annotation is not param.empty:
                    types[k] = param.annotation
                if k in ["i", "j", "n"]:
                    types[k] = int
            inputs = {k: ops.Input(name=k, type=nx.Graph) for k in func.graphs}
            params = {
                name: ops.Parameter.basic(
                    name=name,
                    default=str(param.default)
                    if type(param.default) in [str, int, float]
                    else None,
                    type=types[name],
                )
                for name, param in sig.parameters.items()
                if name in types and types[name] not in [nx.Graph, nx.DiGraph]
            }
            nicename = "NX › " + name.replace("_", " ").title()
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
