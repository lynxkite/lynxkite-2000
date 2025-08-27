"""Automatically wraps all NetworkX functions as LynxKite operations."""

from lynxkite_core import ops
import collections.abc
import enum
import functools
import inspect
import networkx as nx
import pandas as pd
import re
import types

ENV = "LynxKite Graph Analytics"


class UnsupportedParameterType(Exception):
    pass


class Failure(enum.StrEnum):
    UNSUPPORTED = "unsupported"  # This parameter will be hidden.
    SKIP = "skip"  # We have to skip the whole function.


def doc_to_type(name: str, type_hint: str) -> type | types.UnionType | Failure:
    type_hint = type_hint.lower()
    type_hint = re.sub("[(][^)]+[)]", "", type_hint).strip().strip(".")
    if " " in name or "http" in name:
        return Failure.UNSUPPORTED  # Not a parameter type.
    if type_hint.endswith(", optional"):
        w = doc_to_type(name, type_hint.removesuffix(", optional").strip())
        if w is Failure.UNSUPPORTED or w is Failure.SKIP:
            return Failure.SKIP
        assert not isinstance(w, Failure)
        return w | None
    if type_hint in [
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
    ] or type_hint.startswith("networkx graph"):
        return nx.Graph
    elif type_hint in [
        "digraph-like",
        "digraph",
        "directed graph",
        "networkx digraph",
        "networkx directed graph",
        "nx.digraph",
    ]:
        return nx.DiGraph
    elif type_hint == "node":
        return Failure.UNSUPPORTED
    elif type_hint == '"node (optional)"':
        return Failure.SKIP
    elif type_hint == '"edge"':
        return Failure.UNSUPPORTED
    elif type_hint == '"edge (optional)"':
        return Failure.SKIP
    elif type_hint in ["class", "data type"]:
        return Failure.UNSUPPORTED
    elif type_hint in ["string", "str", "node label"]:
        return str
    elif type_hint in ["string or none", "none or string", "string, or none"]:
        return str | None
    elif type_hint in ["int", "integer"]:
        return int
    elif type_hint in ["bool", "boolean"]:
        return bool
    elif type_hint == "tuple":
        return Failure.UNSUPPORTED
    elif type_hint == "set":
        return Failure.UNSUPPORTED
    elif type_hint == "list of floats":
        return Failure.UNSUPPORTED
    elif type_hint == "list of floats or float":
        return float
    elif type_hint in ["dict", "dictionary"]:
        return Failure.UNSUPPORTED
    elif type_hint == "scalar or dictionary":
        return float
    elif type_hint == "none or dict":
        return Failure.SKIP
    elif type_hint in ["function", "callable"]:
        return Failure.UNSUPPORTED
    elif type_hint in [
        "collection",
        "container of nodes",
        "list of nodes",
    ]:
        return Failure.UNSUPPORTED
    elif type_hint in [
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
        return Failure.UNSUPPORTED
    elif type_hint == "generator of sets":
        return Failure.UNSUPPORTED
    elif type_hint == "dict or a set of 2 or 3 tuples":
        return Failure.UNSUPPORTED
    elif type_hint == "set of 2 or 3 tuples":
        return Failure.UNSUPPORTED
    elif type_hint == "none, string or function":
        return str | None
    elif type_hint == "string or function" and name == "weight":
        return str
    elif type_hint == "integer, float, or none":
        return float | None
    elif type_hint in [
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
    elif type_hint in ["integer or none", "int or none"]:
        return int | None
    elif name == "seed":
        return int | None
    elif name == "weight":
        return str
    elif type_hint == "object":
        return Failure.UNSUPPORTED
    return Failure.SKIP


def types_from_doc(doc: str) -> dict[str, type | types.UnionType | Failure]:
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
    async def wrapper(*args, **kwargs):
        for k, v in kwargs.items():
            if v == "None":
                kwargs[k] = None
        res = await ops.make_async(func)(*args, **kwargs)
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


def _get_params(func) -> list[ops.Parameter | ops.ParameterGroup]:
    sig = inspect.signature(func)
    # Get types from docstring.
    types = types_from_doc(func.__doc__)
    # Always hide these.
    for k in ["backend", "backend_kwargs", "create_using"]:
        types[k] = Failure.SKIP
    # Add in types based on signature.
    for k, param in sig.parameters.items():
        if k in types:
            continue
        if param.annotation is not param.empty:
            types[k] = param.annotation
        if k in ["i", "j", "n"]:
            types[k] = int
    params = []
    for name, param in sig.parameters.items():
        _type = types.get(name, Failure.UNSUPPORTED)
        if _type is Failure.UNSUPPORTED:
            raise UnsupportedParameterType(name)
        if _type is Failure.SKIP or _type in [nx.Graph, nx.DiGraph]:
            continue
        p = ops.Parameter.basic(
            name=name,
            default=str(param.default) if type(param.default) in [str, int, float] else None,
            type=_type,
        )
        params.append(p)
    return params


_REPLACEMENTS = [
    (" at free", " AT-free"),
    (" dag", " DAG"),
    (" k out ", " k-out "),
    (" rary", " r-ary"),
    ("2d ", "2D "),
    ("3d ", "3D "),
    ("adamic adar", "Adamic–Adar"),
    ("barabasi albert", "Barabasi–Albert"),
    ("bellman ford", "Bellman–Ford"),
    ("bethe hessian", "Bethe–Hessian"),
    ("bfs", "BFS"),
    ("d separator", "d-separator"),
    ("dag ", "DAG "),
    ("dfs", "DFS"),
    ("dijkstra", "Dijkstra"),
    ("dorogovtsev goltsev mendes", "Dorogovtsev–Goltsev–Mendes"),
    ("erdos renyi", "Erdos–Renyi"),
    ("euler", "Euler"),
    ("floyd warshall", "Floyd–Warshall"),
    ("forceatlas2", "ForceAtlas2"),
    ("gexf ", "GEXF "),
    ("gml", "GML"),
    ("gnc", "G(n,c)"),
    ("gnm", "G(n,m)"),
    ("gnp", "G(n,p)"),
    ("gnr", "G(n,r)"),
    ("graphml", "GraphML"),
    ("harary", "Harary"),
    ("havel hakimi", "Havel–Hakimi"),
    ("hkn", "H(k,n)"),
    ("hnm", "H(n,m)"),
    ("internet", "Internet"),
    ("k core", "k-core"),
    ("k corona", "k-corona"),
    ("k crust", "k-crust"),
    ("k shell", "k-shell"),
    ("k truss", "k-truss"),
    ("kl ", "KL "),
    ("laplacian", "Laplacian"),
    ("lfr ", "LFR "),
    ("margulis gabber galil", "Margulis–Gabber–Galil"),
    ("moebius kantor", "Moebius–Kantor"),
    ("newman watts strogatz", "Newman–Watts–Strogatz"),
    ("numpy", "NumPy"),
    ("pagerank", "PageRank"),
    ("pajek", "Pajek"),
    ("pandas", "Pandas"),
    ("parse leda", "Parse LEDA"),
    ("powerlaw", "power-law"),
    ("prufer", "Prüfer"),
    ("radzik", "Radzik"),
    ("s metric", "s-metric"),
    ("scale free", "Scale-free"),
    ("scipy", "SciPy"),
    ("small world", "small-world"),
    ("soundarajan hopcroft", "Soundarajan–Hopcroft"),
    ("southern women", "Southern women"),
    ("vf2pp", "VF2++"),
    ("watts strogatz", "Watts–Strogatz"),
    ("weisfeiler lehman", "Weisfeiler–Lehman"),
]
_CATEGORY_REPLACEMENTS = [
    ("Networkx", "NetworkX"),
    ("D separation", "D-separation"),
    ("Dag", "DAG"),
    ("Pagerank alg", "PageRank alg"),
    ("Richclub", "Rich-club"),
    ("Smallworld", "Small-world"),
    ("Smetric", "S-metric"),
    ("Structuralholes", "Structural holes"),
    ("Edgedfs", "Edge DFS"),
    ("Edgebfs", "Edge BFS"),
    ("Edge_kcomponents", "Edge k-components"),
    ("Mincost", "Min cost"),
    ("Networksimplex", "Network simplex"),
    ("Vf2pp", "VF2++"),
    ("Mst", "MST"),
    ("Attrmatrix", "Attr matrix"),
    ("Graphmatrix", "Graph matrix"),
    ("Laplacianmatrix", "Laplacian matrix"),
    ("Algebraicconnectivity", "Algebraic connectivity"),
    ("Modularitymatrix", "Modularity matrix"),
    ("Bethehessianmatrix", "Bethe–Hessian matrix"),
]


def _categories(func) -> list[str]:
    """Extract categories from the function's docstring."""
    path = func.__module__.split(".")
    cats = []
    for p in path:
        p = p.replace("_", " ").capitalize()
        for a, b in _CATEGORY_REPLACEMENTS:
            p = p.replace(a, b)
        cats.append(p)
    return cats


def register_networkx(env: str):
    cat = ops.CATALOGS.setdefault(env, {})
    counter = 0
    for name, func in nx.__dict__.items():
        if hasattr(func, "graphs"):
            try:
                params = _get_params(func)
            except UnsupportedParameterType:
                continue
            inputs = [
                ops.Input(name=k, type=nx.Graph, position=ops.Position.LEFT) for k in func.graphs
            ]
            nicename = name.replace("_", " ")
            for a, b in _REPLACEMENTS:
                nicename = nicename.replace(a, b)
            if nicename[1] != "-":
                nicename = nicename[0].upper() + nicename[1:]
            op = ops.Op(
                func=wrapped(name, func),
                name=nicename,
                categories=_categories(func),
                params=params,
                inputs=inputs,
                outputs=[ops.Output(name="output", type=nx.Graph, position=ops.Position.RIGHT)],
                type="basic",
            )
            cat[op.id] = op
            counter += 1
    print(f"Registered {counter} NetworkX operations.")


register_networkx(ENV)
