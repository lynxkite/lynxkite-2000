"""Automatically wraps all NetworkX functions as LynxKite operations."""

from lynxkite.core import ops
import functools
import inspect
import networkx as nx

ENV = "LynxKite Graph Analytics"


def wrapped(name: str, func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        for k, v in kwargs.items():
            if v == "None":
                kwargs[k] = None
        res = func(*args, **kwargs)
        if isinstance(res, nx.Graph):
            return res
        # Otherwise it's a node attribute.
        graph = args[0].copy()
        nx.set_node_attributes(graph, values=res, name=name)
        return graph

    return wrapper


def register_networkx(env: str):
    cat = ops.CATALOGS.setdefault(env, {})
    for name, func in nx.__dict__.items():
        if hasattr(func, "graphs"):
            sig = inspect.signature(func)
            inputs = {k: ops.Input(name=k, type=nx.Graph) for k in func.graphs}
            params = {
                name: ops.Parameter.basic(
                    name,
                    str(param.default)
                    if type(param.default) in [str, int, float]
                    else None,
                    param.annotation,
                )
                for name, param in sig.parameters.items()
                if name not in ["G", "backend", "backend_kwargs", "create_using"]
            }
            for p in params.values():
                if not p.type:
                    # Guess the type based on the name.
                    if len(p.name) == 1:
                        p.type = int
            name = "NX › " + name.replace("_", " ").title()
            op = ops.Op(
                func=wrapped(name, func),
                name=name,
                params=params,
                inputs=inputs,
                outputs={"output": ops.Output(name="output", type=nx.Graph)},
                type="basic",
            )
            cat[name] = op


register_networkx(ENV)
