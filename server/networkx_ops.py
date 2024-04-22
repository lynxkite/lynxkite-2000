"""Automatically wraps all NetworkX functions as LynxKite operations."""
from . import ops
import functools
import inspect
import networkx as nx


def wrapped(func):
  @functools.wraps(func)
  def wrapper(*args, **kwargs):
    for k, v in kwargs.items():
      if v == 'None':
        kwargs[k] = None
    res = func(*args, **kwargs)
    if isinstance(res, nx.Graph):
      return res
    # Otherwise it's a node attribute.
    graph = args[0].copy()
    nx.set_node_attributes(graph, 'attr', name)
    return graph
  return wrapper


for (name, func) in nx.__dict__.items():
  if type(func) == nx.utils.backends._dispatch:
    sig = inspect.signature(func)
    inputs = {'G': nx.Graph} if 'G' in sig.parameters else {}
    params = {
      name:
        str(param.default)
        if type(param.default) in [str, int, float]
        else None
      for name, param in sig.parameters.items()
      if name not in ['G', 'backend', 'backend_kwargs']}
    for k, v in params.items():
      if sig.parameters[k].annotation is inspect._empty and v is None:
        # No annotation, no default — we must guess the type.
        if len(k) == 1:
          params[k] = 1
    if name == 'ladder_graph':
      print(params)
    name = "NX › " + name.replace('_', ' ').title()
    op = ops.Op(wrapped(func), name, params=params, inputs=inputs, outputs={'output': 'yes'}, type='basic')
    ops.ALL_OPS[name] = op
