'''API for implementing LynxKite operations.'''
import dataclasses
import functools
import inspect
import networkx as nx
import pandas as pd

ALL_OPS = {}

@dataclasses.dataclass
class Op:
  func: callable
  name: str
  params: dict
  inputs: dict
  outputs: dict
  type: str

  def __call__(self, *inputs, **params):
    # Convert parameters.
    sig = inspect.signature(self.func)
    for p in params:
      if p in self.params:
        t = sig.parameters[p].annotation
        if t is inspect._empty:
          t = type(self.params[p])
        if t == int:
          params[p] = int(params[p])
        elif t == float:
          params[p] = float(params[p])
    # Convert inputs.
    inputs = list(inputs)
    for i, (x, p) in enumerate(zip(inputs, sig.parameters.values())):
      t = p.annotation
      if t == nx.Graph and isinstance(x, Bundle):
        inputs[i] = o.to_nx()
      elif t == Bundle and isinstance(x, nx.Graph):
        inputs[i] = Bundle.from_nx(x)
    res = self.func(*inputs, **params)
    return res

@dataclasses.dataclass
class RelationDefinition:
  df: str
  source_column: str
  target_column: str
  source_table: str
  target_table: str
  source_key: str
  target_key: str

@dataclasses.dataclass
class Bundle:
  dfs: dict
  relations: list[RelationDefinition]

  @classmethod
  def from_nx(cls, graph: nx.Graph):
    edges = nx.to_pandas_edgelist(graph)
    d = dict(graph.nodes(data=True))
    nodes = pd.DataFrame(d.values(), index=d.keys())
    nodes['id'] = nodes.index
    return cls(
      dfs={'edges': edges, 'nodes': nodes},
      relations=[
        RelationDefinition(
          df='edges',
          source_column='source',
          target_column='target',
          source_table='nodes',
          target_table='nodes',
          source_key='id',
          target_key='id',
        )
      ]
    )

  def to_nx(self):
    graph = nx.from_pandas_edgelist(self.dfs['edges'])
    nx.set_node_attributes(graph, self.dfs['nodes'].set_index('id').to_dict('index'))
    return graph


def nx_node_attribute_func(name):
  '''Decorator for wrapping a function that adds a NetworkX node attribute.'''
  def decorator(func):
    @functools.wraps(func)
    def wrapper(graph: nx.Graph, **kwargs):
      graph = graph.copy()
      attr = func(graph, **kwargs)
      nx.set_node_attributes(graph, attr, name)
      return graph
    return wrapper
  return decorator


def op(name, *, view='basic'):
  '''Decorator for defining an operation.'''
  def decorator(func):
    sig = inspect.signature(func)
    # Positional arguments are inputs.
    inputs = {
      name: param.annotation
      for name, param in sig.parameters.items()
      if param.kind != param.KEYWORD_ONLY}
    params = {
      name: param.default if param.default is not inspect._empty else None
      for name, param in sig.parameters.items()
      if param.kind == param.KEYWORD_ONLY}
    outputs = {'output': 'yes'} if view == 'basic' else {} # Maybe more fancy later.
    op = Op(func, name, params=params, inputs=inputs, outputs=outputs, type=view)
    ALL_OPS[name] = op
    return func
  return decorator
