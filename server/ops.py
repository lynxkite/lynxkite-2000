'''API for implementing LynxKite operations.'''
import dataclasses
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
        if t == int:
          params[p] = int(params[p])
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
class EdgeDefinition:
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
  edges: list[EdgeDefinition]

  @classmethod
  def from_nx(cls, graph: nx.Graph):
    edges = nx.to_pandas_edgelist(graph)
    nodes = pd.DataFrame({'id': list(graph.nodes)})
    return cls(
      dfs={'edges': edges, 'nodes': nodes},
      edges=[
        EdgeDefinition(
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


def op(name):
  '''Decorator for defining an operation.'''
  def decorator(func):
    type = func.__annotations__.get('return') or 'basic'
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
    outputs = {'output': 'yes'} if type == 'basic' else {} # Maybe more fancy later.
    op = Op(func, name, params=params, inputs=inputs, outputs=outputs, type=type)
    ALL_OPS[name] = op
    return func
  return decorator
