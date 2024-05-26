'''API for implementing LynxKite operations.'''
from __future__ import annotations
import dataclasses
import enum
import functools
import inspect
import networkx as nx
import pandas as pd
import pydantic
import typing
from typing_extensions import Annotated

ALL_OPS = {}
typeof = type # We have some arguments called "type".
def type_to_json(t):
  if isinstance(t, type) and issubclass(t, enum.Enum):
    return {'enum': list(t.__members__.keys())}
  if isinstance(t, tuple) and t[0] == 'collapsed':
    return {'collapsed': str(t[1])}
  return {'type': str(t)}
Type = Annotated[
  typing.Any, pydantic.PlainSerializer(type_to_json, return_type=dict)
]
class BaseConfig(pydantic.BaseModel):
  model_config = pydantic.ConfigDict(
    arbitrary_types_allowed=True,
  )


class Parameter(BaseConfig):
  '''Defines a parameter for an operation.'''
  name: str
  default: any
  type: Type = None

  @staticmethod
  def options(name, options, default=None):
    e = enum.Enum(f'OptionsFor_{name}', options)
    return Parameter.basic(name, e[default or options[0]], e)

  @staticmethod
  def collapsed(name, default, type=None):
    return Parameter.basic(name, default, ('collapsed', type or typeof(default)))

  @staticmethod
  def basic(name, default=None, type=None):
    if default is inspect._empty:
      default = None
    if type is None or type is inspect._empty:
      type = typeof(default) if default else None
    return Parameter(name=name, default=default, type=type)


class Op(BaseConfig):
  func: callable = pydantic.Field(exclude=True)
  name: str
  params: dict[str, Parameter]
  inputs: dict[str, Type] # name -> type
  outputs: dict[str, Type] # name -> type
  type: str # The UI to use for this operation.
  sub_nodes: list[Op] = None # If set, these nodes can be placed inside the operation's node.

  def __call__(self, *inputs, **params):
    # Convert parameters.
    for p in params:
      if p in self.params:
        if self.params[p].type == int:
          params[p] = int(params[p])
        elif self.params[p].type == float:
          params[p] = float(params[p])
    # Convert inputs.
    inputs = list(inputs)
    for i, (x, t) in enumerate(zip(inputs, self.inputs.values())):
      if t == nx.Graph and isinstance(x, Bundle):
        inputs[i] = x.to_nx()
      elif t == Bundle and isinstance(x, nx.Graph):
        inputs[i] = Bundle.from_nx(x)
    res = self.func(*inputs, **params)
    return res


@dataclasses.dataclass
class RelationDefinition:
  '''Defines a set of edges.'''
  df: str # The DataFrame that contains the edges.
  source_column: str # The column in the edge DataFrame that contains the source node ID.
  target_column: str # The column in the edge DataFrame that contains the target node ID.
  source_table: str # The DataFrame that contains the source nodes.
  target_table: str # The DataFrame that contains the target nodes.
  source_key: str # The column in the source table that contains the node ID.
  target_key: str # The column in the target table that contains the node ID.

@dataclasses.dataclass
class Bundle:
  '''A collection of DataFrames and other data.

  Can efficiently represent a knowledge graph (homogeneous or heterogeneous) or tabular data.
  It can also carry other data, such as a trained model.
  '''
  dfs: dict[str, pd.DataFrame] = dataclasses.field(default_factory=dict)
  relations: list[RelationDefinition] = dataclasses.field(default_factory=list)
  other: dict[str, typing.Any] = None

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


def op(name, *, view='basic', sub_nodes=None):
  '''Decorator for defining an operation.'''
  def decorator(func):
    sig = inspect.signature(func)
    # Positional arguments are inputs.
    inputs = {
      name: param.annotation
      for name, param in sig.parameters.items()
      if param.kind != param.KEYWORD_ONLY}
    params = {}
    for n, param in sig.parameters.items():
      if param.kind == param.KEYWORD_ONLY:
        params[n] = Parameter.basic(n, param.default, param.annotation)
    outputs = {'output': 'yes'} if view == 'basic' else {} # Maybe more fancy later.
    op = Op(func=func, name=name, params=params, inputs=inputs, outputs=outputs, type=view)
    if sub_nodes is not None:
      op.sub_nodes = sub_nodes
      op.type = 'sub_flow'
    ALL_OPS[name] = op
    return func
  return decorator

def no_op(*args, **kwargs):
  if args:
    return args[0]
  return Bundle()

def register_passive_op(name, inputs={'input': Bundle}, outputs={'output': Bundle}, params=[]):
  '''A passive operation has no associated code.'''
  op = Op(no_op, name, params={p.name: p for p in params}, inputs=inputs, outputs=outputs, type='basic')
  ALL_OPS[name] = op
  return op

def register_area(name, params=[]):
  '''A node that represents an area. It can contain other nodes, but does not restrict movement in any way.'''
  op = register_passive_op(name, params=params, inputs={}, outputs={})
  op.type = 'area'
