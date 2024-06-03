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

CATALOGS = {}
EXECUTORS = {}

typeof = type # We have some arguments called "type".
def type_to_json(t):
  if isinstance(t, type) and issubclass(t, enum.Enum):
    return {'enum': list(t.__members__.keys())}
  if getattr(t, '__metadata__', None):
    return t.__metadata__[-1]
  return {'type': str(t)}
Type = Annotated[
  typing.Any, pydantic.PlainSerializer(type_to_json, return_type=dict)
]
LongStr = Annotated[
  str, {'format': 'textarea'}
]
PathStr = Annotated[
  str, {'format': 'path'}
]
CollapsedStr = Annotated[
  str, {'format': 'collapsed'}
]
class BaseConfig(pydantic.BaseModel):
  model_config = pydantic.ConfigDict(
    arbitrary_types_allowed=True,
  )


class Parameter(BaseConfig):
  '''Defines a parameter for an operation.'''
  name: str
  default: typing.Any
  type: Type = None

  @staticmethod
  def options(name, options, default=None):
    e = enum.Enum(f'OptionsFor_{name}', options)
    return Parameter.basic(name, e[default or options[0]], e)

  @staticmethod
  def collapsed(name, default, type=None):
    return Parameter.basic(name, default, CollapsedStr)

  @staticmethod
  def basic(name, default=None, type=None):
    if default is inspect._empty:
      default = None
    if type is None or type is inspect._empty:
      type = typeof(default) if default else None
    return Parameter(name=name, default=default, type=type)

class Input(BaseConfig):
  name: str
  type: Type
  position: str = 'left'

class Output(BaseConfig):
  name: str
  type: Type
  position: str = 'right'

MULTI_INPUT = Input(name='multi', type='*')
def basic_inputs(*names):
  return {name: Input(name=name, type=None) for name in names}
def basic_outputs(*names):
  return {name: Output(name=name, type=None) for name in names}


class Op(BaseConfig):
  func: typing.Callable = pydantic.Field(exclude=True)
  name: str
  params: dict[str, Parameter]
  inputs: dict[str, Input]
  outputs: dict[str, Output]
  type: str = 'basic' # The UI to use for this operation.
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
    for i, (x, p) in enumerate(zip(inputs, self.inputs.values())):
      if p.type == nx.Graph and isinstance(x, Bundle):
        inputs[i] = x.to_nx()
      elif p.type == Bundle and isinstance(x, nx.Graph):
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


def op(env: str, name: str, *, view='basic', sub_nodes=None):
  '''Decorator for defining an operation.'''
  def decorator(func):
    sig = inspect.signature(func)
    # Positional arguments are inputs.
    inputs = {
      name: Input(name=name, type=param.annotation)
      for name, param in sig.parameters.items()
      if param.kind != param.KEYWORD_ONLY}
    params = {}
    for n, param in sig.parameters.items():
      if param.kind == param.KEYWORD_ONLY:
        params[n] = Parameter.basic(n, param.default, param.annotation)
    outputs = {'output': Output(name='output', type=None)} if view == 'basic' else {} # Maybe more fancy later.
    op = Op(func=func, name=name, params=params, inputs=inputs, outputs=outputs, type=view)
    if sub_nodes is not None:
      op.sub_nodes = sub_nodes
      op.type = 'sub_flow'
    CATALOGS.setdefault(env, {})
    CATALOGS[env][name] = op
    func.__op__ = op
    return func
  return decorator

def input_position(**kwargs):
  '''Decorator for specifying unusual positions for the inputs.'''
  def decorator(func):
    op = func.__op__
    for k, v in kwargs.items():
      op.inputs[k].position = v
    return func
  return decorator

def output_position(**kwargs):
  '''Decorator for specifying unusual positions for the outputs.'''
  def decorator(func):
    op = func.__op__
    for k, v in kwargs.items():
      op.outputs[k].position = v
    return func
  return decorator

def no_op(*args, **kwargs):
  if args:
    return args[0]
  return Bundle()

def register_passive_op(env: str, name: str, inputs=[], outputs=['output'], params=[]):
  '''A passive operation has no associated code.'''
  op = Op(
    func=no_op,
    name=name,
    params={p.name: p for p in params},
    inputs=dict(
      (i, Input(name=i, type=None)) if isinstance(i, str)
      else (i.name, i) for i in inputs),
    outputs=dict(
      (o, Output(name=o, type=None)) if isinstance(o, str)
      else (o.name, o) for o in outputs))
  CATALOGS.setdefault(env, {})
  CATALOGS[env][name] = op
  return op

def register_executor(env: str):
  '''Decorator for registering an executor.'''
  def decorator(func):
    EXECUTORS[env] = func
    return func
  return decorator

def op_registration(env: str):
  return functools.partial(op, env)

def passive_op_registration(env: str):
  return functools.partial(register_passive_op, env)
