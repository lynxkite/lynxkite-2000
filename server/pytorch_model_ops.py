'''Boxes for defining and using PyTorch models.'''
from enum import Enum
import inspect
from . import ops

LAYERS = {}

@ops.op("Define PyTorch model", sub_nodes=LAYERS)
def define_pytorch_model(*, sub_flow):
  print('sub_flow:', sub_flow)
  return ops.Bundle(other={'model': str(sub_flow)})

@ops.op("Train PyTorch model")
def train_pytorch_model(model, graph):
  # import torch # Lazy import because it's slow.
  return 'hello ' + str(model)

def register_layer(name):
  def decorator(func):
    sig = inspect.signature(func)
    inputs = {
      name: ops.Input(name=name, type=param.annotation, position='bottom')
      for name, param in sig.parameters.items()
      if param.kind != param.KEYWORD_ONLY}
    params = {
      name: ops.Parameter.basic(name, param.default, param.annotation)
      for name, param in sig.parameters.items()
      if param.kind == param.KEYWORD_ONLY}
    outputs = {'x': ops.Output(name='x', type='tensor', position='top')}
    LAYERS[name] = ops.Op(func=func, name=name, params=params, inputs=inputs, outputs=outputs)
    return func
  return decorator

@register_layer('LayerNorm')
def layernorm(x):
  return 'LayerNorm'

@register_layer('Dropout')
def dropout(x, *, p=0.5):
  return f'Dropout ({p})'

@register_layer('Linear')
def linear(*, output_dim: int):
  return f'Linear {output_dim}'

class GraphConv(Enum):
  GCNConv = 'GCNConv'
  GATConv = 'GATConv'
  GATv2Conv = 'GATv2Conv'
  SAGEConv = 'SAGEConv'

@register_layer('Graph Convolution')
def graph_convolution(x, edges, *, type: GraphConv):
  return 'GraphConv'

class Nonlinearity(Enum):
  Mish = 'Mish'
  ReLU = 'ReLU'
  Tanh = 'Tanh'

@register_layer('Nonlinearity')
def nonlinearity(x, *, type: Nonlinearity):
  return 'ReLU'

def register_area(name, params=[]):
  '''A node that represents an area. It can contain other nodes, but does not restrict movement in any way.'''
  op = ops.Op(func=ops.no_op, name=name, params={p.name: p for p in params}, inputs={}, outputs={}, type='area')
  LAYERS[name] = op

register_area('Repeat', params=[ops.Parameter.basic('times', 1, int)])
