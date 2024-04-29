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
      name: param.annotation
      for name, param in sig.parameters.items()
      if param.kind != param.KEYWORD_ONLY}
    params = {
      name: param.default if param.default is not inspect._empty else None
      for name, param in sig.parameters.items()
      if param.kind == param.KEYWORD_ONLY}
    outputs = {'x': 'tensor'}
    LAYERS[name] = ops.Op(func, name, params=params, inputs=inputs, outputs=outputs, type='vertical')
    return func
  return decorator

@register_layer('LayerNorm')
def normalization():
  return 'LayerNorm'

@register_layer('Dropout')
def dropout(*, p=0.5):
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
def graph_convolution(*, type: GraphConv):
  return 'GraphConv'

class Nonlinearity(Enum):
  Mish = 'Mish'
  ReLU = 'ReLU'
  Tanh = 'Tanh'

@register_layer('Nonlinearity')
def nonlinearity(*, type: Nonlinearity):
  return 'ReLU'

